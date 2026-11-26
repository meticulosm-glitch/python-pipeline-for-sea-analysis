from __future__ import annotations

import datetime as dt
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from prefect import flow, get_run_logger, task

CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CDSE_ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
CDSE_DOWNLOAD_URL = (
    "https://download.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
)
CDSE_S3_ENDPOINT = "https://eodata.dataspace.copernicus.eu"
CDSE_S3_BUCKET = "EODATA"

_REDIRECT_STATUS = {301, 302, 303, 307, 308}
_MIN_ZIP_BYTES = 1_000_000


@dataclass
class ProductRef:
    product_id: str
    name: str                
    collection: str          
    sensing_date: dt.date
    s3_prefix: str            
    cloud_cover: float | None = None
    overlap_ratio: float = 0.0


@dataclass
class IngestedProduct:
    product: ProductRef
    band_paths: dict[str, str]
    metadata_path: str
    tile_metadata_path: str

@dataclass
class CdseToken:
    access_token: str
    refresh_token: str = ""
    expires_at: float = 0.0

    def expired(self, skew_s: float = 60.0) -> bool:
        return time.time() >= (self.expires_at - skew_s)


_token_lock = threading.Lock()
_cached_token: CdseToken | None = None


def _token_from_response(data: dict) -> CdseToken:
    return CdseToken(
        access_token=data["access_token"],
        refresh_token=data.get("refresh_token", "") or "",
        expires_at=time.time() + float(data.get("expires_in", 600)),
    )


def _password_grant() -> CdseToken:
    resp = requests.post(
        CDSE_TOKEN_URL,
        data={
            "client_id": "cdse-public",
            "grant_type": "password",
            "username": os.environ["CDSE_USERNAME"],
            "password": os.environ["CDSE_PASSWORD"],
        },
        timeout=30,
    )
    resp.raise_for_status()
    return _token_from_response(resp.json())


def _refresh_grant(refresh_token: str) -> CdseToken:
    resp = requests.post(
        CDSE_TOKEN_URL,
        data={
            "client_id": "cdse-public",
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return _token_from_response(resp.json())


def get_valid_cdse_token(force: bool = False) -> CdseToken:
    global _cached_token
    with _token_lock:
        if not force and _cached_token is not None and not _cached_token.expired():
            return _cached_token
        if _cached_token is not None and _cached_token.refresh_token:
            try:
                _cached_token = _refresh_grant(_cached_token.refresh_token)
                return _cached_token
            except requests.HTTPError:
                pass
        _cached_token = _password_grant()
        return _cached_token


@task(retries=3, retry_delay_seconds=10)
def get_cdse_token() -> str:
    return get_valid_cdse_token().access_token


def _get_with_bearer(url: str, token: CdseToken, *, stream: bool,
                     timeout: tuple[int, int] = (30, 120)) -> requests.Response:
    headers = {"Authorization": f"Bearer {token.access_token}"}
    resp = requests.get(
        url, headers=headers, stream=stream,
        allow_redirects=False, timeout=timeout,
    )
    hops = 0
    while resp.status_code in _REDIRECT_STATUS and hops < 8:
        loc = resp.headers.get("Location")
        resp.close()
        if not loc:
            break
        hops += 1
        resp = requests.get(
            loc, headers=headers, stream=stream,
            allow_redirects=False, timeout=timeout,
        )
    return resp

def _overlap_ratio(geojson: dict | None,
                   bbox: tuple[float, float, float, float]) -> float:
    if not geojson:
        return 0.0
    coords: list[tuple[float, float]] = []

    def _walk(node):
        if not node:
            return
        if isinstance(node[0], (int, float)):
            coords.append((float(node[0]), float(node[1])))
            return
        for child in node:
            _walk(child)

    _walk(geojson.get("coordinates", []))
    if not coords:
        return 0.0
    xs, ys = zip(*coords)
    fb = (min(xs), min(ys), max(xs), max(ys))
    q_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    if q_area <= 0:
        return 0.0
    minx, miny = max(fb[0], bbox[0]), max(fb[1], bbox[1])
    maxx, maxy = min(fb[2], bbox[2]), min(fb[3], bbox[3])
    if maxx <= minx or maxy <= miny:
        return 0.0
    return ((maxx - minx) * (maxy - miny)) / q_area


def _cloud_cover_from_item(item: dict) -> float | None:
    for att in item.get("Attributes") or []:
        if att.get("Name") == "cloudCover":
            try:
                return float(att.get("Value"))
            except (TypeError, ValueError):
                return None
    return None


@task(retries=3, retry_delay_seconds=10)
def search_s2_products(bbox: tuple[float, float, float, float],
                        start_date: dt.date, end_date: dt.date,
                        product_type: str = "MSIL1C",
                        max_cloud_cover: float = 20.0) -> list[ProductRef]:
    minx, miny, maxx, maxy = bbox
    polygon = (f"POLYGON(({minx} {miny},{maxx} {miny},{maxx} {maxy},"
               f"{minx} {maxy},{minx} {miny}))")

    filt = (
        f"Collection/Name eq 'SENTINEL-2' and "
        f"contains(Name,'{product_type}') and "
        f"Online eq true and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{polygon}') and "
        f"ContentDate/Start gt {start_date.isoformat()}T00:00:00.000Z and "
        f"ContentDate/Start lt {end_date.isoformat()}T00:00:00.000Z and "
        f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' "
        f"and att/OData.CSC.DoubleAttribute/Value le {max_cloud_cover})"
    )
    resp = requests.get(
        CDSE_ODATA_URL,
        params={
            "$filter": filt,
            "$top": 50,
            "$orderby": "ContentDate/Start desc",
            "$expand": "Attributes",
        },
        timeout=60,
    )
    resp.raise_for_status()

    products: list[ProductRef] = []
    for item in resp.json().get("value", []):
        name = item["Name"].removesuffix(".SAFE")
        sensing_date = dt.datetime.fromisoformat(
            item["ContentDate"]["Start"].replace("Z", "+00:00")
        ).date()
        products.append(ProductRef(
            product_id=item["Id"], name=name, collection="SENTINEL-2",
            sensing_date=sensing_date,
            s3_prefix=_s3_prefix_from_name(name, sensing_date),
            cloud_cover=_cloud_cover_from_item(item),
            overlap_ratio=_overlap_ratio(item.get("GeoFootprint"), bbox),
        ))

    products.sort(key=lambda p: (
        -p.overlap_ratio,
        p.cloud_cover if p.cloud_cover is not None else 999.0,
        -p.sensing_date.toordinal(),
    ))
    return products


def _s3_prefix_from_name(product_name: str, sensing_date: dt.date) -> str:
    level = "L1C" if "MSIL1C" in product_name else "L2A"
    return (f"Sentinel-2/MSI/{level}/{sensing_date:%Y}/{sensing_date:%m}/"
            f"{sensing_date:%d}/{product_name}.SAFE")

def configure_gdal_s3_env() -> None:
    os.environ["AWS_S3_ENDPOINT"] = "eodata.dataspace.copernicus.eu"
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ["CDSE_S3_ACCESS_KEY"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["CDSE_S3_SECRET_KEY"]
    os.environ["AWS_VIRTUAL_HOSTING"] = "FALSE"
    os.environ["AWS_HTTPS"] = "YES"


@task(retries=3, retry_delay_seconds=15)
def list_product_via_s3(product: ProductRef) -> list[str]:
    import boto3
    client = boto3.client(
        "s3", endpoint_url=CDSE_S3_ENDPOINT,
        aws_access_key_id=os.environ["CDSE_S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["CDSE_S3_SECRET_KEY"],
    )
    prefix = product.s3_prefix.rstrip("/") + "/"
    paginator = client.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=CDSE_S3_BUCKET, Prefix=prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []))
    return keys

list_granule_via_s3 = list_product_via_s3


@task
def resolve_band_paths(keys: list[str], band_ids: list[str]) -> dict[str, str]:
    paths = {}
    for band_id in band_ids:
        matches = [k for k in keys if k.endswith(f"_{band_id}.jp2") and "IMG_DATA" in k]
        if not matches:
            raise FileNotFoundError(f"Banda {band_id} non trovata nel listing S3.")
        paths[band_id] = f"/vsis3/{CDSE_S3_BUCKET}/{matches[0]}"
    return paths


@task
def resolve_metadata_path(keys: list[str]) -> str:
    matches = [k for k in keys if k.endswith("MTD_MSIL1C.xml")]
    if not matches:
        raise FileNotFoundError("MTD_MSIL1C.xml non trovato nel listing S3.")
    return f"/vsis3/{CDSE_S3_BUCKET}/{matches[0]}"


@task
def resolve_tile_metadata_path(keys: list[str]) -> str:
    matches = [k for k in keys if k.endswith("MTD_TL.xml")]
    if not matches:
        raise FileNotFoundError("MTD_TL.xml non trovato nel listing S3.")
    return f"/vsis3/{CDSE_S3_BUCKET}/{matches[0]}"

def _safe_dir_if_complete(dest_dir: str, product: ProductRef) -> str | None:
    candidate = Path(dest_dir) / f"{product.name}.SAFE"
    if candidate.is_dir() and (candidate / "MTD_MSIL1C.xml").is_file():
        return str(candidate)
    return None


def _zip_if_complete(dest_dir: str, product: ProductRef) -> str | None:
    candidate = Path(dest_dir) / f"{product.name}.zip"
    if candidate.is_file() and candidate.stat().st_size > _MIN_ZIP_BYTES:
        return str(candidate)
    return None


def _avanzamento(scaricati: int, totale: int, ultimo: int) -> int:
    mb = scaricati / 1e6
    if totale:
        pct = scaricati * 100 // totale
        if pct != ultimo:
            print(f"\rDownload: {pct:3d}% ({mb:.0f}/{totale / 1e6:.0f} MB)",
                  end="", flush=True)
        return pct
    passo = int(mb // 10)
    if passo != ultimo:
        print(f"\rDownload: {mb:.0f} MB", end="", flush=True)
    return passo


@task(retries=2, retry_delay_seconds=30)
def download_product_odata(product: ProductRef, dest_dir: str) -> str:

    dest_path = Path(dest_dir) / f"{product.name}.zip"
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest_path.with_suffix(".zip.part")

    url = CDSE_DOWNLOAD_URL.format(product_id=product.product_id)
    token = get_valid_cdse_token()
    resp = _get_with_bearer(url, token, stream=True)
    if resp.status_code == 401:
        resp.close()
        token = get_valid_cdse_token(force=True)
        resp = _get_with_bearer(url, token, stream=True)

    try:
        if resp.status_code == 202:
            raise RuntimeError(
                f"Prodotto {product.name} non è Online su CDSE (HTTP 202). "
                "Attendi il retrieve oppure filtra `Online eq true`."
            )
        resp.raise_for_status()
        totale = int(resp.headers.get("Content-Length") or 0)
        scaricati, ultimo = 0, -1
        with open(part_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
                    scaricati += len(chunk)
                    ultimo = _avanzamento(scaricati, totale, ultimo)
        print()
    finally:
        resp.close()

    if part_path.stat().st_size < _MIN_ZIP_BYTES:
        part_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Download di {product.name} troppo piccolo "
            f"({part_path} < {_MIN_ZIP_BYTES} byte): token/redirect non validi."
        )
    part_path.replace(dest_path)
    return str(dest_path)


@task
def extract_product(zip_path: str, dest_dir: str) -> str:
    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
        safe_folder_name = zf.namelist()[0].split("/")[0]

    return str(Path(dest_dir) / safe_folder_name)


@task
def resolve_band_paths_local(safe_dir: str, band_ids: list[str]) -> dict[str, str]:
    paths = {}
    for band_id in band_ids:
        matches = list(Path(safe_dir).glob(f"GRANULE/*/IMG_DATA/*_{band_id}.jp2"))
        if not matches:
            raise FileNotFoundError(f"Banda {band_id} non trovata in {safe_dir}.")
        paths[band_id] = str(matches[0])
    return paths


@task
def resolve_metadata_path_local(safe_dir: str) -> str:
    matches = list(Path(safe_dir).glob("MTD_MSIL1C.xml"))
    if not matches:
        raise FileNotFoundError(f"MTD_MSIL1C.xml non trovato in {safe_dir}.")
    return str(matches[0])


@task
def resolve_tile_metadata_path_local(safe_dir: str) -> str:
    matches = list(Path(safe_dir).glob("GRANULE/*/MTD_TL.xml"))
    if not matches:
        raise FileNotFoundError(f"MTD_TL.xml non trovato in {safe_dir}.")
    return str(matches[0])


@flow(name="ingestion-sentinel2")
def ingestion_flow(bbox: tuple[float, float, float, float],
                    start_date: dt.date, end_date: dt.date,
                    band_ids: list[str],
                    product_type: str = "MSIL1C",
                    max_cloud_cover: float = 20.0,
                    use_s3: bool = True,
                    local_dest_dir: str = "./data/raw",
                    max_products: int | None = None) -> list[IngestedProduct]:
    logger = get_run_logger()

    products = search_s2_products(
        bbox, start_date, end_date, product_type, max_cloud_cover,
    )
    logger.info(f"Trovati {len(products)} prodotti {product_type}")

    if max_products is not None:
        products = products[: max(0, max_products)]
        logger.info(f"Limitati a {len(products)} prodotti (max_products={max_products})")

    ingested: list[IngestedProduct] = []

    if use_s3:
        configure_gdal_s3_env()
        for product in products:
            keys = list_product_via_s3(product)
            band_paths = resolve_band_paths(keys, band_ids)
            metadata_path = resolve_metadata_path(keys)
            tile_metadata_path = resolve_tile_metadata_path(keys)
            ingested.append(IngestedProduct(product, band_paths, metadata_path,
                                             tile_metadata_path))
    else:
        for i, product in enumerate(products, start=1):
            logger.info(f"[{i}/{len(products)}] {product.name}")
            safe_dir = _safe_dir_if_complete(local_dest_dir, product)
            if safe_dir:
                logger.info(f"Già estratto in locale, skip download: {safe_dir}")
            else:
                zip_path = _zip_if_complete(local_dest_dir, product)
                if zip_path:
                    logger.info(f"Zip già presente, estraggo: {zip_path}")
                else:
                    zip_path = download_product_odata(product, local_dest_dir)
                safe_dir = extract_product(zip_path, local_dest_dir)
            band_paths = resolve_band_paths_local(safe_dir, band_ids)
            metadata_path = resolve_metadata_path_local(safe_dir)
            tile_metadata_path = resolve_tile_metadata_path_local(safe_dir)
            ingested.append(IngestedProduct(product, band_paths, metadata_path,
                                             tile_metadata_path))
            logger.info(f"Estratto e risolto: {safe_dir}")

    return ingested


if __name__ == "__main__":
    ingestion_flow(
        bbox=(14.20, 40.75, 14.35, 40.85),
        start_date=dt.date(2026, 6, 1),
        end_date=dt.date(2026, 6, 30),
        band_ids=["B03", "B04", "B05", "B08"],
        product_type="MSIL1C",
        max_cloud_cover=10.0,
        use_s3=True,
        max_products=1,
    )
