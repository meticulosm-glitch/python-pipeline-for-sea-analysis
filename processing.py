from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import rasterio
from prefect import flow, get_run_logger, task
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds
from scipy import stats

from Py6S import (
    AeroProfile, AtmosCorr, AtmosProfile, Geometry, GroundReflectance,
    PredefinedWavelengths, SixS, Wavelength,
)

from ingestion import CDSE_S3_ENDPOINT, IngestedProduct


@dataclass
class S2Band:
    band_id: str
    name: str
    center_um: float
    fwhm_um: float
    resolution_m: int


S2_WATER_BANDS = [
    S2Band("B03", "Green", 0.560, 0.035, 10),
    S2Band("B04", "Red",   0.665, 0.030, 10),
    S2Band("B05", "Red edge 1", 0.705, 0.015, 20),
    S2Band("B08", "NIR",   0.842, 0.115, 10),
]
_BANDS_BY_ID = {b.band_id: b for b in S2_WATER_BANDS}


@dataclass
class S2L1CMetadata:
    quantification_value: float
    radio_add_offset: dict[str, float]
    solar_irradiance: dict[str, float]
    u_earth_sun_distance: float
    sun_zenith_deg: float
    sun_azimuth_deg: float
    view_zenith_deg: dict[str, float]
    view_azimuth_deg: dict[str, float]
    sensing_datetime: dt.datetime


def _local(tag: str) -> str:
    return tag.split("}")[-1]


def _find_all(root: ET.Element, name: str) -> list[ET.Element]:
    return [el for el in root.iter() if _local(el.tag) == name]


def _find_one(root: ET.Element, name: str) -> ET.Element:
    matches = _find_all(root, name)
    if not matches:
        raise ValueError(f"Tag '{name}' non trovato nell'XML dei metadati.")
    return matches[0]


_BAND_ID_ORDER = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
                   "B08", "B8A", "B09", "B10", "B11", "B12"]


def _xml_root(mtd_path: str) -> ET.Element:
    if mtd_path.startswith("/vsis3/"):
        import boto3
        rest = mtd_path[len("/vsis3/"):]
        bucket, _, key = rest.partition("/")
        client = boto3.client(
            "s3",
            endpoint_url=CDSE_S3_ENDPOINT,
            aws_access_key_id=os.environ["CDSE_S3_ACCESS_KEY"],
            aws_secret_access_key=os.environ["CDSE_S3_SECRET_KEY"],
        )
        body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
        return ET.fromstring(body)
    return ET.parse(mtd_path).getroot()


@task
def parse_s2_l1c_metadata(mtd_path: str, tile_mtd_path: str) -> S2L1CMetadata:
    root = _xml_root(mtd_path)

    quant = float(_find_one(root, "QUANTIFICATION_VALUE").text)

    radio_offset = {b: 0.0 for b in _BAND_ID_ORDER}
    for el in _find_all(root, "RADIO_ADD_OFFSET"):
        radio_offset[_BAND_ID_ORDER[int(el.attrib["band_id"])]] = float(el.text)

    solar_irr = {}
    for el in _find_all(root, "SOLAR_IRRADIANCE"):
        solar_irr[_BAND_ID_ORDER[int(el.attrib["bandId"])]] = float(el.text)

    u_factor = float(_find_one(root, "U").text)

    sensing_str = _find_one(root, "PRODUCT_START_TIME").text
    sensing_dt = dt.datetime.fromisoformat(sensing_str.replace("Z", "+00:00"))

    tile_root = _xml_root(tile_mtd_path)

    sun_angle = _find_one(tile_root, "Mean_Sun_Angle")
    sun_zenith = float(_find_one(sun_angle, "ZENITH_ANGLE").text)
    sun_azimuth = float(_find_one(sun_angle, "AZIMUTH_ANGLE").text)

    view_zenith, view_azimuth = {}, {}
    for el in _find_all(tile_root, "Mean_Viewing_Incidence_Angle"):
        band_name = _BAND_ID_ORDER[int(el.attrib["bandId"])]
        view_zenith[band_name] = float(_find_one(el, "ZENITH_ANGLE").text)
        view_azimuth[band_name] = float(_find_one(el, "AZIMUTH_ANGLE").text)

    return S2L1CMetadata(
        quantification_value=quant, radio_add_offset=radio_offset,
        solar_irradiance=solar_irr, u_earth_sun_distance=u_factor,
        sun_zenith_deg=sun_zenith, sun_azimuth_deg=sun_azimuth,
        view_zenith_deg=view_zenith, view_azimuth_deg=view_azimuth,
        sensing_datetime=sensing_dt,
    )


def _clip_window(src, bbox: tuple[float, float, float, float] | None) -> Window:
    if bbox is None:
        return Window(0, 0, src.width, src.height)
    left, bottom, right, top = transform_bounds(
        "EPSG:4326", src.crs, bbox[0], bbox[1], bbox[2], bbox[3], densify_pts=21,
    )
    window = from_bounds(left, bottom, right, top, src.transform)
    window = window.intersection(Window(0, 0, src.width, src.height))
    return window.round_offsets().round_lengths()


@task
def load_water_bands(band_paths: dict[str, str],
                     bbox: tuple[float, float, float, float] | None = None
                     ) -> dict:
    with rasterio.open(band_paths["B04"]) as ref:
        window = _clip_window(ref, bbox)
        if window.width <= 0 or window.height <= 0:
            raise ValueError("La bbox non interseca la scena Sentinel-2.")
        ref_transform = ref.window_transform(window)
        ref_crs = ref.crs
        height, width = int(window.height), int(window.width)
        out = {"B04": ref.read(1, window=window)}

    for band_id in ("B03", "B05", "B08"):
        with rasterio.open(band_paths[band_id]) as src:
            dest = np.empty((height, width), dtype=src.dtypes[0])
            reproject(
                source=rasterio.band(src, 1),
                destination=dest,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear,
            )
            out[band_id] = dest

    return {"bands": out, "transform": ref_transform, "crs": ref_crs,
            "shape": (height, width)}


@task
def read_band(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


@task
def dn_to_toa_reflectance(dn: np.ndarray, band: S2Band, meta: S2L1CMetadata) -> np.ndarray:
    offset = meta.radio_add_offset.get(band.band_id, 0.0)
    return (dn.astype(np.float64) + offset) / meta.quantification_value


@task
def toa_reflectance_to_radiance(rho_toa: np.ndarray, band: S2Band,
                                 meta: S2L1CMetadata) -> np.ndarray:
    theta_s = np.radians(meta.sun_zenith_deg)
    esun = meta.solar_irradiance[band.band_id]
    return rho_toa * esun * np.cos(theta_s) * meta.u_earth_sun_distance / np.pi


@dataclass
class AtmosphericParams:
    aod_550: float
    aod_uncertainty: float | None = None
    aero_profile: int = AeroProfile.Maritime
    atmos_profile: int = AtmosProfile.MidlatitudeSummer
    source: str = "cams-eac4"


_EAC4_GRID_DEG = 0.75
_FORECAST_GRID_DEG = 0.4
_AOD_VAR_CANDIDATES = (
    "aod550", "aod_550", "total_aerosol_optical_depth_550nm",
)


def _area_nws_e(lat: float, lon: float, grid_deg: float,
                 min_cells: int = 2, extra_margin_deg: float = 0.0) -> list[float]:
    import math

    half = max(grid_deg * min_cells, extra_margin_deg)
    north = math.ceil((lat + half) / grid_deg) * grid_deg
    south = math.floor((lat - half) / grid_deg) * grid_deg
    west = math.floor((lon - half) / grid_deg) * grid_deg
    east = math.ceil((lon + half) / grid_deg) * grid_deg
    if north <= south:
        north = south + grid_deg * min_cells
    if east <= west:
        east = west + grid_deg * min_cells
    return [north, west, south, east]


def _eac4_likely_available(when: dt.datetime) -> bool:
    d = when.date() if isinstance(when, dt.datetime) else when
    if d.year < 2003 or d.year > 2025:
        return False
    return (dt.date.today() - d).days >= 150


def _load_cams_dataset(path: str):
    import zipfile

    import xarray as xr

    if zipfile.is_zipfile(path):
        extract_dir = path + "_unzip"
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(path) as zf:
            zf.extractall(extract_dir)
        ncs: list[str] = []
        for root, _, files in os.walk(extract_dir):
            ncs.extend(
                os.path.join(root, f)
                for f in files
                if f.endswith((".nc", ".nc4"))
            )
        if not ncs:
            raise RuntimeError(f"Nessun NetCDF nello zip CAMS {path}")
        return xr.open_mfdataset(ncs) if len(ncs) > 1 else xr.open_dataset(ncs[0])
    return xr.open_dataset(path)


def _as_datetime64_ns(values) -> np.ndarray:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[ns]")
    out = np.empty(arr.shape, dtype="datetime64[ns]")
    for idx, val in np.ndenumerate(arr):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            out[idx] = np.datetime64("NaT")
            continue
        out[idx] = np.datetime64(val, "ns")
    return out


def _isel_nearest_datetime(da, coord_name: str, when: dt.datetime):
    coord = da.coords[coord_name]
    when_ns = np.datetime64(when.replace(tzinfo=None), "ns")
    values = _as_datetime64_ns(coord.values)
    delta = np.abs(values.astype("int64") - when_ns.astype("int64"))
    delta = np.where(np.isnat(values), np.iinfo(np.int64).max, delta)
    idx = np.unravel_index(int(np.argmin(delta)), values.shape)
    return da.isel({dim: int(idx[i]) for i, dim in enumerate(coord.dims)})


def _select_nearest_time(da, when: dt.datetime):
    if "time" in da.coords and da.coords["time"].ndim == 1:
        when_np = np.datetime64(when.replace(tzinfo=None))
        return da.sel(time=when_np, method="nearest")
    if "valid_time" in da.coords:
        return _isel_nearest_datetime(da, "valid_time", when)
    if "forecast_reference_time" in da.coords and (
        "forecast_period" in da.coords or "step" in da.coords
    ):
        step_name = "forecast_period" if "forecast_period" in da.coords else "step"
        valid = da.coords["forecast_reference_time"] + da.coords[step_name]
        da = da.assign_coords(valid_time=valid)
        return _isel_nearest_datetime(da, "valid_time", when)
    if "forecast_reference_time" in da.coords:
        return _isel_nearest_datetime(da, "forecast_reference_time", when)
    return da


def _aod_at_point(ds, lat: float, lon: float, when: dt.datetime) -> float:
    var_name = next((n for n in _AOD_VAR_CANDIDATES if n in ds.data_vars), None)
    if var_name is None:
        data_vars = [n for n in ds.data_vars if ds[n].ndim >= 2]
        var_name = data_vars[0] if data_vars else list(ds.data_vars)[0]
    da = ds[var_name]

    lat_name = next((n for n in ("latitude", "lat") if n in da.coords), None)
    lon_name = next((n for n in ("longitude", "lon") if n in da.coords), None)
    if lat_name is None or lon_name is None:
        raise ValueError(f"Coordinate lat/lon assenti in {list(da.coords)}")
    da = da.sel({lat_name: lat, lon_name: lon}, method="nearest")
    da = _select_nearest_time(da, when)

    for extra in ("number", "level", "isobaricInhPa"):
        if extra in da.dims:
            da = da.isel({extra: 0})

    val = np.asarray(da.values, dtype=np.float64).squeeze()
    if val.ndim != 0:
        val = np.nanmean(val)
    if not np.isfinite(val):
        raise ValueError(f"AOD non finito dal NetCDF CAMS (var={var_name}).")
    return float(val)


class CamsLicenceError(RuntimeError):
    """Licenza ADS non accettata: ritentare è inutile finché non si accetta sul sito."""


def _http_status(exc: BaseException) -> int | None:
    resp = getattr(exc, "response", None)
    return getattr(resp, "status_code", None)


def _licence_url(dataset: str) -> str:
    return (
        f"https://ads.atmosphere.copernicus.eu/datasets/{dataset}"
        "?tab=download#manage-licences"
    )


def _raise_if_licence(dataset: str, exc: BaseException) -> None:
    text = str(exc).lower()
    if _http_status(exc) == 403 or "licen" in text:
        raise CamsLicenceError(
            f"Licenza ADS non accettata per `{dataset}`.\n"
            f"1. Apri {_licence_url(dataset)}\n"
            "2. Accedi con lo STESSO account di ~/.cdsapirc\n"
            "3. Accetta la licence (una tantum) e rilancia lo script.\n"
            "Senza questo ADS risponde 403."
        ) from exc


def _should_retry_cams(task, task_run, state) -> bool:
    try:
        state.result()
    except (CamsLicenceError, ValueError):
        return False
    except Exception:
        return True
    return False


@task(retries=2, retry_delay_seconds=30, retry_condition_fn=_should_retry_cams)
def get_aod_from_cams(lat: float, lon: float, when: dt.datetime,
                       aod_uncertainty: float,
                       area_margin_deg: float = 0.5) -> AtmosphericParams:
    import cdsapi
    import requests

    try:
        logger = get_run_logger()
    except Exception:
        logger = None

    def _log(msg: str) -> None:
        if logger is not None:
            logger.info(msg)

    client = cdsapi.Client(quiet=True)
    day = when.strftime("%Y-%m-%d")
    cache_dir = Path("data/cams")
    cache_dir.mkdir(parents=True, exist_ok=True)

    eac4_request = {
        "variable": ["total_aerosol_optical_depth_550nm"],
        "date": day,
        "time": [f"{h:02d}:00" for h in (0, 3, 6, 9, 12, 15, 18, 21)],
        "data_format": "netcdf_zip",
        "area": _area_nws_e(lat, lon, _EAC4_GRID_DEG,
                            extra_margin_deg=area_margin_deg),
    }
    forecast_request = {
        "variable": ["total_aerosol_optical_depth_550nm"],
        "date": day,
        "time": ["00:00", "12:00"],
        "leadtime_hour": ["0", "3", "6", "9", "12"],
        "type": ["forecast"],
        "data_format": "netcdf_zip",
        "area": _area_nws_e(lat, lon, _FORECAST_GRID_DEG,
                            extra_margin_deg=area_margin_deg),
    }

    attempts: list[tuple[str, str, dict]] = []
    if _eac4_likely_available(when):
        attempts.append(("cams-eac4", "cams-global-reanalysis-eac4", eac4_request))
    attempts.append((
        "cams-forecast",
        "cams-global-atmospheric-composition-forecasts",
        forecast_request,
    ))
    if not _eac4_likely_available(when):
        _log(
            f"EAC4 non copre {day} (rianalisi 2003–2025, latenza 4–6 mesi): "
            "uso cams-global-atmospheric-composition-forecasts."
        )

    last_error: Exception | None = None
    for source, dataset, request in attempts:
        cached = cache_dir / f"{dataset}_{day}.zip"
        tmp_name = str(cached)
        try:
            if cached.is_file() and cached.stat().st_size > 1000:
                _log(f"Uso cache CAMS già scaricata: {cached}")
            else:
                print(f"CAMS: richiesta {dataset} in coda su ADS...", flush=True)
                client.retrieve(dataset, request, tmp_name)
            ds = _load_cams_dataset(tmp_name)
            try:
                aod_550 = _aod_at_point(ds, lat, lon, when)
            finally:
                ds.close()
            _log(f"AOD@550nm={aod_550:.4f} da {dataset} per {day} "
                 f"(sensing≈{when.strftime('%Y-%m-%d %H:%M')})")
            return AtmosphericParams(
                aod_550=aod_550,
                aod_uncertainty=aod_uncertainty,
                source=source,
            )
        except requests.HTTPError as exc:
            _raise_if_licence(dataset, exc)
            last_error = exc
            status = _http_status(exc) or "?"
            _log(f"{dataset} rifiutato ({status}): {exc}")
            continue
        except Exception as exc:
            _raise_if_licence(dataset, exc)
            last_error = exc
            _log(f"{dataset} fallito: {exc}")
            continue

    raise RuntimeError(
        f"Impossibile ottenere AOD CAMS per lat={lat}, lon={lon}, {day}. "
        "Verifica ~/.cdsapirc (URL ADS, non CDS) e che il dataset "
        "cams-global-atmospheric-composition-forecasts copra quella data."
    ) from last_error


def aod_da_aux_camsfo(path: str | Path, lat: float, lon: float,
                      variabile: str = "aod550") -> float:
    import eccodes

    with open(path, "rb") as f:
        while (h := eccodes.codes_grib_new_from_file(f)) is not None:
            try:
                if eccodes.codes_get(h, "shortName") != variabile:
                    continue
                g = lambda k: eccodes.codes_get(h, k)
                ni, nj = g("Ni"), g("Nj")
                lat1, lat2 = g("latitudeOfFirstGridPointInDegrees"), g("latitudeOfLastGridPointInDegrees")
                lon1, lon2 = g("longitudeOfFirstGridPointInDegrees"), g("longitudeOfLastGridPointInDegrees")
                valori = eccodes.codes_get_values(h).reshape(nj, ni)
            finally:
                eccodes.codes_release(h)
            lats, lons = np.linspace(lat1, lat2, nj), np.linspace(lon1, lon2, ni)
            if lats[0] > lats[-1]:
                lats, valori = lats[::-1], valori[::-1]
            i = int(np.clip(np.searchsorted(lats, lat) - 1, 0, nj - 2))
            j = int(np.clip(np.searchsorted(lons, lon) - 1, 0, ni - 2))
            ty = (lat - lats[i]) / (lats[i + 1] - lats[i])
            tx = (lon - lons[j]) / (lons[j + 1] - lons[j])
            v = valori[i:i + 2, j:j + 2]
            return float((1 - ty) * ((1 - tx) * v[0, 0] + tx * v[0, 1])
                         + ty * ((1 - tx) * v[1, 0] + tx * v[1, 1]))
    raise ValueError(f"{variabile} non trovato in {path}")


@task
def get_aod_from_aeronet(aeronet_csv_path: str, when: dt.datetime) -> AtmosphericParams:
    raise NotImplementedError("Implementa il parsing AERONET qui.")


_S2_SRF_ATTR = {
    "S2A": {"B03": "S2A_MSI_03", "B04": "S2A_MSI_04",
            "B05": "S2A_MSI_05", "B08": "S2A_MSI_08"},
    "S2B": {"B03": "S2B_MSI_03", "B04": "S2B_MSI_04",
            "B05": "S2B_MSI_05", "B08": "S2B_MSI_08"},
}


def _percorso_6s() -> str | None:
    candidati = [os.environ.get("SIXS_PATH", ""), Path(__file__).resolve().parent / "6S" / "build" / "sixs"]
    for c in candidati:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return str(c)
    return None


@task
def build_6s_coefficients(band: S2Band, meta: S2L1CMetadata,
                           atmos: AtmosphericParams,
                           spectral_mode: str = "boxcar",
                           srf_satellite: str = "S2A") -> tuple[float, float, float]:
    s = SixS(_percorso_6s())

    s.geometry = Geometry.User()
    s.geometry.solar_z = meta.sun_zenith_deg
    s.geometry.solar_a = meta.sun_azimuth_deg
    s.geometry.view_z = meta.view_zenith_deg[band.band_id]
    s.geometry.view_a = meta.view_azimuth_deg[band.band_id]
    s.geometry.month = meta.sensing_datetime.month
    s.geometry.day = meta.sensing_datetime.day

    s.altitudes.set_target_sea_level()
    s.altitudes.set_sensor_satellite_level()

    s.atmos_profile = AtmosProfile.PredefinedType(atmos.atmos_profile)
    s.aero_profile = AeroProfile.PredefinedType(atmos.aero_profile)
    s.aot550 = atmos.aod_550

    s.ground_reflectance = GroundReflectance.HomogeneousLambertian(0.02)

    s.atmos_corr = AtmosCorr.AtmosCorrLambertianFromRadiance(100)

    if spectral_mode == "boxcar":
        half = band.fwhm_um / 2.0
        s.wavelength = Wavelength(band.center_um - half, band.center_um + half)
    elif spectral_mode == "srf":
        attr_name = _S2_SRF_ATTR[srf_satellite][band.band_id]
        s.wavelength = Wavelength(getattr(PredefinedWavelengths, attr_name))
    else:
        raise ValueError(f"spectral_mode sconosciuto: {spectral_mode!r} "
                          "(atteso 'boxcar' o 'srf')")

    s.run()
    return s.outputs.coef_xa, s.outputs.coef_xb, s.outputs.coef_xc


@task
def invert_to_boa(radiance: np.ndarray, xa: float, xb: float, xc: float) -> np.ndarray:
    y = xa * radiance - xb
    return y / (1.0 + xc * y)


@task
def compute_ndci(boa_b04: np.ndarray, boa_b05: np.ndarray,
                  min_denominator: float = 1e-3) -> np.ndarray:

    denom = boa_b04 + boa_b05
    unstable = np.abs(denom) < min_denominator
    with np.errstate(divide="ignore", invalid="ignore"):
        ndci = (boa_b05 - boa_b04) / denom
    return np.where(unstable, np.nan, ndci)


@task
def compute_turbidity_index(boa_b03: np.ndarray, boa_b08: np.ndarray,
                             min_denominator: float = 1e-3) -> np.ndarray:

    unstable = np.abs(boa_b03) < min_denominator
    with np.errstate(divide="ignore", invalid="ignore"):
        turbidity = boa_b08 / boa_b03
    return np.where(unstable, np.nan, turbidity)


@task
def water_mask(boa_b03: np.ndarray, boa_b08: np.ndarray,
                threshold: float = 0.0) -> np.ndarray:
    ndwi = (boa_b03 - boa_b08) / (boa_b03 + boa_b08)
    return ndwi > threshold


@task
def monte_carlo_uncertainty(dn: np.ndarray, band: S2Band, meta: S2L1CMetadata,
                             atmos: AtmosphericParams, n_draws: int = 30,
                             rng_seed: int = 42) -> np.ndarray:

    if atmos.aod_uncertainty is None:
        raise ValueError("atmos.aod_uncertainty non impostata.")

    rng = np.random.default_rng(rng_seed)
    rho_toa = dn_to_toa_reflectance.fn(dn, band, meta)
    radiance = toa_reflectance_to_radiance.fn(rho_toa, band, meta)
    draws = np.empty((n_draws, *dn.shape), dtype=np.float64)

    for i in range(n_draws):
        perturbed_aod = max(0.0, rng.normal(atmos.aod_550, atmos.aod_uncertainty))
        perturbed = AtmosphericParams(
            aod_550=perturbed_aod, aero_profile=atmos.aero_profile,
            atmos_profile=atmos.atmos_profile,
        )
        xa, xb, xc = build_6s_coefficients.fn(band, meta, perturbed)
        draws[i] = invert_to_boa.fn(radiance, xa, xb, xc)

    return draws.std(axis=0)


@task
def compute_scenario(radiance: dict[str, np.ndarray], meta: S2L1CMetadata,
                      atmos: AtmosphericParams, mask: np.ndarray,
                      aero_profile: int | None = None,
                      spectral_mode: str = "boxcar",
                      srf_satellite: str = "S2A") -> dict:

    profile = aero_profile if aero_profile is not None else atmos.aero_profile
    fixed = AtmosphericParams(aod_550=atmos.aod_550, aero_profile=profile,
                               atmos_profile=atmos.atmos_profile)
    boa = {}
    for band_id, rad in radiance.items():
        band = _BANDS_BY_ID[band_id]
        xa, xb, xc = build_6s_coefficients.fn(
            band, meta, fixed, spectral_mode, srf_satellite
        )
        boa[band_id] = invert_to_boa.fn(rad, xa, xb, xc)
    ndci = compute_ndci.fn(boa["B04"], boa["B05"])
    turbidity = compute_turbidity_index.fn(boa["B03"], boa["B08"])
    return {
        "boa": boa, "ndci": ndci, "turbidity": turbidity,
        "area_mean_ndci": float(np.nanmean(ndci[mask])),
        "area_mean_turbidity": float(np.nanmean(turbidity[mask])),
    }


N_GRIGLIA_6S = 41
PASSO_GRIGLIA_FINE = 0.0002
N_VERIFICA_ESTRAZIONI = 2000


def _medie_area(rad_acqua: dict[str, np.ndarray], coef: dict[str, tuple]) -> dict:
    boa = {b: invert_to_boa.fn(rad_acqua[b], *coef[b]) for b in rad_acqua}
    ndci = compute_ndci.fn(boa["B04"], boa["B05"])
    turb = compute_turbidity_index.fn(boa["B03"], boa["B08"])
    return {
        "boa": {b: float(np.nanmean(v)) for b, v in boa.items()},
        "ndci": float(np.nanmean(ndci)),
        "turbidity": float(np.nanmean(turb)),
    }


def _appiattisci(medie: dict) -> dict[str, float]:
    out = {f"boa_{b}": v for b, v in medie["boa"].items()}
    out["ndci"] = medie["ndci"]
    out["turbidity"] = medie["turbidity"]
    return out


@task
def monte_carlo_area_means_fast(radiance: dict[str, np.ndarray], meta: S2L1CMetadata,
                                 atmos: AtmosphericParams, mask: np.ndarray,
                                 aero_profile: int | None = None,
                                 spectral_mode: str = "boxcar",
                                 srf_satellite: str = "S2A",
                                 n_draws: int = 1_000_000, rng_seed: int = 42,
                                 n_test_interpolazione: int = 5,
                                 n_verifica_estrazioni: int = N_VERIFICA_ESTRAZIONI,
                                 n_nodi_6s: int = N_GRIGLIA_6S,
                                 passo_griglia_fine: float = PASSO_GRIGLIA_FINE) -> dict:
    # 6S sui nodi di AOD, medie d'area tabulate su griglia fine, estrazioni interpolate
    from scipy.interpolate import CubicSpline

    if atmos.aod_uncertainty is None:
        raise ValueError("atmos.aod_uncertainty non impostata.")
    profile = aero_profile if aero_profile is not None else atmos.aero_profile

    rng = np.random.default_rng(rng_seed)
    aods = np.maximum(0.0, rng.normal(atmos.aod_550, atmos.aod_uncertainty, n_draws))

    aod_max = max(0.8, float(np.ceil(aods.max() * 10) / 10))
    griglia = np.linspace(0.0, aod_max, n_nodi_6s)

    def params(aod: float) -> AtmosphericParams:
        return AtmosphericParams(aod_550=float(aod), aero_profile=profile,
                                 atmos_profile=atmos.atmos_profile)

    band_ids = list(radiance.keys())
    spline = {}
    for b in band_ids:
        tab = np.array([build_6s_coefficients.fn(_BANDS_BY_ID[b], meta, params(a),
                                                 spectral_mode, srf_satellite)
                        for a in griglia])
        spline[b] = CubicSpline(griglia, tab, axis=0)

    def coef_interp(aod: float) -> dict[str, tuple]:
        return {b: tuple(float(c) for c in spline[b](aod)) for b in band_ids}

    rad_acqua = {b: radiance[b][mask] for b in band_ids}
    fine = np.arange(0.0, aod_max + passo_griglia_fine / 2, passo_griglia_fine)
    righe = [_appiattisci(_medie_area(rad_acqua, coef_interp(a))) for a in fine]
    tab_fine = {k: np.array([r[k] for r in righe]) for k in righe[0]}

    def valuta(a: np.ndarray, chiave: str) -> np.ndarray:
        return np.interp(a, fine, tab_fine[chiave])

    out = {
        "boa": {b: valuta(aods, f"boa_{b}") for b in band_ids},
        "ndci": valuta(aods, "ndci"),
        "turbidity": valuta(aods, "turbidity"),
        "aod": aods,
        "info": {
            "metodo": "6S tabulato su griglia + interpolazione",
            "n_draws": int(n_draws),
            "n_nodi_6s": int(n_nodi_6s),
            "aod_max_griglia": aod_max,
            "passo_griglia_fine": float(passo_griglia_fine),
            "frazione_aod_zero": float(np.mean(aods == 0.0)),
            "aod_media": float(atmos.aod_550),
            "aod_sigma": float(atmos.aod_uncertainty),
            "seme": int(rng_seed),
        },
        "tabulazione": {"aod": fine, **tab_fine},
    }

    if n_test_interpolazione:
        out["verifica"] = verifica_interpolazione(
            rad_acqua, meta, atmos, profile, spectral_mode, srf_satellite,
            griglia, valuta, n_test_interpolazione, passo_griglia_fine,
        )
        m = min(n_verifica_estrazioni, n_draws)
        esatto = [_appiattisci(_medie_area(rad_acqua, coef_interp(a))) for a in aods[:m]]
        out["verifica"]["n_estrazioni"] = m
        out["verifica"]["estrazioni"] = {}
        for k in esatto[0]:
            e = np.array([r[k] for r in esatto])
            t = valuta(aods[:m], k)
            out["verifica"]["estrazioni"][k] = {
                "max_abs": float(np.max(np.abs(t - e))),
                "sigma_esatta": float(e.std(ddof=1)),
                "sigma_tabulata": float(t.std(ddof=1)),
                "pct_esatti": np.percentile(e, [2.5, 97.5]).tolist(),
                "pct_tabulati": np.percentile(t, [2.5, 97.5]).tolist(),
            }
    return out


def verifica_interpolazione(rad_acqua, meta, atmos, profile, spectral_mode,
                            srf_satellite, griglia, valuta, n_test,
                            passo_griglia_fine: float = PASSO_GRIGLIA_FINE) -> dict:
    obiettivi = atmos.aod_550 + atmos.aod_uncertainty * np.array([-0.5, 0.0, 1.0, 2.0, 3.0])
    obiettivi = np.clip(obiettivi, griglia[0], griglia[-1])[:n_test]
    idx = np.clip(np.searchsorted(griglia, obiettivi) - 1, 0, griglia.size - 2)
    test = np.unique((griglia[idx] + griglia[idx + 1]) / 2 + passo_griglia_fine / 2)

    diff = {}
    for a in test:
        p = AtmosphericParams(aod_550=float(a), aero_profile=profile,
                              atmos_profile=atmos.atmos_profile)
        coef = {b: build_6s_coefficients.fn(_BANDS_BY_ID[b], meta, p,
                                            spectral_mode, srf_satellite)
                for b in rad_acqua}
        diretto = _appiattisci(_medie_area(rad_acqua, coef))
        for k, v in diretto.items():
            diff.setdefault(k, []).append(abs(float(valuta(np.array([a]), k)[0]) - v))
    return {"aod_test": test.tolist(),
            "max_abs": {k: float(max(v)) for k, v in diff.items()}}


@task
def align_reference_band(path: str, target_shape: tuple[int, int],
                          target_transform, target_crs,
                          quantification_value: float = 10000.0,
                          add_offset: float = 0.0) -> np.ndarray:
    height, width = target_shape
    with rasterio.open(path) as src:
        dest = np.empty((height, width), dtype=np.float64)
        reproject(
            source=rasterio.band(src, 1),
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
        )
    return (dest + add_offset) / quantification_value


@task
def read_reference_product(path: str, quantification_value: float = 10000.0,
                            add_offset: float = 0.0) -> np.ndarray:
    with rasterio.open(path) as src:
        dn = src.read(1).astype(np.float64)
    return (dn + add_offset) / quantification_value


@task
def compare_to_reference(own: np.ndarray, reference: np.ndarray,
                          mask: np.ndarray | None = None,
                          shapiro_max_n: int = 5000) -> dict:
    if mask is not None:
        own = own[mask]
        reference = reference[mask]
    valid = ~(np.isnan(own) | np.isnan(reference) |
              np.isinf(own) | np.isinf(reference))
    own_v, ref_v = own[valid], reference[valid]
    if own_v.size == 0:
        raise ValueError("Nessun pixel valido in comune tra stima propria e riferimento.")
    residui = own_v - ref_v
    n = residui.size

    if n <= shapiro_max_n:
        _, p_norm = stats.shapiro(residui)
        test_usato = f"Shapiro-Wilk (n={n})"
    else:
        mu, sigma = residui.mean(), residui.std(ddof=1)
        if sigma == 0:
            raise ValueError("Deviazione standard dei residui nulla: "
                              "stima propria e riferimento identici pixel "
                              "per pixel, il test di normalità non è "
                              "definito.")
        residui_standardizzati = (residui - mu) / sigma
        _, p_norm = stats.kstest(residui_standardizzati, "norm")
        test_usato = f"Kolmogorov-Smirnov (n={n}, parametri stimati dal campione)"

    r = float(np.corrcoef(own_v, ref_v)[0, 1])

    return {
        "bias": float(residui.mean()),
        "rmse": float(np.sqrt(np.mean(residui ** 2))),
        "r": r,
        "p_value_normalita": float(p_norm),
        "test_usato": test_usato,
        "n_pixel": n,
    }


@flow(name="processing-water-quality")
def processing_flow(ingested: IngestedProduct, atmos: AtmosphericParams,
                     reference_ndci_path: str | None = None,
                     reference_quantification_value: float = 10000.0,
                     reference_add_offset: float = 0.0,
                     bbox: tuple[float, float, float, float] | None = None) -> dict:
    meta = parse_s2_l1c_metadata(ingested.metadata_path, ingested.tile_metadata_path)

    grid = load_water_bands(ingested.band_paths, bbox=bbox)
    dns = grid["bands"]

    boa = {}
    radiances = {}
    for band_id in ("B03", "B04", "B05", "B08"):
        band = _BANDS_BY_ID[band_id]
        dn = dns[band_id]
        rho_toa = dn_to_toa_reflectance(dn, band, meta)
        radiance = toa_reflectance_to_radiance(rho_toa, band, meta)
        radiances[band_id] = radiance
        xa, xb, xc = build_6s_coefficients(band, meta, atmos)
        boa[band_id] = invert_to_boa(radiance, xa, xb, xc)

    ndci = compute_ndci(boa["B04"], boa["B05"])
    turbidity = compute_turbidity_index(boa["B03"], boa["B08"])
    mask = water_mask(boa["B03"], boa["B08"])

    results: dict = {"boa": boa, "ndci": ndci, "turbidity": turbidity,
                      "water_mask": mask, "radiance": radiances, "meta": meta,
                      "grid_transform": grid["transform"], "grid_crs": grid["crs"]}

    if atmos.aod_uncertainty is not None:
        results["ndci_uncertainty_proxy"] = monte_carlo_uncertainty(
            dns["B05"], _BANDS_BY_ID["B05"], meta, atmos
        )

    if reference_ndci_path:
        ref_ndci = read_reference_product(
            reference_ndci_path, reference_quantification_value, reference_add_offset
        )
        results["validation"] = compare_to_reference(ndci, ref_ndci, mask=mask)

    return results


def default_sen2cor_paths(base_dir: str, granule_name: str, tile_id: str,
                           sensing_stamp: str) -> dict[str, str]:
    img_data = os.path.join(base_dir, "GRANULE", granule_name, "IMG_DATA")
    return {
        "B03": os.path.join(img_data, "R10m", f"{tile_id}_{sensing_stamp}_B03_10m.jp2"),
        "B04": os.path.join(img_data, "R10m", f"{tile_id}_{sensing_stamp}_B04_10m.jp2"),
        "B05": os.path.join(img_data, "R20m", f"{tile_id}_{sensing_stamp}_B05_20m.jp2"),
        "B08": os.path.join(img_data, "R10m", f"{tile_id}_{sensing_stamp}_B08_10m.jp2"),
    }


def load_sen2cor_reflectance(risultati: dict, sen2cor_paths: dict[str, str],
                              quantification_value: float = 10000.0,
                              add_offset: float = -1000.0,
                              band_ids: tuple[str, ...] = ("B03", "B04", "B05", "B08"),
                              ) -> dict:

    mancanti = [b for b in band_ids if not os.path.isfile(sen2cor_paths.get(b, ""))]
    if mancanti:
        raise FileNotFoundError(
            f"File Sen2Cor non trovati per: {', '.join(mancanti)}. "
            f"Path attesi: { {b: sen2cor_paths.get(b) for b in mancanti} }"
        )
    grid_shape = risultati["boa"]["B04"].shape
    boa = {
        band_id: align_reference_band.fn(
            sen2cor_paths[band_id], grid_shape,
            risultati["grid_transform"], risultati["grid_crs"],
            quantification_value=quantification_value, add_offset=add_offset,
        )
        for band_id in band_ids
    }
    return {
        "boa": boa,
        "ndci": compute_ndci.fn(boa["B04"], boa["B05"]),
        "turbidity": compute_turbidity_index.fn(boa["B03"], boa["B08"]),
    }


def compare_to_sen2cor(risultati: dict, sen2cor_paths: dict[str, str],
                        quantification_value: float = 10000.0,
                        add_offset: float = -1000.0,
                        band_ids: tuple[str, ...] = ("B03", "B04", "B05", "B08"),
                        ) -> dict[str, dict]:

    ref = load_sen2cor_reflectance(
        risultati, sen2cor_paths, quantification_value, add_offset, band_ids,
    )
    mask = risultati["water_mask"]

    righe: dict[str, dict] = {}
    for band_id in band_ids:
        righe[band_id] = compare_to_reference.fn(
            risultati["boa"][band_id], ref["boa"][band_id], mask=mask,
        )

    righe["NDCI"] = compare_to_reference.fn(risultati["ndci"], ref["ndci"], mask=mask)
    righe["Torbidita"] = compare_to_reference.fn(
        risultati["turbidity"], ref["turbidity"], mask=mask,
    )
    return righe
