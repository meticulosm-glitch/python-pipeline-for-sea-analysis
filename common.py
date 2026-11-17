from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

from ingestion import IngestedProduct, ingestion_flow
from processing import (
    AtmosphericParams,
    S2L1CMetadata,
    default_sen2cor_paths,
    get_aod_from_cams,
    processing_flow,
)

BBOX = (9.78, 44.00, 9.92, 44.13)  # lon_min, lat_min, lon_max, lat_max
BAND_IDS = ["B03", "B04", "B05", "B08"]
MAX_CLOUD_COVER = 10.0
AOD_UNCERTAINTY_FROM_LITERATURE = 0.11

# Da adattare se la scena elaborata cambia.
GRANULE_NAME = "L2A_T32TNP_A010431_20260904T101800"
TILE_ID = "T32TNP"
SENSING_STAMP = "20260904T100551"


def sen2cor_paths() -> dict[str, str]:
    """Path attesi delle 4 bande Sen2Cor, condivisi da report_risultati.py
    e report_mappe.py (prima erano duplicati in ciascun file)."""
    base_dir = str(Path(__file__).resolve().parent)
    return default_sen2cor_paths(
        base_dir, granule_name=GRANULE_NAME, tile_id=TILE_ID,
        sensing_stamp=SENSING_STAMP,
    )


@dataclass
class Scene:

    ingested: IngestedProduct
    atmos: AtmosphericParams
    risultati: dict

    @property
    def radiance(self) -> dict[str, np.ndarray]:
        return self.risultati["radiance"]

    @property
    def mask(self) -> np.ndarray:
        return self.risultati["water_mask"]

    @property
    def meta(self) -> S2L1CMetadata:
        return self.risultati["meta"]


def load_scene(max_products: int = 1, use_s3: bool = False) -> Scene:

    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=90)

    ingested_products = ingestion_flow(
        bbox=BBOX,
        start_date=start_date,
        end_date=end_date,
        band_ids=BAND_IDS,
        product_type="MSIL1C",
        max_cloud_cover=MAX_CLOUD_COVER,
        use_s3=use_s3,
        max_products=max_products,
    )
    if not ingested_products:
        raise RuntimeError(
            "Nessun prodotto trovato con questi parametri. Prima di cambiare "
            "criteri,verifica manualmente su "
            "https://browser.dataspace.copernicus.eu se esistono scene "
            "recenti sull'area."
        )

    primo = ingested_products[0]
    print(
        f"Prodotto selezionato: {primo.product.name}  "
        f"nubi={primo.product.cloud_cover}  "
        f"overlap={primo.product.overlap_ratio:.2f}"
    )

    lon_min, lat_min, lon_max, lat_max = BBOX
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2

    # Metto data/ora esatta di acquisizione nei metadati del prodotto
    sensing_datetime = dt.datetime.combine(primo.product.sensing_date, dt.time(11, 0))

    atmos = get_aod_from_cams(
        lat=center_lat,
        lon=center_lon,
        when=sensing_datetime,
        aod_uncertainty=AOD_UNCERTAINTY_FROM_LITERATURE,
    )
    print(
        f"AOD@550nm da CAMS ({atmos.source}): {atmos.aod_550:.4f} "
        f"(incertezza: {atmos.aod_uncertainty}, DA VERIFICARE)"
    )

    risultati = processing_flow(ingested=primo, atmos=atmos, bbox=BBOX)
    return Scene(ingested=primo, atmos=atmos, risultati=risultati)


def describe(
    arr: np.ndarray,
    mask: np.ndarray,
    nome: str,
    n_max_shapiro: int = 5000,
    seed: int = 42,
) -> dict | None:

    valid = arr[mask & ~np.isnan(arr) & ~np.isinf(arr)]
    if valid.size == 0:
        print(f"ATTENZIONE - {nome}: nessun pixel valido dopo maschera/filtro NaN-Inf.")
        return None

    q1, q3 = np.percentile(valid, [25, 75])
    rng = np.random.default_rng(seed)
    campione = (
        valid if valid.size <= n_max_shapiro
        else rng.choice(valid, size=n_max_shapiro, replace=False)
    )
    _, p_value = stats.shapiro(campione)

    result = {
        "media": float(valid.mean()),
        "mediana": float(np.median(valid)),
        "iqr": float(q3 - q1),
        "min": float(valid.min()),
        "max": float(valid.max()),
        "n_validi": int(valid.size),
        "n_totali": int(arr.size),
        "p_value": float(p_value),
        "n_campione_shapiro": int(campione.size),
    }
    return result
