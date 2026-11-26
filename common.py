from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

from config import Configurazione
from ingestion import IngestedProduct, ingestion_flow
from processing import (
    AtmosphericParams,
    S2L1CMetadata,
    aod_da_aux_camsfo,
    get_aod_from_cams,
    processing_flow,
)

BAND_IDS = ["B03", "B04", "B05", "B08"]


BASE_DIR = Path(__file__).resolve().parent


def trova_sen2cor(nome_prodotto: str, granule: str | None = None) -> dict[str, str] | None:
    parti = nome_prodotto.removesuffix(".SAFE").split("_")
    data = parti[2][:8]
    tile = next((x for x in parti if re.fullmatch(r"T\d{2}[A-Z]{3}", x)), None)

    if granule:
        candidati = [Path(granule) if Path(granule).is_absolute() else BASE_DIR / granule]
    else:
        candidati = sorted((BASE_DIR / "GRANULE").glob(f"L2A_{tile}_*_{data}T*"))
    for cartella in candidati:
        img = cartella / "IMG_DATA"
        paths = {}
        for band_id, res in (("B03", "10m"), ("B04", "10m"), ("B05", "20m"), ("B08", "10m")):
            trovati = sorted((img / f"R{res}").glob(f"*_{band_id}_{res}.jp2"))
            if trovati:
                paths[band_id] = str(trovati[0])
        if len(paths) == 4:
            return paths
    return None


@dataclass
class Scene:

    ingested: IngestedProduct
    atmos: AtmosphericParams
    risultati: dict
    sen2cor_paths: dict[str, str] | None = None

    @property
    def radiance(self) -> dict[str, np.ndarray]:
        return self.risultati["radiance"]

    @property
    def mask(self) -> np.ndarray:
        return self.risultati["water_mask"]

    @property
    def meta(self) -> S2L1CMetadata:
        return self.risultati["meta"]


def load_scene(cfg: Configurazione) -> Scene:
    area = cfg.area
    ingested_products = ingestion_flow(
        bbox=area.bbox,
        start_date=area.data_inizio,
        end_date=area.data_fine + dt.timedelta(days=1),  # il filtro CDSE esclude la data finale
        band_ids=BAND_IDS,
        product_type="MSIL1C",
        max_cloud_cover=cfg.max_nubi,
        use_s3=cfg.use_s3,
        max_products=1,
    )
    if not ingested_products:
        raise RuntimeError(
            f"Nessun prodotto trovato per l'area {area.nome} tra {area.data_inizio} e "
            f"{area.data_fine} con nubi ≤ {cfg.max_nubi}%. Verifica su "
            "https://browser.dataspace.copernicus.eu se esistono scene in quelle date."
        )

    primo = ingested_products[0]
    if area.prodotto_atteso and primo.product.name.removesuffix(".SAFE") != area.prodotto_atteso:
        raise RuntimeError(
            f"Selezionato {primo.product.name}, atteso {area.prodotto_atteso}. "
            "Controlla le date dell'area in config.py."
        )

    center_lat, center_lon = area.centro
    sensing_datetime = dt.datetime.combine(primo.product.sensing_date, dt.time(11, 0))
    if cfg.aod is not None:
        atmos = AtmosphericParams(aod_550=cfg.aod, aod_uncertainty=cfg.incertezza_aod,
                                  source="manuale")
    else:
        aod = None
        aux = Path(primo.tile_metadata_path).parent / "AUX_DATA" / "AUX_CAMSFO"
        if cfg.fonte_aod == "prodotto" and aux.is_file():
            try:
                aod = aod_da_aux_camsfo(aux, center_lat, center_lon)
            except Exception as exc:
                print(f"AUX_CAMSFO non leggibile ({exc}): uso ADS")
        if aod is not None:
            atmos = AtmosphericParams(aod_550=aod, aod_uncertainty=cfg.incertezza_aod,
                                      source="cams-aux-l1c")
        else:
            atmos = get_aod_from_cams(
                lat=center_lat,
                lon=center_lon,
                when=sensing_datetime,
                aod_uncertainty=cfg.incertezza_aod,
            )
    print(f"Scena {primo.product.name}, AOD {atmos.aod_550:.3f} ({atmos.source})")

    risultati = processing_flow(ingested=primo, atmos=atmos, bbox=area.bbox)

    sen2cor = trova_sen2cor(primo.product.name, area.sen2cor_granule)
    if sen2cor is None:
        print("Sen2Cor non trovato in GRANULE/: confronto saltato")
    return Scene(ingested=primo, atmos=atmos, risultati=risultati, sen2cor_paths=sen2cor)


def describe(
    arr: np.ndarray,
    mask: np.ndarray,
    nome: str,
    n_max_shapiro: int = 5000,
    seed: int = 42,
) -> dict | None:

    valid = arr[mask & ~np.isnan(arr) & ~np.isinf(arr)]
    if valid.size == 0:
        print(f"{nome}: nessun pixel valido")
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
