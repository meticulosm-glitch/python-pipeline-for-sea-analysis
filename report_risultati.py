from __future__ import annotations

import re
import numpy as np
from Py6S import AeroProfile, AtmosProfile

from common import BAND_IDS, Scene, describe
from processing import compare_to_sen2cor
from tabelle import RaccoglitoreTabelle

_ATMOS_NAMES = {v: k for k, v in vars(AtmosProfile).items()
                if isinstance(v, int) and not k.startswith("_")}
_AERO_NAMES = {v: k for k, v in vars(AeroProfile).items()
               if isinstance(v, int) and not k.startswith("_")}
_AOD_SOURCE_NAMES = {
    "cams-eac4": "CAMS EAC4 (rianalisi)",
    "cams-forecast": "CAMS global atmospheric composition forecasts",
    "cams-aux-l1c": "CAMS forecast allegato al prodotto L1C (AUX_CAMSFO)",
    "manuale": "valore impostato manualmente",
}


def tabella_info_scena(rac: RaccoglitoreTabelle, scena: Scene) -> None:
    prodotto = scena.ingested.product
    meta, atmos = scena.meta, scena.atmos

    satellite = {"S2A": "Sentinel-2A", "S2B": "Sentinel-2B", "S2C": "Sentinel-2C"}.get(
        prodotto.name[:3], prodotto.name[:3])
    m = re.search(r"_N(\d{4})_", prodotto.name)
    baseline = f"N{m.group(1)}" if m else "n.d."
    nubi = "n.d." if prodotto.cloud_cover is None else f"{prodotto.cloud_cover:.2f}"
    vza_medio = float(np.mean(list(meta.view_zenith_deg.values())))

    righe = [
        ["Identificativo prodotto", f"{prodotto.name}.SAFE"],
        ["Satellite", satellite],
        ["Data e ora di acquisizione", meta.sensing_datetime.strftime("%d-%m-%Y %H:%M:%S UTC")],
        ["Copertura nuvolosa sulla scena (%)", nubi],
        ["Angolo zenitale solare (°)", f"{meta.sun_zenith_deg:.2f}"],
        ["Angolo zenitale di vista medio (°)", f"{vza_medio:.2f}"],
        ["Baseline di elaborazione", baseline],
        ["Fonte AOD", _AOD_SOURCE_NAMES.get(atmos.source, atmos.source)],
        ["AOD@550nm utilizzato", f"{atmos.aod_550:.4f}"],
        ["Incertezza sull'AOD (1σ)",
         "n.d." if atmos.aod_uncertainty is None else f"{atmos.aod_uncertainty:.3f}"],
        ["Profilo atmosferico", _ATMOS_NAMES.get(atmos.atmos_profile, str(atmos.atmos_profile))],
        ["Modello aerosol", _AERO_NAMES.get(atmos.aero_profile, str(atmos.aero_profile))],
    ]
    rac.aggiungi("Informazioni sulla scena", ["Parametro", "Valore"], righe)


def tabella_incertezza_proxy_ndci(rac, risultati) -> None:
    titolo = "Incertezza NDCI (proxy B05)"
    if "ndci_uncertainty_proxy" not in risultati:
        rac.aggiungi(titolo, ["Grandezza", "Media"],
                     [["Incertezza NDCI (proxy B05)", "non disponibile"]])
        return
    rac.aggiungi(
        titolo, ["Grandezza", "Media"],
        [["Incertezza NDCI (proxy B05)",
          float(np.nanmean(risultati["ndci_uncertainty_proxy"]))]],
    )


def _stat_o_errore(arr: np.ndarray, mask: np.ndarray, nome: str) -> dict:
    s = describe(arr, mask, nome)
    if s is None:
        raise RuntimeError(f"Nessun pixel valido per {nome}: controlla la maschera.")
    return s


def tabella_3_2(rac: RaccoglitoreTabelle, scena: Scene) -> None:
    righe = []
    for band_id in BAND_IDS:
        s = _stat_o_errore(scena.risultati["boa"][band_id], scena.mask, band_id)
        righe.append([band_id, s["media"], s["mediana"], s["iqr"], s["min"], s["max"],
                      s["n_validi"]])
    rac.aggiungi(
        "§3.2 — Riflettanza BOA ottenuta (pixel d'acqua)",
        ["Banda", "Media ρ_BOA", "Mediana", "IQR", "Min", "Max", "N. pixel validi"],
        righe,
    )


def tabella_3_3(rac: RaccoglitoreTabelle, scena: Scene) -> None:
    righe = []
    for nome, chiave in (("NDCI", "ndci"), ("Torbidità (B08/B03)", "turbidity")):
        s = _stat_o_errore(scena.risultati[chiave], scena.mask, nome)
        righe.append([nome, s["media"], s["mediana"], s["iqr"], s["min"], s["max"],
                      s["n_validi"], s["p_value"]])
    rac.aggiungi(
        "Indicatori di qualità dell'acqua derivati",
        ["Indicatore", "Media", "Mediana", "IQR", "Min", "Max", "N. pixel validi",
         "Normalità (p-value)"],
        righe,
    )


_INTESTAZIONI_3_5 = ["Banda / Indicatore", "Riferimento", "Bias", "RMSE", "r",
                     "Normalità residui (p-value)", "Test usato"]
_RIGA_C2RCC = ["NDCI", "C2RCC/ACOLITE (se confermati)", "—", "—", "—", "—", "—"]


def tabella_3_5(rac: RaccoglitoreTabelle, scena: Scene) -> None:
    titolo = "Verifica di compatibilità con la misura di riferimento"
    try:
        if scena.sen2cor_paths is None:
            raise FileNotFoundError("prodotto Sen2Cor dell'area non trovato in GRANULE/")
        righe_sen2cor = compare_to_sen2cor(scena.risultati, scena.sen2cor_paths)
    except FileNotFoundError:
        rac.aggiungi(
            titolo, _INTESTAZIONI_3_5,
            [["Confronto Sen2Cor non disponibile (file mancanti)", "Sen2Cor",
              "—", "—", "—", "—", "—"], _RIGA_C2RCC],
        )
        return

    nomi = {"Torbidita": "Torbidità"}
    righe = [
        [nomi.get(nome, nome), "Sen2Cor", r["bias"], r["rmse"], r["r"],
         r["p_value_normalita"], r["test_usato"]]
        for nome, r in righe_sen2cor.items()
    ]
    righe.append(_RIGA_C2RCC)
    rac.aggiungi(titolo, _INTESTAZIONI_3_5, righe)


def aggiungi_tabelle(rac: RaccoglitoreTabelle, scena: Scene) -> None:
    tabella_info_scena(rac, scena)
    tabella_incertezza_proxy_ndci(rac, scena.risultati)
    tabella_3_2(rac, scena)
    tabella_3_3(rac, scena)
    tabella_3_5(rac, scena)
