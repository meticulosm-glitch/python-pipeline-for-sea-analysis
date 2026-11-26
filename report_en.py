from __future__ import annotations

import numpy as np

from common import BAND_IDS, Scene
from processing import load_sen2cor_reflectance
from tabelle import RaccoglitoreTabelle

MAPE_ASSUNTO = 0.20  # U_rif Sen2Cor: assunzione di Tipo B

_INTESTAZIONI = ["Indicatore", "Riferimento", "ρ_riferimento", "U_propria",
                  "U_riferimento (stima Tipo B)", "E_n", "Overlap? (|E_n|≤1)"]
_NOTA_U_RIF_BANDA = f"MAPE<{int(MAPE_ASSUNTO * 100)}% × ρ"


def _media_mascherata(arr: np.ndarray, mask: np.ndarray) -> float:
    valid = arr[mask & ~np.isnan(arr) & ~np.isinf(arr)]
    return float(valid.mean())


def _en_overlap(propria: float, riferimento: float,
                 u_propria: float, u_riferimento: float) -> tuple[float, bool]:
    denom = np.sqrt(u_propria**2 + u_riferimento**2)
    en = (propria - riferimento) / denom
    return float(en), bool(abs(en) <= 1)


def _u_rif_ndci(ref: dict, mask: np.ndarray) -> float:
    rho_b04 = _media_mascherata(ref["boa"]["B04"], mask)
    rho_b05 = _media_mascherata(ref["boa"]["B05"], mask)
    u_b04 = MAPE_ASSUNTO * rho_b04
    u_b05 = MAPE_ASSUNTO * rho_b05
    d_b04 = -2 * rho_b05 / (rho_b04 + rho_b05) ** 2
    d_b05 = 2 * rho_b04 / (rho_b04 + rho_b05) ** 2
    return float(np.sqrt((d_b04 * u_b04) ** 2 + (d_b05 * u_b05) ** 2))


def _u_rif_torbidita(ref: dict, mask: np.ndarray) -> float:
    rho_b03 = _media_mascherata(ref["boa"]["B03"], mask)
    rho_b08 = _media_mascherata(ref["boa"]["B08"], mask)
    u_b03 = MAPE_ASSUNTO * rho_b03
    u_b08 = MAPE_ASSUNTO * rho_b08
    d_b03 = -rho_b08 / rho_b03 ** 2
    d_b08 = 1 / rho_b03
    return float(np.sqrt((d_b03 * u_b03) ** 2 + (d_b08 * u_b08) ** 2))


def _riga_banda(band_id: str, risultati: dict, ref: dict, mask: np.ndarray,
                 mc_baseline: dict) -> list:
    rho_propria = _media_mascherata(risultati["boa"][band_id], mask)
    rho_rif = _media_mascherata(ref["boa"][band_id], mask)
    u_propria = 2 * float(mc_baseline["boa"][band_id].std(ddof=1))
    u_rif = MAPE_ASSUNTO * rho_rif
    en, overlap = _en_overlap(rho_propria, rho_rif, u_propria, u_rif)
    return [band_id, "Sen2Cor", rho_rif, u_propria,
            f"{u_rif:.5f} ({_NOTA_U_RIF_BANDA})", en, overlap]


def _riga_ndci(risultati: dict, ref: dict, mask: np.ndarray, mc_baseline: dict) -> list:
    rho_propria = _media_mascherata(risultati["ndci"], mask)
    rho_rif = _media_mascherata(ref["ndci"], mask)
    u_propria = 2 * float(mc_baseline["ndci"].std(ddof=1))

    u_rif = _u_rif_ndci(ref, mask)

    en, overlap = _en_overlap(rho_propria, rho_rif, u_propria, u_rif)
    return ["NDCI", "Sen2Cor", rho_rif, u_propria,
            f"{u_rif:.5f} (propagato da B04,B05)", en, overlap]


def _riga_torbidita(risultati: dict, ref: dict, mask: np.ndarray, mc_baseline: dict) -> list:
    rho_propria = _media_mascherata(risultati["turbidity"], mask)
    rho_rif = _media_mascherata(ref["turbidity"], mask)
    u_propria = 2 * float(mc_baseline["turbidity"].std(ddof=1))

    u_rif = _u_rif_torbidita(ref, mask)

    en, overlap = _en_overlap(rho_propria, rho_rif, u_propria, u_rif)
    return ["Torbidità", "Sen2Cor", rho_rif, u_propria,
            f"{u_rif:.5f} (propagato da B03,B08)", en, overlap]


def compatibilita_simulata(draws_propria: np.ndarray, y_rif: float, u_rif_estesa: float,
                           rng: np.random.Generator) -> tuple[float, float, bool]:
    sigma_rif = u_rif_estesa / 2.0
    rif = rng.normal(y_rif, sigma_rif, size=draws_propria.size)
    d = draws_propria - rif
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(lo), float(hi), bool(lo <= 0.0 <= hi)


def _indicatori(scena: Scene, ref: dict, mc_baseline: dict):
    mask, ris = scena.mask, scena.risultati
    out = []
    for b in BAND_IDS:
        y_rif = _media_mascherata(ref["boa"][b], mask)
        out.append((b, _media_mascherata(ris["boa"][b], mask), y_rif, mc_baseline["boa"][b],
                    MAPE_ASSUNTO * y_rif))
    out.append(("NDCI", _media_mascherata(ris["ndci"], mask), _media_mascherata(ref["ndci"], mask),
                mc_baseline["ndci"], _u_rif_ndci(ref, mask)))
    out.append(("Torbidità", _media_mascherata(ris["turbidity"], mask),
                _media_mascherata(ref["turbidity"], mask), mc_baseline["turbidity"],
                _u_rif_torbidita(ref, mask)))
    return [(n, yp, yr, d, 2 * float(d.std(ddof=1)), ur) for n, yp, yr, d, ur in out]


def tabella_en_correlazione(rac: RaccoglitoreTabelle, indicatori, rs=(0.0, 0.5, 1.0)) -> None:
    righe = []
    for nome, yp, yr, _, up, ur in indicatori:
        riga = [nome, yp - yr, up, ur]
        for r in rs:
            den = np.sqrt(max(up**2 + ur**2 - 2 * r * up * ur, 0.0))
            riga.append((yp - yr) / den if den > 0 else float("nan"))
        righe.append(riga)
    rac.aggiungi("Errore normalizzato per diversi valori del coefficiente di correlazione",
                 ["Indicatore", "Δ (propria − Sen2Cor)", "U_propria (2σ)", "U_rif"]
                 + [f"E_n (r = {r:g})" for r in rs], righe)


def tabella_differenza_simulata(rac: RaccoglitoreTabelle, indicatori, rng_seed: int = 42) -> None:
    rng = np.random.default_rng(rng_seed)
    righe = []
    for nome, _, yr, draws, _, ur in indicatori:
        lo, hi, contiene = compatibilita_simulata(draws, yr, ur, rng)
        righe.append([nome, f"[{lo:.4f}; {hi:.4f}]", contiene])
    rac.aggiungi("Differenza simulata D = Y_propria − Y_rif (intervallo di copertura al 95%)",
                 ["Indicatore", "Intervallo di D al 95%", "Contiene lo zero?"], righe)


def tabella_en(rac: RaccoglitoreTabelle, scena: Scene, mc_baseline: dict, seme: int = 42) -> None:
    titolo = f"Compatibilità metrologica con Sen2Cor (numero di En, assunto MAPE Sen2Cor {int(MAPE_ASSUNTO*100)}%)"
    try:
        if scena.sen2cor_paths is None:
            raise FileNotFoundError("prodotto Sen2Cor dell'area non trovato in GRANULE/")
        ref = load_sen2cor_reflectance(scena.risultati, scena.sen2cor_paths)
    except FileNotFoundError:
        rac.aggiungi(titolo, _INTESTAZIONI,
                     [["Confronto Sen2Cor non disponibile (file mancanti)",
                       "Sen2Cor", "—", "—", "—", "—", "—"]])
        return

    mask = scena.mask
    righe = [_riga_banda(b, scena.risultati, ref, mask, mc_baseline) for b in BAND_IDS]
    righe.append(_riga_ndci(scena.risultati, ref, mask, mc_baseline))
    righe.append(_riga_torbidita(scena.risultati, ref, mask, mc_baseline))
    rac.aggiungi(titolo, _INTESTAZIONI, righe)

    indicatori = _indicatori(scena, ref, mc_baseline)
    tabella_en_correlazione(rac, indicatori)
    tabella_differenza_simulata(rac, indicatori, rng_seed=seme)
