from __future__ import annotations

import time

import numpy as np
from Py6S import AeroProfile
from scipy import stats

from common import BAND_IDS, Scene
from config import Configurazione
from processing import (
    _BANDS_BY_ID,
    build_6s_coefficients,
    compute_scenario,
    monte_carlo_area_means_fast,
)
from tabelle import RaccoglitoreTabelle

SEME_SHAPIRO = 42


def shortest_interval(draws: np.ndarray, p: float = 0.95) -> tuple[float, float]:
    x = np.sort(draws)
    n = x.size
    k = int(np.ceil(p * n))
    i = int(np.argmin(x[k - 1:] - x[:n - k + 1]))
    return float(x[i]), float(x[i + k - 1])


def mc_stats(draws: np.ndarray, n_max_shapiro: int = 5000, seed: int = SEME_SHAPIRO) -> dict:
    sigma = float(draws.std(ddof=1))
    p025, q1, p50, q3, p975 = np.percentile(draws, [2.5, 25, 50, 75, 97.5])
    lo, hi = shortest_interval(draws, 0.95)
    rng = np.random.default_rng(seed)
    campione = draws if draws.size <= n_max_shapiro else rng.choice(draws, n_max_shapiro, replace=False)
    _, p_value = stats.shapiro(campione)
    return {
        "media": float(draws.mean()),
        "mediana": float(p50),
        "sigma": sigma,
        "k2": 2 * sigma,
        "p2_5": float(p025),
        "p97_5": float(p975),
        "corto_lo": lo,
        "corto_hi": hi,
        "iqr": float(q3 - q1),
        "p_value": float(p_value),
        "normale": bool(p_value >= 0.05),
        "n_draws": int(draws.size),
    }


def _media_mascherata(arr: np.ndarray, mask: np.ndarray) -> float:
    valid = arr[mask & ~np.isnan(arr) & ~np.isinf(arr)]
    return float(valid.mean())


def _it(x: float, cifre: int = 2) -> str:
    if x == 0 or not np.isfinite(x):
        return f"{x:g}".replace(".", ",")
    decimali = max(0, cifre - 1 - int(np.floor(np.log10(abs(x)))))
    return f"{x:.{decimali}f}".replace(".", ",").replace("-", "−")


def _grandezze(scena: Scene, mc: dict) -> list[tuple[str, float, np.ndarray]]:
    ris, mask = scena.risultati, scena.mask
    g = [(f"ρ_BOA - {b}", _media_mascherata(ris["boa"][b], mask), mc["boa"][b]) for b in BAND_IDS]
    g.append(("NDCI", _media_mascherata(ris["ndci"], mask), mc["ndci"]))
    g.append(("Indice di torbidità", _media_mascherata(ris["turbidity"], mask), mc["turbidity"]))
    return g


def tabella_4_5_1_2(rac: RaccoglitoreTabelle, scena: Scene, mc_baseline: dict) -> None:
    righe, righe_testo = [], []
    for nome, det, draws in _grandezze(scena, mc_baseline):
        s = mc_stats(draws)
        if s["normale"]:
            nota_normalita = f"{s['p_value']:.4g}"
        else:
            nota_normalita = f"{s['p_value']:.4g} - normalità rigettata"
        righe.append([nome, det, s["media"], s["mediana"], s["sigma"], s["k2"],
                      f"[{s['p2_5']:.4f}; {s['p97_5']:.4f}]",
                      f"[{s['corto_lo']:.4f}; {s['corto_hi']:.4f}]",
                      nota_normalita])
        righe_testo.append([nome,
            f"{_it(det, 3)}; mediana {_it(s['mediana'])}; σ = {_it(s['sigma'])}; "
            f"intervallo di copertura al 95% dai percentili [{_it(s['p2_5'])}; {_it(s['p97_5'])}]; "
            f"intervallo più corto [{_it(s['corto_lo'])}; {_it(s['corto_hi'])}]. "
            f"L'incertezza estesa 2σ = {_it(s['k2'])} è riportata solo per confronto "
            f"con l'errore normalizzato."])
    rac.aggiungi(
        "Incertezza da AOD (sorgente parametrica)",
        ["Grandezza", "Valore deterministico", "Media MC", "Mediana MC",
         "Deviazione standard (1σ)", "2σ", "Intervallo 95% (percentili 2,5-97,5)",
         "Intervallo 95% più corto", "Normalità (p-value Shapiro, 5000 estrazioni)"],
        righe,
    )
    rac.aggiungi("Righe «Risultato» per le sintesi per indice (3.4.2)",
                 ["Grandezza", "Testo"], righe_testo)


def tabella_info_mc(rac: RaccoglitoreTabelle, scena: Scene, mc_baseline: dict,
                    rng_seed: int, srf_satellite: str) -> None:
    info = mc_baseline["info"]
    n = info["n_draws"]
    rac.aggiungi(
        "Informazioni sull'analisi di incertezza",
        ["Prodotto", "N estrazioni", "Seme", "SRF reale usata", "Metodo",
         "Nodi 6S per banda", "AOD max griglia", "Passo griglia fine",
         "Frazione estrazioni con AOD = 0", "Incertezza relativa di σ (≈0,71/√N)",
         "Estrazioni oltre ciascun percentile (0,025·N)"],
        [[scena.ingested.product.name, n, rng_seed, srf_satellite, info["metodo"],
          info["n_nodi_6s"], info["aod_max_griglia"], info["passo_griglia_fine"],
          info["frazione_aod_zero"], f"{100 * 0.7071 / np.sqrt(n):.2f}%", int(0.025 * n)]],
    )
    ver = mc_baseline.get("verifica")
    if ver:
        sigma = {f"boa_{b}": mc_baseline["boa"][b].std(ddof=1) for b in BAND_IDS}
        sigma["ndci"] = mc_baseline["ndci"].std(ddof=1)
        sigma["turbidity"] = mc_baseline["turbidity"].std(ddof=1)
        m = ver["n_estrazioni"]
        righe = []
        for k, v in ver["max_abs"].items():
            e = ver["estrazioni"][k]
            righe.append([k, v, e["max_abs"], e["max_abs"] / sigma[k] if sigma[k] > 0 else float("nan"),
                          e["sigma_esatta"], e["sigma_tabulata"],
                          "[{:.4f}; {:.4f}]".format(*e["pct_esatti"]),
                          "[{:.4f}; {:.4f}]".format(*e["pct_tabulati"])])
        rac.aggiungi(
            "Verifica della tabulazione (6S diretto in AOD = "
            + ", ".join(f"{a:.4f}" for a in ver["aod_test"])
            + f"; calcolo esatto sulle prime {m} estrazioni)",
            ["Grandezza", "Max |tabulato − 6S diretto|", f"Max |tabulato − esatto| ({m} estr.)",
             "Rapporto con σ", f"σ esatta ({m} estr.)", f"σ tabulata ({m} estr.)",
             "P2,5-P97,5 esatti", "P2,5-P97,5 tabulati"],
            righe,
        )


def tabella_4_5_1_3(rac: RaccoglitoreTabelle, scenari_profilo: dict) -> None:
    righe = []
    for nome, chiave in (
        ("NDCI (media sull'area)", "area_mean_ndci"),
        ("Indice di torbidità (media sull'area)", "area_mean_turbidity"),
    ):
        valori = {p: scenari_profilo[p][chiave] for p in ("Maritime", "Continental", "Urban")}
        rng = max(valori.values()) - min(valori.values())
        righe.append([nome, valori["Maritime"], valori["Continental"], valori["Urban"], rng])
    rac.aggiungi(
        "Incertezza/sensibilità da scelta del profilo aerosol",
        ["Grandezza", "Maritime (riferimento)", "Continental", "Urban", "Range tra scenari"],
        righe,
    )


def tabella_sensibilita(rac: RaccoglitoreTabelle, scenario_boxcar: dict, scenario_srf: dict,
                     srf_satellite: str) -> None:
    righe = []
    for nome, chiave in (
        ("NDCI (media sull'area)", "area_mean_ndci"),
        ("Indice di torbidità (media sull'area)", "area_mean_turbidity"),
    ):
        boxcar, srf = scenario_boxcar[chiave], scenario_srf[chiave]
        righe.append([nome, boxcar, srf, srf - boxcar])
    rac.aggiungi(
        "Incertezza/sensibilità da approssimazione della risposta spettrale",
        ["Grandezza", "Boxcar (adottato)", f"SRF reale (usato {srf_satellite})", "Differenza"],
        righe,
    )


def tabella_incertezza_combinata(rac: RaccoglitoreTabelle, combinazioni: dict) -> None:
    righe = []
    for nome, mc in combinazioni.items():
        sn, st = mc_stats(mc["ndci"]), mc_stats(mc["turbidity"])
        righe.append([nome, sn["sigma"], f"[{sn['p2_5']:.4f}; {sn['p97_5']:.4f}]",
                      st["sigma"], f"[{st['p2_5']:.4f}; {st['p97_5']:.4f}]"])
    rac.aggiungi(
        "Incertezza combinata",
        ["Combinazione (profilo × SRF)", "σ NDCI da AOD", "Intervallo 95% NDCI",
         "σ Torbidità da AOD", "Intervallo 95% torbidità"],
        righe,
    )


def stima_tempi(meta, atmos, cfg: Configurazione) -> None:
    p = cfg.precisione
    t0 = time.monotonic()
    build_6s_coefficients.fn(_BANDS_BY_ID["B03"], meta, atmos, "boxcar", cfg.srf_satellite)
    single_run_s = time.monotonic() - t0
    n_mc_runs = 3 * (p.n_nodi_6s + p.n_test_6s) * 4
    n_scenario_runs = 3 * 4 + 4
    totale = single_run_s * (n_mc_runs + n_scenario_runs)
    print(f"6S: {n_mc_runs + n_scenario_runs} run, ~{totale / 60:.0f} min")


def _mc(scena: Scene, cfg: Configurazione, profilo, spectral_mode: str) -> dict:
    p = cfg.precisione
    return monte_carlo_area_means_fast(
        scena.radiance, scena.meta, scena.atmos, scena.mask,
        aero_profile=profilo, spectral_mode=spectral_mode, srf_satellite=cfg.srf_satellite,
        n_draws=p.n_estrazioni, rng_seed=cfg.seme,
        n_test_interpolazione=p.n_test_6s, n_verifica_estrazioni=p.n_verifica_estrazioni,
        n_nodi_6s=p.n_nodi_6s, passo_griglia_fine=p.passo_griglia_fine,
    )


def aggiungi_tabelle(rac: RaccoglitoreTabelle, scena: Scene, cfg: Configurazione) -> dict:
    radiance, mask, meta, atmos = scena.radiance, scena.mask, scena.meta, scena.atmos
    srf = cfg.srf_satellite

    stima_tempi(meta, atmos, cfg)

    mc_baseline = _mc(scena, cfg, AeroProfile.Maritime, "boxcar")
    tabella_info_mc(rac, scena, mc_baseline, cfg.seme, srf)
    tabella_4_5_1_2(rac, scena, mc_baseline)

    scenari_profilo = {}
    for nome_profilo, costante in (
        ("Maritime", AeroProfile.Maritime),
        ("Continental", AeroProfile.Continental),
        ("Urban", AeroProfile.Urban),
    ):
        scenari_profilo[nome_profilo] = compute_scenario(
            radiance, meta, atmos, mask, aero_profile=costante, spectral_mode="boxcar",
        )
    tabella_4_5_1_3(rac, scenari_profilo)

    scenario_srf = compute_scenario(
        radiance, meta, atmos, mask,
        aero_profile=AeroProfile.Maritime, spectral_mode="srf", srf_satellite=srf,
    )
    tabella_sensibilita(rac, scenari_profilo["Maritime"], scenario_srf, srf)

    combinazioni = {
        "Maritime × Boxcar (riferimento)": mc_baseline,
        "Continental × Boxcar": _mc(scena, cfg, AeroProfile.Continental, "boxcar"),
        "Maritime × SRF reale (se disponibile)": _mc(scena, cfg, AeroProfile.Maritime, "srf"),
    }
    tabella_incertezza_combinata(rac, combinazioni)
    return combinazioni
