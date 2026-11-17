from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from Py6S import AeroProfile
from scipy import stats

from common import BAND_IDS, Scene, load_scene
from processing import (
    _BANDS_BY_ID,
    build_6s_coefficients,
    compute_scenario,
    monte_carlo_area_means,
)
from tabelle import RaccoglitoreTabelle

N_DRAWS = 100
RNG_SEED = 42
SRF_SATELLITE = "S2A"
OUTPUT_PATH = Path("tabelle_incertezza_4_5_1.md")


# --------------------------------------------------------------------------
# Statistiche di una distribuzione Monte Carlo
# --------------------------------------------------------------------------

def mc_stats(draws: np.ndarray) -> dict:
    """sigma (1), incertezza estesa (k=2, §3.4), IQR empirico e p-value
    Shapiro-Wilk su una serie di draw Monte Carlo (medie d'area per draw,
    vedi monte_carlo_area_means in processing.py)."""
    sigma = float(draws.std(ddof=1))
    q1, q3 = np.percentile(draws, [25, 75])
    _, p_value = stats.shapiro(draws)
    return {
        "sigma": sigma,
        "k2": 2 * sigma,
        "iqr": float(q3 - q1),
        "p_value": float(p_value),
        "normale": bool(p_value >= 0.05),
        "n_draws": int(draws.size),
    }


# --------------------------------------------------------------------------
# Tabelle (schermo + markdown), nello stesso formato dei placeholder in tesi
# --------------------------------------------------------------------------

def tabella_4_5_1_2(rac: RaccoglitoreTabelle, mc_baseline: dict) -> None:
    grandezze = [(f"ρ_BOA - {b}", mc_baseline["boa"][b]) for b in BAND_IDS]
    grandezze += [("NDCI", mc_baseline["ndci"]), ("Indice di torbidità", mc_baseline["turbidity"])]
    righe = []
    for nome, draws in grandezze:
        s = mc_stats(draws)
        if s["normale"]:
            nota_normalita = f"{s['p_value']:.4g}"
        else:
            nota_normalita = f"{s['p_value']:.4g} - normalità rigettata, IQR={s['iqr']:.5f}"
        righe.append([nome, s["sigma"], s["k2"], nota_normalita])
    rac.aggiungi(
        "Incertezza da AOD (sorgente parametrica)",
        ["Grandezza", "Deviazione standard (1σ)", "Incertezza estesa (k=2)",
         "Normalità distribuzione MC (p-value)"],
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
        righe.append([nome, float(mc["ndci"].std(ddof=1)), float(mc["turbidity"].std(ddof=1))])
    rac.aggiungi(
        "Incertezza combinata",
        ["Combinazione (profilo × SRF)", "σ NDCI da AOD", "σ Torbidità da AOD"],
        righe,
    )


# --------------------------------------------------------------------------
# Orchestrazione
# --------------------------------------------------------------------------

def stima_tempi(meta, atmos, n_draws: int) -> None:
    """Un run di 6S per farsi un'idea dei tempi totali prima di lanciare
    tutta la Monte Carlo (3 combinazioni x n_draws x 4 bande, più gli
    scenari singoli di §4.5.1.3/4.5.1.4)."""
    t0 = time.monotonic()
    build_6s_coefficients.fn(_BANDS_BY_ID["B03"], meta, atmos, "boxcar", "S2A")
    single_run_s = time.monotonic() - t0
    n_mc_runs = 3 * n_draws * 4  # baseline + Continental x Boxcar + Maritime x SRF (§4.5.1.5)
    n_scenario_runs = 3 * 4 + 4  # 3 profili aerosol (§4.5.1.3) + 1 SRF (§4.5.1.4), 4 bande
    totale = single_run_s * (n_mc_runs + n_scenario_runs)
    print(
        f"\nUn run di 6S impiega ~{single_run_s:.2f}s su questa macchina. "
        f"Stima per l'analisi di incertezza (~{n_mc_runs + n_scenario_runs} run totali, "
        f"n_draws={n_draws}): ~{totale / 60:.1f} minuti.\n"
        "Se ti sembra troppo, interrompi (Ctrl+C) e abbassa N_DRAWS.\n"
    )


def aggiungi_tabelle(
    rac: RaccoglitoreTabelle,
    scena: Scene,
    n_draws: int = N_DRAWS,
    rng_seed: int = RNG_SEED,
    srf_satellite: str = SRF_SATELLITE,
) -> dict:
    """Esegue l'analisi di incertezza (§4.5.1.2-4.5.1.5) e aggiunge (e
    stampa) le relative tabelle al raccoglitore.

    Restituisce mc_baseline (la Monte Carlo AOD-only, Maritime x Boxcar):
    report_en.py la riusa per calcolare U_propria nella tabella En senza
    rilanciare la stessa Monte Carlo una seconda volta."""
    radiance, mask, meta, atmos = scena.radiance, scena.mask, scena.meta, scena.atmos

    stima_tempi(meta, atmos, n_draws)

    rac.aggiungi(
        "Informazioni sull'analisi di incertezza",
        ["Prodotto", "N draws", "Seed", "SRF reale usata"],
        [[scena.ingested.product.name, n_draws, rng_seed, srf_satellite]],
    )

    # -- 4.5.1.2: incertezza da AOD (baseline Maritime x Boxcar) -----------
    mc_baseline = monte_carlo_area_means(
        radiance, meta, atmos, mask,
        aero_profile=AeroProfile.Maritime, spectral_mode="boxcar",
        n_draws=n_draws, rng_seed=rng_seed, label="Maritime x Boxcar",
    )
    tabella_4_5_1_2(rac, mc_baseline)

    # -- 4.5.1.3: sensibilità a profilo aerosol (AOD fisso, Boxcar) --------
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

    # -- 4.5.1.4: sensibilità a SRF (Maritime, AOD fisso) --------------------
    scenario_boxcar = scenari_profilo["Maritime"]  # già calcolato sopra
    scenario_srf = compute_scenario(
        radiance, meta, atmos, mask,
        aero_profile=AeroProfile.Maritime, spectral_mode="srf", srf_satellite=srf_satellite,
    )
    tabella_sensibilita(rac, scenario_boxcar, scenario_srf, srf_satellite)

    # -- 4.5.1.5: incertezza combinata (profilo x SRF) ------------------------
    combinazioni = {"Maritime × Boxcar (riferimento)": mc_baseline}
    combinazioni["Continental × Boxcar"] = monte_carlo_area_means(
        radiance, meta, atmos, mask,
        aero_profile=AeroProfile.Continental, spectral_mode="boxcar",
        n_draws=n_draws, rng_seed=rng_seed, label="Continental x Boxcar",
    )
    combinazioni["Maritime × SRF reale (se disponibile)"] = monte_carlo_area_means(
        radiance, meta, atmos, mask,
        aero_profile=AeroProfile.Maritime, spectral_mode="srf", srf_satellite=srf_satellite,
        n_draws=n_draws, rng_seed=rng_seed, label="Maritime x SRF reale",
    )
    tabella_incertezza_combinata(rac, combinazioni)

    return mc_baseline


def main() -> None:
    load_dotenv()
    scena = load_scene()
    rac = RaccoglitoreTabelle()
    mc_baseline = aggiungi_tabelle(rac, scena)

    from report_en import tabella_en  # import qui: report_en dipende da
                                       # report_risultati, che a sua volta
                                       # non deve dipendere da questo file
    tabella_en(rac, scena, mc_baseline)

    rac.scrivi_markdown(OUTPUT_PATH)
    print(f"\nTabelle scritte in {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
