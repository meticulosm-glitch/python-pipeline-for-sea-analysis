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
    N_GRIGLIA_6S,
    build_6s_coefficients,
    compute_scenario,
    monte_carlo_area_means_fast,
)
from tabelle import RaccoglitoreTabelle

N_DRAWS = 1_000_000   # JCGM 101 §7.2: 10^6 per un intervallo al 95%
RNG_SEED = 42
SRF_SATELLITE = "S2A"
OUTPUT_PATH = Path("tabelle_incertezza_4_5_1.md")


# --------------------------------------------------------------------------
# Statistiche di una distribuzione Monte Carlo
# --------------------------------------------------------------------------

def shortest_interval(draws: np.ndarray, p: float = 0.95) -> tuple[float, float]:
    """Intervallo di copertura più corto con probabilità p (JCGM 101 §7.7)."""
    x = np.sort(draws)
    n = x.size
    k = int(np.ceil(p * n))                       # numero di valori dentro l'intervallo
    i = int(np.argmin(x[k - 1:] - x[:n - k + 1]))
    return float(x[i]), float(x[i + k - 1])


def mc_stats(draws: np.ndarray, n_max_shapiro: int = 5000, seed: int = RNG_SEED) -> dict:
    """Statistiche di una serie di estrazioni Monte Carlo (medie d'area per
    estrazione): media, mediana, sigma, 2 sigma, percentili 2,5/97,5 (intervallo
    probabilisticamente simmetrico al 95%), intervallo più corto al 95%, IQR e
    p-value di Shapiro-Wilk. Shapiro è calcolato su un sottocampione di
    n_max_shapiro estrazioni: scipy non lo supporta oltre 5000 valori (stesso
    criterio di common.describe)."""
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
    """Numero con `cifre` cifre significative e virgola decimale (per la tesi)."""
    if x == 0 or not np.isfinite(x):
        return f"{x:g}".replace(".", ",")
    decimali = max(0, cifre - 1 - int(np.floor(np.log10(abs(x)))))
    return f"{x:.{decimali}f}".replace(".", ",").replace("-", "−")


def _grandezze(scena: Scene, mc: dict) -> list[tuple[str, float, np.ndarray]]:
    """(nome, valore deterministico, estrazioni MC) per bande e indici."""
    ris, mask = scena.risultati, scena.mask
    g = [(f"ρ_BOA - {b}", _media_mascherata(ris["boa"][b], mask), mc["boa"][b]) for b in BAND_IDS]
    g.append(("NDCI", _media_mascherata(ris["ndci"], mask), mc["ndci"]))
    g.append(("Indice di torbidità", _media_mascherata(ris["turbidity"], mask), mc["turbidity"]))
    return g


# --------------------------------------------------------------------------
# Tabelle (schermo + markdown), nello stesso formato dei placeholder in tesi
# --------------------------------------------------------------------------

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
    # 3 Monte Carlo (baseline + Continental x Boxcar + Maritime x SRF), ognuna con
    # N_GRIGLIA_6S nodi + 5 punti di verifica per banda: non dipende da n_draws
    n_mc_runs = 3 * (N_GRIGLIA_6S + 5) * 4
    n_scenario_runs = 3 * 4 + 4  # 3 profili aerosol (§4.5.1.3) + 1 SRF (§4.5.1.4), 4 bande
    totale = single_run_s * (n_mc_runs + n_scenario_runs)
    print(
        f"\nUn run di 6S impiega ~{single_run_s:.2f}s su questa macchina. "
        f"Stima per l'analisi di incertezza (~{n_mc_runs + n_scenario_runs} run totali, "
        f"n_draws={n_draws}): ~{totale / 60:.1f} minuti.\n"
        "A questo si aggiungono alcuni minuti per combinazione per la griglia fine "
        "e la verifica (dipende dal numero di pixel d'acqua).\n"
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

    # -- 4.5.1.2: incertezza da AOD (baseline Maritime x Boxcar) -----------
    mc_baseline = monte_carlo_area_means_fast(
        radiance, meta, atmos, mask,
        aero_profile=AeroProfile.Maritime, spectral_mode="boxcar",
        n_draws=n_draws, rng_seed=rng_seed, label="Maritime x Boxcar",
    )
    tabella_info_mc(rac, scena, mc_baseline, rng_seed, srf_satellite)
    tabella_4_5_1_2(rac, scena, mc_baseline)

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
    combinazioni["Continental × Boxcar"] = monte_carlo_area_means_fast(
        radiance, meta, atmos, mask,
        aero_profile=AeroProfile.Continental, spectral_mode="boxcar",
        n_draws=n_draws, rng_seed=rng_seed, label="Continental x Boxcar",
    )
    combinazioni["Maritime × SRF reale (se disponibile)"] = monte_carlo_area_means_fast(
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
