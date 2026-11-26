from __future__ import annotations

from pathlib import Path

import numpy as np

from config import Configurazione
from tabelle import RaccoglitoreTabelle

GRANDEZZE = (("ndci", "NDCI"), ("turbidity", "Indice di torbidità"))
STATISTICHE = ("sigma", "p2_5", "p97_5")

BLU = "#257de0"
ARANCIO = "#f47230"
TESTO = "#191919"
TESTO_2 = "#666666"
GRIGLIA = "#e6e6e6"
DPI = 200


def nuove_estrazioni(mc: dict, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    info, tab = mc["info"], mc["tabulazione"]
    aods = np.maximum(0.0, rng.normal(info["aod_media"], info["aod_sigma"], n))
    return {k: np.interp(aods, tab["aod"], tab[k]) for k, _ in GRANDEZZE}


def _statistiche(x: np.ndarray) -> dict[str, float]:
    p025, p975 = np.percentile(x, [2.5, 97.5])
    return {"sigma": float(x.std(ddof=1)), "p2_5": float(p025), "p97_5": float(p975)}


def studio_convergenza(mc: dict, ns: list[int], n_semi: int, seme: int) -> dict:
    n_max = max(ns)
    rif = {k: {s: [] for s in STATISTICHE} for k, _ in GRANDEZZE}
    for k, _ in GRANDEZZE:
        serie = mc[k] if mc[k].size >= n_max else None
        if serie is None:
            raise ValueError("la Monte Carlo ha meno estrazioni del massimo N richiesto")
        for n in ns:
            for s, v in _statistiche(serie[:n]).items():
                rif[k][s].append(v)

    semi = {k: {s: np.empty((n_semi, len(ns))) for s in STATISTICHE} for k, _ in GRANDEZZE}
    for i, figlio in enumerate(np.random.SeedSequence(seme).spawn(n_semi)):
        d = nuove_estrazioni(mc, n_max, np.random.default_rng(figlio))
        for k, _ in GRANDEZZE:
            for j, n in enumerate(ns):
                for s, v in _statistiche(d[k][:n]).items():
                    semi[k][s][i, j] = v
    return {"ns": list(ns), "n_semi": n_semi, "seme": seme, "rif": rif, "semi": semi}


def tabella_convergenza(rac: RaccoglitoreTabelle, studio: dict) -> None:
    ns, n_semi = studio["ns"], studio["n_semi"]
    righe = []
    for k, nome in GRANDEZZE:
        rif, semi = studio["rif"][k], studio["semi"][k]
        for j, n in enumerate(ns):
            righe.append([
                nome, f"{n:,}".replace(",", "."),
                rif["sigma"][j], float(semi["sigma"][:, j].std(ddof=1)),
                rif["p2_5"][j], float(semi["p2_5"][:, j].std(ddof=1)),
                rif["p97_5"][j], float(semi["p97_5"][:, j].std(ddof=1)),
            ])
    rac.aggiungi(
        f"Convergenza del Monte Carlo al variare di N (seme {studio['seme']}; "
        f"dispersione = deviazione standard tra {n_semi} semi indipendenti)",
        ["Grandezza", "N", "σ", "Dispersione σ", "P2,5", "Dispersione P2,5",
         "P97,5", "Dispersione P97,5"],
        righe,
    )


def _stile_assi(ax, plt) -> None:
    ax.set_xscale("log")
    ax.grid(True, which="major", color=GRIGLIA, linewidth=0.8)
    ax.set_axisbelow(True)
    for lato in ("top", "right"):
        ax.spines[lato].set_visible(False)
    for lato in ("left", "bottom"):
        ax.spines[lato].set_color(TESTO_2)
    ax.tick_params(colors=TESTO_2, labelcolor=TESTO, labelsize=8)


def _serie(ax, ns, rif, semi, colore, etichetta) -> None:
    ax.fill_between(ns, semi.min(axis=0), semi.max(axis=0), color=colore, alpha=0.18,
                    linewidth=0)
    ax.plot(ns, rif, color=colore, linewidth=2, marker="o", markersize=5,
            markeredgecolor="white", markeredgewidth=1, label=etichetta)
    ax.axhline(rif[-1], color=colore, linewidth=1, linestyle=(0, (4, 3)), alpha=0.8)


def figura_convergenza(studio: dict, path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    ns = np.array(studio["ns"])
    titoli = {"sigma": "deviazione standard σ", "p2_5": "percentile 2,5%",
              "p97_5": "percentile 97,5%"}
    fig, axes = plt.subplots(len(GRANDEZZE), len(STATISTICHE), figsize=(12, 6.8),
                             constrained_layout=True, squeeze=False)
    for r, (k, nome) in enumerate(GRANDEZZE):
        for c, stat in enumerate(STATISTICHE):
            ax = axes[r, c]
            _serie(ax, ns, studio["rif"][k][stat], studio["semi"][k][stat], BLU,
                   f"seme {studio['seme']}")
            ax.set_title(f"{nome}: {titoli[stat]}", fontsize=10, color=TESTO, loc="left")
            _stile_assi(ax, plt)
            ax.set_xlabel("Numero di estrazioni N", color=TESTO, fontsize=9)
            ax.set_xlim(ns[0] * 0.7, ns[-1] * 1.4)

    h, _ = axes[0, 0].get_legend_handles_labels()
    banda = Patch(facecolor=BLU, alpha=0.18, label=f"min–max su {studio['n_semi']} semi")
    tratteggio = plt.Line2D([], [], color=BLU, linewidth=1, linestyle=(0, (4, 3)),
                            label=f"valore con N = {ns[-1]:,}".replace(",", "."))
    fig.legend(handles=h + [banda, tratteggio], loc="outside lower center", ncol=3,
               fontsize=9, frameon=False)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def figura_distribuzioni(mc: dict, ns: list[int], path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scelti = sorted({ns[0], ns[len(ns) // 2], ns[-1]})
    fig, axes = plt.subplots(len(GRANDEZZE), len(scelti), figsize=(3.6 * len(scelti), 6),
                             constrained_layout=True, squeeze=False)
    for r, (k, nome) in enumerate(GRANDEZZE):
        tutti = mc[k][:ns[-1]]
        bins = np.linspace(tutti.min(), tutti.max(), 60)
        for c, n in enumerate(scelti):
            ax = axes[r, c]
            x = mc[k][:n]
            ax.hist(x, bins=bins, density=True, color=BLU, edgecolor="white", linewidth=0.3)
            p025, p975 = np.percentile(x, [2.5, 97.5])
            for v in (p025, p975):
                ax.axvline(v, color=ARANCIO, linewidth=1.5, linestyle=(0, (4, 3)))
            ax.set_title(f"{nome}, N = {n:,}".replace(",", ".")
                         + f"\nσ = {x.std(ddof=1):.3g};  95%: [{p025:.3g}; {p975:.3g}]",
                         fontsize=9, color=TESTO, loc="left")
            ax.set_yticks([])
            for lato in ("top", "right", "left"):
                ax.spines[lato].set_visible(False)
            ax.spines["bottom"].set_color(TESTO_2)
            ax.tick_params(colors=TESTO_2, labelcolor=TESTO, labelsize=8)
            ax.set_xlabel(nome, color=TESTO, fontsize=9)
    linea = plt.Line2D([], [], color=ARANCIO, linewidth=1.5, linestyle=(0, (4, 3)),
                       label="percentili 2,5% e 97,5%")
    fig.legend(handles=[linea], loc="outside lower center", fontsize=9, frameon=False)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def figura_funzione_misura(mc: dict, path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tab, info = mc["tabulazione"], mc["info"]
    x_max = float(np.percentile(mc["aod"], 99.9))
    sel = tab["aod"] <= x_max
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 8), sharex=True, constrained_layout=True,
                             gridspec_kw={"height_ratios": [1, 2, 2]})

    ax = axes[0]
    a = mc["aod"]
    ax.hist(a[a > 0], bins=np.linspace(0, x_max, 80), density=True, color=BLU,
            edgecolor="white", linewidth=0.3)
    ax.set_title(f"Distribuzione dell'AOD (normale {info['aod_media']:.3f} ± "
                 f"{info['aod_sigma']:.2f}; {info['frazione_aod_zero']:.0%} delle estrazioni a 0)",
                 fontsize=9, color=TESTO, loc="left")
    ax.set_yticks([])

    for ax, (k, nome) in zip(axes[1:], GRANDEZZE):
        ax.plot(tab["aod"][sel], tab[k][sel], color=BLU, linewidth=1.5)
        ax.set_title(f"{nome} medio sui pixel d'acqua", fontsize=9, color=TESTO, loc="left")
        ax.set_ylabel(nome, color=TESTO, fontsize=9)
        ax.grid(True, color=GRIGLIA, linewidth=0.8)
        ax.set_axisbelow(True)

    for ax in axes:
        ax.axvline(info["aod_media"], color=ARANCIO, linewidth=1.2, linestyle=(0, (4, 3)))
        for lato in ("top", "right"):
            ax.spines[lato].set_visible(False)
        ax.tick_params(colors=TESTO_2, labelcolor=TESTO, labelsize=8)
    axes[-1].set_xlabel("AOD a 550 nm", color=TESTO, fontsize=9)
    linea = plt.Line2D([], [], color=ARANCIO, linewidth=1.2, linestyle=(0, (4, 3)),
                       label="AOD di riferimento (CAMS)")
    fig.legend(handles=[linea], loc="outside lower center", fontsize=9, frameon=False)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def salva_tabulazione(mc: dict, path: Path) -> Path:
    tab = mc["tabulazione"]
    chiavi = list(tab)
    np.savetxt(path, np.column_stack([tab[k] for k in chiavi]), delimiter=",",
               header=",".join(chiavi), comments="", fmt="%.8g")
    return path


def aggiungi_convergenza(rac: RaccoglitoreTabelle, mc_baseline: dict,
                         cfg: Configurazione, cartella_figure: Path) -> dict:
    p = cfg.precisione
    ns = p.n_convergenza
    studio = studio_convergenza(mc_baseline, ns, p.n_semi_convergenza, cfg.seme)
    tabella_convergenza(rac, studio)
    figura_convergenza(studio, cartella_figure / "convergenza_mc.png")
    figura_distribuzioni(mc_baseline, ns, cartella_figure / "distribuzioni_mc.png")
    figura_funzione_misura(mc_baseline, cartella_figure / "funzione_misura.png")
    salva_tabulazione(mc_baseline, cartella_figure.parent / "tabulazione_aod.csv")
    return studio
