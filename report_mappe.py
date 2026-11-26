from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rasterio.transform import array_bounds

from common import BAND_IDS
from processing import load_sen2cor_reflectance

DPI = 200

WAVELENGTH_NM = {"B03": 560, "B04": 665, "B05": 705, "B08": 842}

LAND_COLOR = "#d9d9d9"


def _extent(risultati: dict) -> tuple[float, float, float, float]:
    h, w = risultati["boa"]["B04"].shape
    west, south, east, north = array_bounds(h, w, risultati["grid_transform"])
    scale = 1000.0 if risultati["grid_crs"].is_projected else 1.0
    return west / scale, east / scale, south / scale, north / scale


def _axis_labels(ax, risultati: dict) -> None:
    if risultati["grid_crs"].is_projected:
        ax.set_xlabel("Est UTM (km)")
        ax.set_ylabel("Nord UTM (km)")
    else:
        ax.set_xlabel("Longitudine (°)")
        ax.set_ylabel("Latitudine (°)")


def _clean(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(float, copy=True)
    out[~np.isfinite(out)] = np.nan
    return out


def _water_only(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = _clean(arr)
    out[~mask] = np.nan
    return out


def _percentile_limits(arr: np.ndarray, lo: float = 2, hi: float = 98,
                        symmetric: bool = False) -> tuple[float, float]:
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(valid, [lo, hi])
    if symmetric:
        m = max(abs(vmin), abs(vmax))
        return -m, m
    return float(vmin), float(vmax)


def _draw_land(ax, mask: np.ndarray, extent) -> None:
    land = np.where(mask, np.nan, 1.0)
    ax.imshow(land, cmap=matplotlib.colors.ListedColormap([LAND_COLOR]),
              extent=extent, origin="upper", interpolation="nearest")


def _map_panel(ax, fig, data: np.ndarray, mask: np.ndarray, extent, risultati,
               title: str, cmap: str, cbar_label: str,
               symmetric: bool = False) -> None:
    _draw_land(ax, mask, extent)
    vmin, vmax = _percentile_limits(data[mask], symmetric=symmetric)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent,
                   origin="upper", interpolation="nearest")
    ax.set_title(title, fontsize=10)
    _axis_labels(ax, risultati)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label + "\n(scala: 2°-98° percentile)", fontsize=8)


def figura_mappa_boa(risultati: dict, path: Path) -> None:
    mask = risultati["water_mask"]
    extent = _extent(risultati)
    b04 = _clean(risultati["boa"]["B04"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

    vmin, vmax = _percentile_limits(b04)
    im1 = ax1.imshow(b04, cmap="gray", vmin=max(vmin, 0), vmax=vmax,
                     extent=extent, origin="upper", interpolation="nearest")
    ax1.contour(mask.astype(float), levels=[0.5], colors="cyan", linewidths=0.6,
                extent=extent, origin="upper")
    ax1.set_title("(a) ρ_BOA B04 — contorno maschera d'acqua (ciano)", fontsize=10)
    _axis_labels(ax1, risultati)
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.set_label("ρ_BOA B04 (2°-98° percentile)", fontsize=8)

    _map_panel(ax2, fig, _water_only(b04, mask), mask, extent, risultati,
               "(b) ρ_BOA B04 — solo pixel d'acqua", "viridis", "ρ_BOA B04")

    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def figura_spettro(risultati: dict, path: Path) -> None:
    mask = risultati["water_mask"]
    lam, med, medie, q1s, q3s = [], [], [], [], []
    for band_id in BAND_IDS:
        v = _water_only(risultati["boa"][band_id], mask)
        v = v[np.isfinite(v)]
        q1, q3 = np.percentile(v, [25, 75])
        lam.append(WAVELENGTH_NM[band_id])
        med.append(np.median(v))
        medie.append(v.mean())
        q1s.append(q1)
        q3s.append(q3)

    med, medie = np.array(med), np.array(medie)
    yerr = np.vstack([med - np.array(q1s), np.array(q3s) - med])

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.errorbar(lam, med, yerr=yerr, fmt="o-", capsize=4, label="Mediana ± IQR")
    ax.plot(lam, medie, "x--", color="tab:red", label="Media")
    ax.set_xticks(lam)
    ax.set_xticklabels([f"{b}\n{WAVELENGTH_NM[b]} nm" for b in BAND_IDS])
    ax.set_ylabel("ρ_BOA")
    ax.set_title("Spettro di riflettanza BOA, pixel d'acqua", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def figura_indicatori(risultati: dict, path: Path) -> None:
    mask = risultati["water_mask"]
    extent = _extent(risultati)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    _map_panel(ax1, fig, _water_only(risultati["ndci"], mask), mask, extent,
               risultati, "NDCI", "RdYlGn", "NDCI")
    _map_panel(ax2, fig, _water_only(risultati["turbidity"], mask), mask, extent,
               risultati, "Indice di torbidità (B08/B03)", "YlOrBr",
               "B08/B03")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def figura_residui_ndci(risultati: dict, sen2cor: dict, path: Path) -> None:
    mask = risultati["water_mask"]
    extent = _extent(risultati)
    residui = _clean(risultati["ndci"]) - _clean(sen2cor["ndci"])
    residui = np.where(mask, residui, np.nan)

    fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
    _map_panel(ax, fig, residui, mask, extent, risultati,
               "Residui NDCI (propria − Sen2Cor)", "RdBu_r",
               "ΔNDCI", symmetric=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def genera_figure(risultati: dict, out_dir: Path,
                  sen2cor_paths: dict[str, str] | None) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prodotte = []

    for nome, fn in (
        ("mappa_boa_b04.png", figura_mappa_boa),
        ("spettro_boa.png", figura_spettro),
        ("mappe_indicatori.png", figura_indicatori),
    ):
        p = out_dir / nome
        fn(risultati, p)
        prodotte.append(p)

    try:
        if sen2cor_paths is None:
            raise FileNotFoundError("prodotto Sen2Cor dell'area non trovato in GRANULE/")
        sen2cor = load_sen2cor_reflectance(risultati, sen2cor_paths)
    except FileNotFoundError:
        pass
    else:
        p = out_dir / "mappa_residui_ndci.png"
        figura_residui_ndci(risultati, sen2cor, p)
        prodotte.append(p)
    return prodotte


