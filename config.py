from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path


@dataclass(frozen=True)
class Area:
    nome: str
    bbox: tuple[float, float, float, float]
    data_inizio: dt.date
    data_fine: dt.date
    descrizione: str = ""
    prodotto_atteso: str | None = None
    sen2cor_granule: str | None = None

    def __post_init__(self) -> None:
        lon_min, lat_min, lon_max, lat_max = self.bbox
        if not (-180 <= lon_min < lon_max <= 180 and -90 <= lat_min < lat_max <= 90):
            raise ValueError(f"bbox non valido: {self.bbox} "
                             "(atteso lon_min lat_min lon_max lat_max)")
        if self.data_fine < self.data_inizio:
            raise ValueError("data_fine precede data_inizio")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.nome):
            raise ValueError(f"nome area non valido: {self.nome!r} (solo lettere, cifre, - _ .)")

    @property
    def centro(self) -> tuple[float, float]:
        lon_min, lat_min, lon_max, lat_max = self.bbox
        return (lat_min + lat_max) / 2, (lon_min + lon_max) / 2


AREE: dict[str, Area] = {
    "golfo-spezia": Area(
        nome="golfo-spezia",
        bbox=(9.78, 44.00, 9.92, 44.13),
        data_inizio=dt.date(2026, 9, 4),
        data_fine=dt.date(2026, 9, 4),
        descrizione="Golfo della Spezia, scena default (Sentinel-2C, 4/9/2026)",
        prodotto_atteso="S2C_MSIL1C_20260904T100551_N0512_R022_T32TNP_20260904T142309",
        sen2cor_granule="GRANULE/L2A_T32TNP_A010431_20260904T101800",
    ),
}


@dataclass(frozen=True)
class Precisione:
    nome: str
    n_estrazioni: int
    n_nodi_6s: int
    passo_griglia_fine: float
    n_verifica_estrazioni: int
    n_test_6s: int = 5
    n_semi_convergenza: int = 20
    descrizione: str = ""

    def __post_init__(self) -> None:
        if self.n_estrazioni < 100:
            raise ValueError("servono almeno 100 estrazioni")
        if self.n_nodi_6s < 4:
            raise ValueError("servono almeno 4 nodi di 6S (spline cubica)")
        if not 0 < self.passo_griglia_fine <= 0.01:
            raise ValueError("passo_griglia_fine fuori dall'intervallo (0; 0,01]")

    @property
    def n_convergenza(self) -> list[int]:
        ns = [10 ** k for k in range(2, 8) if 10 ** k < self.n_estrazioni]
        return ns + [self.n_estrazioni]


PRECISIONI: dict[str, Precisione] = {
    "rapida": Precisione(
        nome="rapida", n_estrazioni=10_000, n_nodi_6s=21, passo_griglia_fine=0.001,
        n_verifica_estrazioni=200, n_test_6s=3, n_semi_convergenza=10,
        descrizione="Prova veloce del flusso; percentili indicativi",
    ),
    "standard": Precisione(
        nome="standard", n_estrazioni=100_000, n_nodi_6s=41, passo_griglia_fine=0.0005,
        n_verifica_estrazioni=1000,
        descrizione="Intervalli al 95% stabili alla seconda cifra",
    ),
    "default": Precisione(
        nome="default", n_estrazioni=1_000_000, n_nodi_6s=41, passo_griglia_fine=0.0002,
        n_verifica_estrazioni=2000,
        descrizione="Valori della default: 10^6 estrazioni (JCGM 101 §7.2)",
    ),
}


@dataclass(frozen=True)
class Configurazione:
    area: Area
    precisione: Precisione
    seme: int = 42
    incertezza_aod: float = 0.11
    aod: float | None = None
    fonte_aod: str = "prodotto"
    max_nubi: float = 10.0
    srf_satellite: str = "S2A"
    use_s3: bool = False
    mappe: bool = True
    incertezza: bool = True
    convergenza: bool = True
    cartella_output: str | None = None

    @property
    def output_dir(self) -> Path:
        if self.cartella_output:
            return Path(self.cartella_output)
        return Path("risultati") / f"{self.area.nome}_{self.precisione.nome}"


    def to_dict(self) -> dict:
        d = asdict(self)
        d["area"]["data_inizio"] = self.area.data_inizio.isoformat()
        d["area"]["data_fine"] = self.area.data_fine.isoformat()
        d["area"]["bbox"] = list(self.area.bbox)
        return d

    def salva(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                        encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, d: dict) -> "Configurazione":
        a = dict(d["area"])
        a["bbox"] = tuple(a["bbox"])
        a["data_inizio"] = dt.date.fromisoformat(a["data_inizio"])
        a["data_fine"] = dt.date.fromisoformat(a["data_fine"])
        altri = {k: v for k, v in d.items() if k not in ("area", "precisione")}
        return cls(area=Area(**a), precisione=Precisione(**d["precisione"]), **altri)

    @classmethod
    def carica(cls, path: str | Path) -> "Configurazione":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def descrivi(self) -> str:
        a, p = self.area, self.precisione
        date = (a.data_inizio.isoformat() if a.data_inizio == a.data_fine
                else f"{a.data_inizio.isoformat()} → {a.data_fine.isoformat()}")
        return (f"Area: {a.nome}  bbox={a.bbox}  date={date}\n"
                f"Precisione: {p.nome}  N={p.n_estrazioni:,}  nodi 6S={p.n_nodi_6s}  "
                f"passo={p.passo_griglia_fine}  verifica={p.n_verifica_estrazioni}  "
                f"semi convergenza={p.n_semi_convergenza}\n"
                f"Seme={self.seme}  σ_AOD={self.incertezza_aod}  output={self.output_dir}")


__all__ = ["Area", "AREE", "Precisione", "PRECISIONI", "Configurazione", "replace", "field"]
