from __future__ import annotations

from pathlib import Path


class RaccoglitoreTabelle:

    def __init__(self, autosave: str | Path | None = None) -> None:
        self._tabelle: list[tuple[str, list[str], list[list[str]]]] = []
        self._autosave = Path(autosave) if autosave else None

    def aggiungi(self, titolo, intestazioni, righe) -> None:
        righe_str = [[self._cella(c) for c in riga] for riga in righe]
        self._tabelle.append((titolo, list(intestazioni), righe_str))
        if self._autosave:
            self.scrivi_markdown(self._autosave)

    @staticmethod
    def _cella(c) -> str:
        if isinstance(c, bool):
            return "Sì" if c else "No"
        if isinstance(c, float):
            return f"{c:.4g}" if c != 0 and abs(c) < 1e-3 else f"{c:.4f}"
        return str(c)

    @staticmethod
    def _tabella_markdown(intestazioni: list[str], righe: list[list[str]]) -> str:
        header = "| " + " | ".join(intestazioni) + " |"
        sep = "| " + " | ".join("---" for _ in intestazioni) + " |"
        corpo = "".join("\n| " + " | ".join(r) + " |" for r in righe)
        return header + "\n" + sep + corpo

    def markdown(self) -> str:
        blocchi = [f"#### {t}\n\n" + self._tabella_markdown(i, r) for t, i, r in self._tabelle]
        return "\n\n".join(blocchi) + "\n"

    def scrivi_markdown(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(self.markdown(), encoding="utf-8")
        return path
