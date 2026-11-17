"""
tabelle.py
==========

Helper condiviso da esempio_esecuzione.py, report_incertezza.py e
run_reference_and_uncertainty_analysis (in processing.py) per mostrare i
risultati in tabelle leggibili a schermo e, se richiesto, esportarle in un
file .md che contiene SOLO le tabelle (nessun testo di contorno, nessuna
nota metodologica) - pronto da incollare in tesi al posto dei placeholder.

Uso tipico:

    from tabelle import RaccoglitoreTabelle

    rac = RaccoglitoreTabelle()
    rac.aggiungi(
        "§4.4.1 - Statistiche BOA",
        ["Banda", "Media", "Mediana"],
        [["B03", f"{0.0123:.4f}", f"{0.0110:.4f}"]],
    )
    ...
    rac.scrivi_markdown("risultati.md")

Ogni cella va già formattata dal chiamante (stringa) prima di essere
passata a `aggiungi`, così la stessa rappresentazione (n° di decimali,
notazione scientifica per i p-value molto piccoli, ecc.) compare
identica sia a schermo sia nel file .md - non c'è una doppia logica di
formattazione da tenere allineata.

Se il pacchetto 'tabulate' e' installato (pip install tabulate) le
tabelle a schermo usano un bordo a griglia; altrimenti si usa un fallback
a colonne allineate che non richiede dipendenze aggiuntive.
"""

from __future__ import annotations

from pathlib import Path

try:
    from tabulate import tabulate
    _HAS_TABULATE = True
except ImportError:  # fallback senza dipendenze extra
    _HAS_TABULATE = False


class RaccoglitoreTabelle:


    def __init__(self, autosave: str | Path | None = None) -> None:
        self._tabelle: list[tuple[str, list[str], list[list[str]]]] = []
        self._autosave = Path(autosave) if autosave else None

    def aggiungi(self, titolo, intestazioni, righe, stampa: bool = True) -> None:
        righe_str = [[self._cella(c) for c in riga] for riga in righe]
        self._tabelle.append((titolo, list(intestazioni), righe_str))
        if stampa:
            self._stampa_una(titolo, intestazioni, righe_str)
        if self._autosave:
            self.scrivi_markdown(self._autosave)

    @staticmethod
    def _cella(c) -> str:
        if isinstance(c, bool):
            return "Sì" if c else "No"
        if isinstance(c, float):
            # notazione compatta per valori molto piccoli (es. p-value),
            # altrimenti 4 decimali - stesso criterio ovunque nel progetto
            return f"{c:.4g}" if c != 0 and abs(c) < 1e-3 else f"{c:.4f}"
        return str(c)

    @staticmethod
    def _stampa_una(titolo: str, intestazioni: list[str], righe: list[list[str]]) -> None:
        print(f"\n{titolo}")
        if not righe:
            print("(nessun dato)")
            return
        if _HAS_TABULATE:
            print(tabulate(righe, headers=intestazioni, tablefmt="fancy_grid"))
        else:
            larghezze = [
                max(len(str(intestazioni[i])), max((len(r[i]) for r in righe), default=0))
                for i in range(len(intestazioni))
            ]
            riga_fmt = "  ".join(f"{{:<{w}}}" for w in larghezze)
            print(riga_fmt.format(*intestazioni))
            print("  ".join("-" * w for w in larghezze))
            for r in righe:
                print(riga_fmt.format(*r))

    @staticmethod
    def _tabella_markdown(intestazioni: list[str], righe: list[list[str]]) -> str:
        header = "| " + " | ".join(intestazioni) + " |"
        sep = "| " + " | ".join("---" for _ in intestazioni) + " |"
        if not righe:
            return header + "\n" + sep
        corpo = "\n".join("| " + " | ".join(r) + " |" for r in righe)
        return header + "\n" + sep + "\n" + corpo

    def markdown(self) -> str:
        """Ritorna SOLO le tabelle accumulate finora, in markdown (un
        titolo `####` per tabella, nessun altro testo)."""
        blocchi = [
            f"#### {titolo}\n\n" + self._tabella_markdown(intestazioni, righe)
            for titolo, intestazioni, righe in self._tabelle
        ]
        return "\n\n".join(blocchi) + "\n"

    def scrivi_markdown(self, path: str | Path) -> Path:
        """Scrive il file .md con solo le tabelle (sovrascrive se esiste)."""
        path = Path(path)
        path.write_text(self.markdown(), encoding="utf-8")
        return path
