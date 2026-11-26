from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

DEBUG = "--debug" in sys.argv
os.environ.setdefault("PREFECT_LOGGING_LEVEL", "INFO" if DEBUG else "CRITICAL")
os.environ.setdefault("PREFECT_LOGGING_TO_API_WHEN_MISSING_FLOW", "ignore")

from config import AREE, PRECISIONI, Area, Configurazione, replace


def _data(s: str) -> dt.date:
    try:
        return dt.date.fromisoformat(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"data non valida: {s!r} (formato AAAA-MM-GG)")


def crea_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pipeline Sentinel-2 per la qualità dell'acqua con analisi di incertezza.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Esempi in README.md.",
    )
    g = p.add_argument_group("area d'analisi")
    g.add_argument("--area", choices=sorted(AREE), default="golfo-spezia",
                   help="area predefinita (default: golfo-spezia)")
    g.add_argument("--bbox", nargs=4, type=float,
                   metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
                   help="riquadro personalizzato in gradi (richiede --data o --da/--a)")
    g.add_argument("--nome", help="nome dell'area (usato per la cartella di output)")
    g.add_argument("--data", type=_data, help="data della scena (AAAA-MM-GG)")
    g.add_argument("--da", type=_data, help="inizio dell'intervallo di ricerca")
    g.add_argument("--a", type=_data, help="fine dell'intervallo di ricerca (inclusa)")
    g.add_argument("--prodotto", help="identificativo L1C atteso (controllo facoltativo)")
    g.add_argument("--sen2cor", metavar="CARTELLA",
                   help="cartella GRANULE/L2A_... di Sen2Cor (default: ricerca automatica)")

    g = p.add_argument_group("precisione dell'analisi di incertezza")
    g.add_argument("--precisione", choices=list(PRECISIONI), default="default",
                   help="livello predefinito (default: default)")
    g.add_argument("--n-estrazioni", type=int, help="estrazioni Monte Carlo per combinazione")
    g.add_argument("--nodi-6s", type=int, help="valori di AOD in cui eseguire 6S")
    g.add_argument("--passo-griglia", type=float, help="passo della tabulazione in AOD")
    g.add_argument("--verifica", type=int, help="estrazioni ricalcolate senza tabulazione")
    g.add_argument("--semi-convergenza", type=int,
                   help="semi indipendenti nello studio di convergenza")

    g = p.add_argument_group("altri parametri")
    g.add_argument("--seme", type=int, help="seme del generatore (default 42)")
    g.add_argument("--aod", type=float, help="AOD a 550 nm fisso, senza scaricarlo da CAMS")
    g.add_argument("--incertezza-aod", type=float, help="1σ dell'AOD (default 0,11)")
    g.add_argument("--max-nubi", type=float, help="copertura nuvolosa massima in %% (default 10)")
    g.add_argument("--srf", choices=["S2A", "S2B"], help="SRF reale per la sensibilità")
    g.add_argument("--s3", action="store_true", help="scarica da CDSE S3 (servono le chiavi)")
    g.add_argument("--senza-mappe", action="store_true")
    g.add_argument("--senza-incertezza", action="store_true",
                   help="salta Monte Carlo, E_n e convergenza")
    g.add_argument("--senza-convergenza", action="store_true")
    g.add_argument("--output", metavar="CARTELLA", help="cartella dei risultati")
    g.add_argument("--config", metavar="FILE", help="configurazione.json da riusare "
                   "(le altre opzioni indicate la modificano)")
    g.add_argument("--elenca", action="store_true", help="elenca aree e precisioni ed esce")
    g.add_argument("--debug", action="store_true", help="log di Prefect e traceback completo")
    g.add_argument("--mostra-config", action="store_true",
                   help="stampa la configurazione risultante ed esce")
    return p


def configurazione_da_argomenti(args: argparse.Namespace) -> Configurazione:
    if args.config:
        cfg = Configurazione.carica(args.config)
        area, prec = cfg.area, cfg.precisione
    else:
        cfg = None
        area, prec = AREE[args.area], PRECISIONI[args.precisione]

    if args.data and (args.da or args.a):
        raise SystemExit("Usa --data oppure --da/--a, non entrambi.")
    if args.bbox:
        if not (args.data or (args.da and args.a)):
            raise SystemExit("Con --bbox indica anche --data oppure --da e --a.")
        area = Area(nome=args.nome or "personalizzata", bbox=tuple(args.bbox),
                    data_inizio=area.data_inizio, data_fine=area.data_fine)
    modifiche: dict = {}
    if args.data:
        modifiche.update(data_inizio=args.data, data_fine=args.data)
    if args.da or args.a:
        modifiche.update(data_inizio=args.da or area.data_inizio,
                         data_fine=args.a or area.data_fine)
    if modifiche and not args.bbox:
        modifiche.update(prodotto_atteso=None, sen2cor_granule=None)
        inizio, fine = modifiche["data_inizio"], modifiche["data_fine"]
        suffisso = f"{inizio:%Y%m%d}" + ("" if inizio == fine else f"-{fine:%Y%m%d}")
        modifiche["nome"] = f"{area.nome}-{suffisso}"
    if args.nome:
        modifiche["nome"] = args.nome
    if args.prodotto:
        modifiche["prodotto_atteso"] = args.prodotto
    if args.sen2cor:
        modifiche["sen2cor_granule"] = args.sen2cor
    if modifiche:
        area = replace(area, **modifiche)

    modifiche = {k: v for k, v in {
        "n_estrazioni": args.n_estrazioni, "n_nodi_6s": args.nodi_6s,
        "passo_griglia_fine": args.passo_griglia, "n_verifica_estrazioni": args.verifica,
        "n_semi_convergenza": args.semi_convergenza,
    }.items() if v is not None}
    if modifiche:
        nome = prec.nome if prec.nome.endswith("-mod") else f"{prec.nome}-mod"
        prec = replace(prec, nome=nome, descrizione="", **modifiche)

    base = cfg or Configurazione(area=area, precisione=prec)
    altri = {k: v for k, v in {
        "seme": args.seme, "incertezza_aod": args.incertezza_aod, "aod": args.aod,
        "max_nubi": args.max_nubi, "srf_satellite": args.srf, "cartella_output": args.output,
    }.items() if v is not None}
    if args.s3:
        altri["use_s3"] = True
    if args.senza_mappe:
        altri["mappe"] = False
    if args.senza_incertezza:
        altri["incertezza"] = False
    if args.senza_convergenza:
        altri["convergenza"] = False
    return replace(base, area=area, precisione=prec, **altri)


def elenca() -> None:
    print("Aree (--area):")
    for a in AREE.values():
        print(f"  {a.nome:<16} bbox={a.bbox}  {a.data_inizio}→{a.data_fine}  {a.descrizione}")
    print("\nPrecisioni (--precisione):")
    for p in PRECISIONI.values():
        print(f"  {p.nome:<10} N={p.n_estrazioni:>9,}  nodi 6S={p.n_nodi_6s:<3} "
              f"passo={p.passo_griglia_fine:<7} {p.descrizione}")


def esegui(cfg: Configurazione) -> None:
    from dotenv import load_dotenv

    from common import load_scene
    from report_risultati import aggiungi_tabelle as tabelle_risultati
    from tabelle import RaccoglitoreTabelle

    load_dotenv()
    rc = Path(__file__).resolve().parent / ".cdsapirc"
    if rc.is_file():
        os.environ.setdefault("CDSAPI_RC", str(rc))
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    cfg.salva(out / "configurazione.json")
    p = cfg.precisione
    print(f"{cfg.area.nome} | {p.nome} | N = {p.n_estrazioni:,}".replace(",", "."))

    rac = RaccoglitoreTabelle(autosave=out / "risultati_default.md")
    print("[1/5] scena")
    scena = load_scene(cfg)
    print("[2/5] tabelle")
    tabelle_risultati(rac, scena)

    if cfg.mappe:
        print("[3/5] mappe")
        try:
            from report_mappe import genera_figure
            genera_figure(scena.risultati, out / "figure", scena.sen2cor_paths)
        except Exception as exc:
            print(f"mappe non generate: {exc}")

    if cfg.incertezza:
        from report_en import tabella_en
        from report_incertezza import aggiungi_tabelle as tabelle_incertezza

        print("[4/5] incertezza")
        combinazioni = tabelle_incertezza(rac, scena, cfg)
        mc_baseline = next(iter(combinazioni.values()))
        tabella_en(rac, scena, mc_baseline, cfg.seme)

        if cfg.convergenza:
            from report_convergenza import aggiungi_convergenza
            print("[5/5] convergenza")
            aggiungi_convergenza(rac, mc_baseline, cfg, out / "figure")

    print(f"Risultati: {out.resolve()}")


def main(argv: list[str] | None = None) -> None:
    args = crea_parser().parse_args(argv)
    if args.elenca:
        elenca()
        return
    try:
        cfg = configurazione_da_argomenti(args)
    except ValueError as exc:
        raise SystemExit(f"Configurazione non valida: {exc}")
    if args.mostra_config:
        print(cfg.descrivi())
        return
    try:
        esegui(cfg)
    except KeyboardInterrupt:
        raise SystemExit("\ninterrotto")
    except Exception as exc:
        if args.debug:
            raise
        raise SystemExit(f"Errore: {type(exc).__name__}: {exc}\n(--debug per i dettagli)")


if __name__ == "__main__":
    main(sys.argv[1:])
