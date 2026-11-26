"""
esempio_esecuzione.py
======================

Unico punto di lancio: `python esempio_esecuzione.py` esegue in sequenza

  1. ingestion + elaborazione della scena (common.load_scene)
  2. tabelle di scena e risultati (report_risultati): stampate a schermo
     e scritte in risultati_tesi.md
  3. mappe e figure (report_mappe): cartella figure/
  4. tabelle di incertezza (report_incertezza, LENTA: ~570 run di 6S,
     6S tabulato su griglia di AOD e interpolato, 10^6 estrazioni):
     aggiunte allo stesso risultati_tesi.md
  5. tabella di compatibilità metrologica En vs Sen2Cor (report_en),
     E_n per r = 0; 0,5; 1 e intervallo al 95% della differenza simulata
     D = Y_propria − Y_rif: riusa la Monte Carlo del punto 4, nessuna run
     aggiuntiva di 6S

Le parti veloci girano per prime, e il .md viene scritto dopo ogni fase:
se interrompi la parte di incertezza non perdi le tabelle già calcolate.

Prerequisiti

1. Dipendenze Python:
   pip install prefect Py6S numpy scipy rasterio boto3 cdsapi xarray netCDF4 python-dotenv tabulate matplotlib

2. Binario 6S compilato https://github.com/robintw/6S
3. Credenziali CDSE (Copernicus Data Space Ecosystem) da mettere in env
4. Credenziali CAMS/ADS da mettere in .cdsapirc
5. (per §3.5, la tabella En e la mappa dei residui) prodotto Sen2Cor L2A: vedi common.py
"""

import traceback
from pathlib import Path

from dotenv import load_dotenv

from common import load_scene
from report_en import tabella_en
from report_incertezza import aggiungi_tabelle as tabelle_incertezza
from report_risultati import aggiungi_tabelle as tabelle_risultati
from tabelle import RaccoglitoreTabelle

# --------------------------------------------------------------------------
# Impostazioni
# --------------------------------------------------------------------------

MD_OUTPUT_PATH = Path("risultati_tesi.md")

GENERA_MAPPE = True
GENERA_TABELLE_INCERTEZZA = True   # False = salta la parte lenta (Monte Carlo)
                                    # (salta anche la tabella En: ha bisogno
                                    # della stessa Monte Carlo per U_propria)
N_DRAWS = 1_000_000                 # estrazioni Monte Carlo per combinazione (§3.2 tesi).
                                    # Con 6S tabulato il costo non dipende da N:
                                    # 10^6 è il valore indicato da JCGM 101 §7.2.


def main() -> None:
    load_dotenv()

    scena = load_scene(use_s3=False)  # True = più veloce ma richiede CDSE_S3_ACCESS_KEY/SECRET_KEY

    rac = RaccoglitoreTabelle(autosave=MD_OUTPUT_PATH)

    # -- 2. tabelle di scena e risultati (veloci) -------------------------
    tabelle_risultati(rac, scena)

    # -- 3. mappe (veloci) ------------------------------------------------
    if GENERA_MAPPE:
        print("\nGenerazione mappe...")
        try:
            # import qui (non in cima): se matplotlib manca, le tabelle
            # vengono prodotte comunque
            from report_mappe import genera_figure
            genera_figure(scena.risultati)
        except Exception:  # le mappe non devono far perdere il resto della run
            print("Mappe NON generate:")
            traceback.print_exc()

    # -- 4. incertezza (lenta) e 5. tabella En -----------------------------
    if GENERA_TABELLE_INCERTEZZA:
        mc_baseline = tabelle_incertezza(rac, scena, n_draws=N_DRAWS)
        tabella_en(rac, scena, mc_baseline)

    print(f"\nTabelle scritte in {MD_OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
