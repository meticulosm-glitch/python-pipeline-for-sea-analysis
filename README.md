Unico punto di lancio: `python esempio_esecuzione.py` esegue in sequenza

  1. ingestion + elaborazione della scena (common.load_scene)
  2. tabelle di scena e risultati (report_risultati): stampate a schermo
     e scritte in risultati_tesi.md
  3. mappe e figure (report_mappe): cartella figure/
  4. tabelle di incertezza (report_incertezza, LENTA: molte run di 6S):
     aggiunte allo stesso risultati_tesi.md

Le parti veloci girano per prime, e il .md viene scritto dopo ogni fase:
se interrompi la parte di incertezza non perdi le tabelle già calcolate.

Prerequisiti

1. Dipendenze Python:
   pip install prefect Py6S numpy scipy rasterio boto3 cdsapi xarray netCDF4 python-dotenv tabulate matplotlib

2. Binario 6S compilato https://github.com/robintw/6S
3. Credenziali CDSE (Copernicus Data Space Ecosystem) da mettere in env
4. Credenziali CAMS/ADS da mettere in .cdsapirc
5. (per §3.5 e la mappa dei residui) prodotto Sen2Cor L2A: vedi common.py