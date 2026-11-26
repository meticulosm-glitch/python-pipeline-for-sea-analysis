# Pipeline Sentinel-2 per la qualità dell'acqua

Unico punto di accesso: `esempio_esecuzione.py`.

```bash
python esempio_esecuzione.py                        # valori di default, golfo La Spezia
python esempio_esecuzione.py --precisione rapida    # prova veloce
python esempio_esecuzione.py --elenca               # aree e precisioni disponibili
python esempio_esecuzione.py --help                 # tutte le opzioni
```

## Cosa si può cambiare

**Area d'analisi** (predefinite in `config.py`, dizionario `AREE`):

```bash
python esempio_esecuzione.py --bbox 9.70 43.95 9.85 44.05 --data 2026-09-04 --nome cinque-terre
python esempio_esecuzione.py --bbox ... --da 2026-07-01 --a 2026-09-30 --nome estate
```

Con un intervallo di date viene scelta la scena con più copertura dell'area,
meno nubi e, a parità, la più recente. Il confronto con Sen2Cor usa il
prodotto L2A dello stesso tile e della stessa data, se è in `GRANULE/`
(oppure `--sen2cor <cartella>`); se manca, quelle tabelle vengono saltate.

**Precisione** (predefinite in `config.py`, dizionario `PRECISIONI`):

| Nome     | Estrazioni | Nodi 6S | Passo tabulazione | Uso |
|----------|-----------:|--------:|------------------:|-----|
| rapida   | 10.000     | 21      | 0,001             | prova del flusso |
| standard | 100.000    | 41      | 0,0005            | intervalli stabili alla seconda cifra |
| default     | 1.000.000  | 41      | 0,0002            | valori della default (JCGM 101) |

Ogni valore si può cambiare singolarmente (`--n-estrazioni`, `--nodi-6s`,
`--passo-griglia`, `--verifica`, `--semi-convergenza`), così come seme,
incertezza dell'AOD, nubi massime e SRF (`--seme`, `--incertezza-aod`,
`--max-nubi`, `--srf`). Si possono saltare fasi con `--senza-mappe`,
`--senza-incertezza`, `--senza-convergenza`.

## Risultati

Ogni esecuzione scrive in `risultati/<area>_<precisione>/`:

- `risultati_default.md`: tutte le tabelle (riscritto dopo ognuna)
- `figure/`: mappe, `convergenza_mc.png`, `distribuzioni_mc.png`, `funzione_misura.png`
- `tabulazione_aod.csv`: medie d'area in funzione dell'AOD
- `configurazione.json`: i parametri usati; per ripetere l'esecuzione
  `python esempio_esecuzione.py --config risultati/.../configurazione.json`

## Moduli

| File | Ruolo |
|------|-------|
| `esempio_esecuzione.py` | punto di accesso, riga di comando |
| `config.py` | aree, livelli di precisione, configurazione |
| `common.py` | caricamento della scena, ricerca del prodotto Sen2Cor |
| `ingestion.py` | ricerca e download da CDSE |
| `processing.py` | lettura, correzione atmosferica 6S, indici, Monte Carlo |
| `report_risultati.py` | tabelle di scena e confronto pixel per pixel |
| `report_mappe.py` | mappe |
| `report_incertezza.py` | Monte Carlo sull'AOD, sensibilità, intervalli di copertura |
| `report_en.py` | E_n e differenza simulata con Sen2Cor |
| `report_convergenza.py` | come cambiano i risultati al variare di N |
| `tabelle.py` | raccolta delle tabelle in Markdown |

## Prerequisiti

1. `pip install -r requirements.txt`
2. 6S compilato in `6S/build/sixs` (oppure indicato con `SIXS_PATH`):
   ```bash
   cd 6S && mkdir -p build
   gfortran -std=legacy -ffixed-line-length-none -fno-f2c -O3 -funroll-all-loops -ffpe-summary=none -o build/sixs src/*.f
   ```
   `src/main.f` è modificato (formato 944) per scrivere i coefficienti xa, xb, xc
   con 8 cifre significative invece di 5 decimali; l'originale è `src/main.f.orig`.
3. Credenziali CDSE (Copernicus Data Space Ecosystem) in `.env`
4. Credenziali CAMS/ADS in `.cdsapirc`
5. Per il confronto con Sen2Cor: prodotto L2A in `GRANULE/`
