# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-spirit-vs-vitality-bo3-KtWhzrlsNkWCCS0U9BIlr3/spirit-vs-vitality-m2-nuke.csv`
- round_num: `10`

## Largest probability jumps

- tick `80663`, seconds `70.50`, LSTM `0.3170`, delta `-0.2154`
- tick `80695`, seconds `71.00`, LSTM `0.1976`, delta `-0.1194`
- tick `80759`, seconds `72.00`, LSTM `0.1203`, delta `-0.0701`
- tick `80887`, seconds `74.00`, LSTM `0.0417`, delta `-0.0283`
- tick `79319`, seconds `49.50`, LSTM `0.4900`, delta `-0.0240`
- tick `80791`, seconds `72.50`, LSTM `0.0982`, delta `-0.0221`
- tick `79575`, seconds `53.50`, LSTM `0.5061`, delta `+0.0209`
- tick `80855`, seconds `73.50`, LSTM `0.0700`, delta `-0.0196`
- tick `76727`, seconds `9.00`, LSTM `0.4687`, delta `+0.0145`
- tick `76311`, seconds `2.50`, LSTM `0.4720`, delta `-0.0145`

## Top 15 local ridge features

- `lag_02__T_place_TROPHY`: coefficient `-0.003017`, |coef| `0.003017`
- `lag_10__T_place_SQUEAKY`: coefficient `0.002722`, |coef| `0.002722`
- `lag_12__T_place_VENDING`: coefficient `-0.002699`, |coef| `0.002699`
- `lag_11__T_place_SQUEAKY`: coefficient `0.002122`, |coef| `0.002122`
- `lag_03__T_place_TROPHY`: coefficient `-0.001949`, |coef| `0.001949`
- `lag_13__T_place_VENDING`: coefficient `-0.001914`, |coef| `0.001914`
- `lag_07__T2__duck_amount`: coefficient `0.001671`, |coef| `0.001671`
- `lag_04__CT_flashed_players`: coefficient `-0.001606`, |coef| `0.001606`
- `lag_00__CT4__alive`: coefficient `0.001601`, |coef| `0.001601`
- `lag_00__CT_place_RAMP`: coefficient `0.001588`, |coef| `0.001588`
- `lag_00__CT4__hp`: coefficient `0.001577`, |coef| `0.001577`
- `lag_01__T_place_LOBBY`: coefficient `0.001565`, |coef| `0.001565`
- `lag_04__CT3__is_scoped`: coefficient `0.001507`, |coef| `0.001507`
- `lag_00__T_kills_last_3s`: coefficient `-0.001487`, |coef| `0.001487`
- `lag_02__T_place_CONTROL`: coefficient `-0.001458`, |coef| `0.001458`

## Top 10 utility ridge features

- `lag_13__T2__smoke`: coefficient `0.001261` (raises CT win probability)
- `lag_04__CT3__flash_duration`: coefficient `-0.000951` (lowers CT win probability)
- `lag_14__T2__smoke`: coefficient `0.000822` (raises CT win probability)
- `lag_08__T_B_site_active_smokes`: coefficient `-0.000752` (lowers CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `0.000736` (raises CT win probability)
- `lag_07__T2__flash`: coefficient `0.000719` (raises CT win probability)
- `lag_08__T_A_site_active_smokes`: coefficient `-0.000701` (lowers CT win probability)
- `lag_11__CT_B_site_active_smokes`: coefficient `0.000646` (raises CT win probability)
- `lag_02__CT1__flash_duration`: coefficient `0.000633` (raises CT win probability)
- `lag_11__CT_A_site_active_smokes`: coefficient `0.000620` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_02__T_place_TROPHY`: coefficient `-0.003017` (lowers CT win probability)
- `lag_10__T_place_SQUEAKY`: coefficient `0.002722` (raises CT win probability)
- `lag_12__T_place_VENDING`: coefficient `-0.002699` (lowers CT win probability)
- `lag_11__T_place_SQUEAKY`: coefficient `0.002122` (raises CT win probability)
- `lag_03__T_place_TROPHY`: coefficient `-0.001949` (lowers CT win probability)
- `lag_13__T_place_VENDING`: coefficient `-0.001914` (lowers CT win probability)
- `lag_07__T2__duck_amount`: coefficient `0.001671` (raises CT win probability)
- `lag_04__CT_flashed_players`: coefficient `-0.001606` (lowers CT win probability)
- `lag_00__CT4__alive`: coefficient `0.001601` (raises CT win probability)
- `lag_00__CT_place_RAMP`: coefficient `0.001588` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `80663`, seconds `70.50`, LSTM delta `-0.2154`

Top all feature movements:
- `lag_02__T_place_TROPHY`: contribution `-0.019131`
- `lag_10__T_place_SQUEAKY`: contribution `-0.016950`
- `lag_12__T_place_VENDING`: contribution `-0.013682`
- `lag_01__T_place_VENDING`: contribution `-0.007252`
- `lag_04__CT_flashed_players`: contribution `-0.007033`

Top utility-only movements:
- `lag_04__CT3__flash_duration`: contribution `-0.002810`

### tick `80695`, seconds `71.00`, LSTM delta `-0.1194`

Top all feature movements:
- `lag_11__T_place_SQUEAKY`: contribution `-0.013214`
- `lag_03__T_place_TROPHY`: contribution `-0.012361`
- `lag_13__T_place_VENDING`: contribution `-0.009706`
- `lag_00__T_place_TROPHY`: contribution `-0.007109`
- `lag_04__T_place_VENDING`: contribution `-0.005402`

Top utility-only movements:
- `lag_00__CT3__flash_duration`: contribution `-0.002176`

### tick `80759`, seconds `72.00`, LSTM delta `-0.0701`

Top all feature movements:
- `lag_02__T_place_TROPHY`: contribution `-0.019131`
- `lag_00__T_place_TROPHY`: contribution `+0.014217`
- `lag_00__T_place_CONTROL`: contribution `-0.013765`
- `lag_01__T_place_VENDING`: contribution `+0.007252`
- `lag_01__T_place_TROPHY`: contribution `-0.006837`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `80887`, seconds `74.00`, LSTM delta `-0.0283`

Top all feature movements:
- `lag_04__T_place_CONTROL`: contribution `-0.014605`
- `lag_00__T_place_CONTROL`: contribution `+0.013765`
- `lag_01__T_place_TROPHY`: contribution `+0.006837`
- `lag_03__T_place_CONTROL`: contribution `-0.006388`
- `lag_00__T_place_RAMP`: contribution `-0.005894`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `79319`, seconds `49.50`, LSTM delta `-0.0240`

Top all feature movements:
- `lag_00__T_place_DECON`: contribution `-0.014422`
- `lag_00__CT_place_MINI`: contribution `-0.006234`
- `lag_01__CT5__is_walking`: contribution `+0.002627`
- `lag_06__CT5__is_walking`: contribution `+0.002621`
- `lag_11__CT1__is_walking`: contribution `-0.002130`

Top utility-only movements:
- No utility movement among the top local contributors.
