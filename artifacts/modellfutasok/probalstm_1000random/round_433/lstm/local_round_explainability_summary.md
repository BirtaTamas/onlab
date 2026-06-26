# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-legacy-bo3-o9hWAn-ugamRsSw8ngEfHF/astralis-vs-legacy-m3-ancient.csv`
- round_num: `1`

## Largest probability jumps

- tick `2079`, seconds `13.00`, LSTM `0.0695`, delta `-0.2709`
- tick `2047`, seconds `12.50`, LSTM `0.3404`, delta `-0.0568`
- tick `1983`, seconds `11.50`, LSTM `0.4060`, delta `-0.0495`
- tick `2271`, seconds `16.00`, LSTM `0.0252`, delta `-0.0403`
- tick `5023`, seconds `59.00`, LSTM `0.0524`, delta `+0.0365`
- tick `1951`, seconds `11.00`, LSTM `0.4555`, delta `-0.0338`
- tick `2111`, seconds `13.50`, LSTM `0.0390`, delta `-0.0305`
- tick `5151`, seconds `61.00`, LSTM `0.0492`, delta `-0.0264`
- tick `2175`, seconds `14.50`, LSTM `0.0735`, delta `+0.0254`
- tick `2431`, seconds `18.50`, LSTM `0.0060`, delta `-0.0178`

## Top 15 local ridge features

- `lag_15__T_place_WATER`: coefficient `-0.002530`, |coef| `0.002530`
- `lag_00__CT_place_UNKNOWN`: coefficient `0.002316`, |coef| `0.002316`
- `lag_15__T_place_TUNNEL`: coefficient `0.002209`, |coef| `0.002209`
- `lag_05__CT_place_MAINHALL`: coefficient `-0.001804`, |coef| `0.001804`
- `lag_14__T_place_RUINS`: coefficient `-0.001733`, |coef| `0.001733`
- `lag_00__T4__flash_duration`: coefficient `-0.001587`, |coef| `0.001587`
- `lag_13__T_place_RUINS`: coefficient `-0.001380`, |coef| `0.001380`
- `lag_09__T_place_RUINS`: coefficient `-0.001167`, |coef| `0.001167`
- `lag_02__CT_place_UNKNOWN`: coefficient `0.001118`, |coef| `0.001118`
- `lag_13__T_place_WATER`: coefficient `0.001048`, |coef| `0.001048`
- `lag_06__T_place_WATER`: coefficient `0.001007`, |coef| `0.001007`
- `lag_06__T_place_RAMP`: coefficient `-0.000972`, |coef| `0.000972`
- `lag_00__T_flashed_players`: coefficient `-0.000941`, |coef| `0.000941`
- `lag_07__T_place_TSIDELOWER`: coefficient `-0.000929`, |coef| `0.000929`
- `lag_14__T_place_TUNNEL`: coefficient `0.000897`, |coef| `0.000897`

## Top 10 utility ridge features

- `lag_00__T4__flash_duration`: coefficient `-0.001587` (lowers CT win probability)
- `lag_03__CT1__flash_duration`: coefficient `-0.000882` (lowers CT win probability)
- `lag_00__T_flash_duration_sum`: coefficient `-0.000857` (lowers CT win probability)
- `lag_05__CT1__flash_duration`: coefficient `-0.000795` (lowers CT win probability)
- `lag_04__CT1__flash_duration`: coefficient `-0.000769` (lowers CT win probability)
- `lag_12__CT1__flash_duration`: coefficient `-0.000696` (lowers CT win probability)
- `lag_13__CT1__flash_duration`: coefficient `-0.000686` (lowers CT win probability)
- `lag_11__CT1__flash_duration`: coefficient `-0.000676` (lowers CT win probability)
- `lag_00__CT3__flash`: coefficient `0.000662` (raises CT win probability)
- `lag_02__CT1__flash_duration`: coefficient `-0.000661` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_15__T_place_WATER`: coefficient `-0.002530` (lowers CT win probability)
- `lag_00__CT_place_UNKNOWN`: coefficient `0.002316` (raises CT win probability)
- `lag_15__T_place_TUNNEL`: coefficient `0.002209` (raises CT win probability)
- `lag_05__CT_place_MAINHALL`: coefficient `-0.001804` (lowers CT win probability)
- `lag_14__T_place_RUINS`: coefficient `-0.001733` (lowers CT win probability)
- `lag_13__T_place_RUINS`: coefficient `-0.001380` (lowers CT win probability)
- `lag_09__T_place_RUINS`: coefficient `-0.001167` (lowers CT win probability)
- `lag_02__CT_place_UNKNOWN`: coefficient `0.001118` (raises CT win probability)
- `lag_13__T_place_WATER`: coefficient `0.001048` (raises CT win probability)
- `lag_06__T_place_WATER`: coefficient `0.001007` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `2079`, seconds `13.00`, LSTM delta `-0.2709`

Top all feature movements:
- `lag_15__T_place_WATER`: contribution `-0.028882`
- `lag_15__T_place_TUNNEL`: contribution `-0.026836`
- `lag_05__CT_place_MAINHALL`: contribution `-0.014929`
- `lag_13__T_place_WATER`: contribution `-0.011962`
- `lag_00__T4__flash_duration`: contribution `-0.011019`

Top utility-only movements:
- `lag_00__T4__flash_duration`: contribution `-0.011019`
- `lag_00__T_flash_duration_sum`: contribution `-0.002818`

### tick `2047`, seconds `12.50`, LSTM delta `-0.0568`

Top all feature movements:
- `lag_15__T_place_WATER`: contribution `-0.028882`
- `lag_15__T_place_TUNNEL`: contribution `-0.026836`
- `lag_14__T_place_TUNNEL`: contribution `-0.010894`
- `lag_12__T_place_WATER`: contribution `+0.007646`
- `lag_13__T_place_RUINS`: contribution `-0.007342`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `1983`, seconds `11.50`, LSTM delta `-0.0495`

Top all feature movements:
- `lag_13__T_place_WATER`: contribution `+0.011962`
- `lag_12__T_place_WATER`: contribution `-0.007646`
- `lag_06__T_place_WATER`: contribution `-0.005749`
- `lag_13__T_place_TUNNEL`: contribution `-0.005195`
- `lag_02__CT_place_MAINHALL`: contribution `-0.005151`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `2271`, seconds `16.00`, LSTM delta `-0.0403`

Top all feature movements:
- `lag_15__T_place_WATER`: contribution `+0.014441`
- `lag_03__CT1__flash_duration`: contribution `-0.005127`
- `lag_13__T_place_RUINS`: contribution `+0.003671`
- `lag_07__T_place_TSIDELOWER`: contribution `-0.003483`
- `lag_00__T_place_TSIDELOWER`: contribution `-0.003326`

Top utility-only movements:
- `lag_03__CT1__flash_duration`: contribution `-0.005127`
- `lag_06__T4__flash_duration`: contribution `-0.001712`
- `lag_00__CT3__flash_duration`: contribution `-0.001464`

### tick `5023`, seconds `59.00`, LSTM delta `+0.0365`

Top all feature movements:
- `lag_02__T_place_TUNNEL`: contribution `+0.004125`
- `lag_01__T_place_TUNNEL`: contribution `+0.003709`
- `lag_07__T_place_TSIDELOWER`: contribution `-0.003483`
- `lag_03__T_place_WATER`: contribution `+0.002920`
- `lag_04__T_place_WATER`: contribution `+0.002556`

Top utility-only movements:
- No utility movement among the top local contributors.
