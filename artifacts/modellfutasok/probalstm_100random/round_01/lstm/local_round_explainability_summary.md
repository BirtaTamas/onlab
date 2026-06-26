# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-furia-vs-vitality-bo3-ZNzuF_vw0WBzn8QEbGrbgj/furia-vs-vitality-m1-overpass.csv`
- round_num: `20`

## Largest probability jumps

- tick `185878`, seconds `62.50`, LSTM `0.3418`, delta `-0.3421`
- tick `184662`, seconds `43.50`, LSTM `0.6837`, delta `-0.2334`
- tick `186326`, seconds `69.50`, LSTM `0.0289`, delta `-0.1360`
- tick `184822`, seconds `46.00`, LSTM `0.5378`, delta `-0.1188`
- tick `184182`, seconds `36.00`, LSTM `0.8041`, delta `+0.1091`
- tick `185910`, seconds `63.00`, LSTM `0.2422`, delta `-0.0996`
- tick `184854`, seconds `46.50`, LSTM `0.6050`, delta `+0.0672`
- tick `184726`, seconds `44.50`, LSTM `0.6486`, delta `-0.0574`
- tick `184982`, seconds `48.50`, LSTM `0.6937`, delta `+0.0528`
- tick `185942`, seconds `63.50`, LSTM `0.1922`, delta `-0.0500`

## Top 15 local ridge features

- `lag_07__T_place_CONSTRUCTION`: coefficient `-0.002321`, |coef| `0.002321`
- `lag_01__T1__flash_duration`: coefficient `0.002177`, |coef| `0.002177`
- `lag_00__T_kills_last_3s`: coefficient `-0.002170`, |coef| `0.002170`
- `lag_12__T1__flash_duration`: coefficient `-0.002130`, |coef| `0.002130`
- `lag_12__T5__flash_duration`: coefficient `-0.002082`, |coef| `0.002082`
- `lag_03__CT5__flash_duration`: coefficient `-0.002040`, |coef| `0.002040`
- `lag_00__kill_diff_last_3s`: coefficient `0.001987`, |coef| `0.001987`
- `lag_03__CT3__flash_duration`: coefficient `-0.001958`, |coef| `0.001958`
- `lag_01__T5__flash_duration`: coefficient `0.001940`, |coef| `0.001940`
- `lag_00__CT_place_STAIRS`: coefficient `-0.001870`, |coef| `0.001870`
- `lag_00__CT3__flash_duration`: coefficient `0.001864`, |coef| `0.001864`
- `lag_00__CT_place_SNIPERSNEST`: coefficient `0.001841`, |coef| `0.001841`
- `lag_05__T_place_CONSTRUCTION`: coefficient `-0.001783`, |coef| `0.001783`
- `lag_03__CT_flash_duration_sum`: coefficient `-0.001741`, |coef| `0.001741`
- `lag_00__CT_place_CONSTRUCTION`: coefficient `0.001733`, |coef| `0.001733`

## Top 10 utility ridge features

- `lag_01__T1__flash_duration`: coefficient `0.002177` (raises CT win probability)
- `lag_12__T1__flash_duration`: coefficient `-0.002130` (lowers CT win probability)
- `lag_12__T5__flash_duration`: coefficient `-0.002082` (lowers CT win probability)
- `lag_03__CT5__flash_duration`: coefficient `-0.002040` (lowers CT win probability)
- `lag_03__CT3__flash_duration`: coefficient `-0.001958` (lowers CT win probability)
- `lag_01__T5__flash_duration`: coefficient `0.001940` (raises CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `0.001864` (raises CT win probability)
- `lag_03__CT_flash_duration_sum`: coefficient `-0.001741` (lowers CT win probability)
- `lag_12__T_flash_duration_sum`: coefficient `-0.001685` (lowers CT win probability)
- `lag_01__T_flash_duration_sum`: coefficient `0.001586` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_07__T_place_CONSTRUCTION`: coefficient `-0.002321` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002170` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001987` (raises CT win probability)
- `lag_00__CT_place_STAIRS`: coefficient `-0.001870` (lowers CT win probability)
- `lag_00__CT_place_SNIPERSNEST`: coefficient `0.001841` (raises CT win probability)
- `lag_05__T_place_CONSTRUCTION`: coefficient `-0.001783` (lowers CT win probability)
- `lag_00__CT_place_CONSTRUCTION`: coefficient `0.001733` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001655` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001574` (raises CT win probability)
- `lag_06__T_place_CONSTRUCTION`: coefficient `-0.001570` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `185878`, seconds `62.50`, LSTM delta `-0.3421`

Top all feature movements:
- `lag_03__CT5__flash_duration`: contribution `-0.015008`
- `lag_00__CT_place_STAIRS`: contribution `-0.014554`
- `lag_01__T1__flash_duration`: contribution `-0.013620`
- `lag_12__T1__flash_duration`: contribution `-0.013323`
- `lag_12__T5__flash_duration`: contribution `-0.012779`

Top utility-only movements:
- `lag_03__CT5__flash_duration`: contribution `-0.015008`
- `lag_01__T1__flash_duration`: contribution `-0.013620`
- `lag_12__T1__flash_duration`: contribution `-0.013323`
- `lag_12__T5__flash_duration`: contribution `-0.012779`
- `lag_01__T5__flash_duration`: contribution `-0.011907`

### tick `184662`, seconds `43.50`, LSTM delta `-0.2334`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.024571`
- `lag_00__CT_place_CONSTRUCTION`: contribution `-0.021803`
- `lag_00__T_kills_last_3s`: contribution `-0.013748`
- `lag_13__CT_place_CONSTRUCTION`: contribution `-0.011684`
- `lag_07__CT_place_STORAGEROOM`: contribution `-0.009653`

Top utility-only movements:
- `lag_05__CT5__flash_duration`: contribution `-0.004480`
- `lag_00__CT_flash_duration_sum`: contribution `-0.002931`
- `lag_05__CT_flash_duration_sum`: contribution `-0.002715`

### tick `186326`, seconds `69.50`, LSTM delta `-0.1360`

Top all feature movements:
- `lag_07__T_place_CONSTRUCTION`: contribution `-0.028852`
- `lag_00__CT_place_WATER`: contribution `-0.008559`
- `lag_07__CT_place_SNIPERSNEST`: contribution `-0.007109`
- `lag_09__CT_place_STAIRS`: contribution `-0.006990`
- `lag_00__T_kills_last_3s`: contribution `-0.006874`

Top utility-only movements:
- `lag_14__CT3__flash_duration`: contribution `-0.002436`
- `lag_15__T5__flash_duration`: contribution `-0.002279`

### tick `184822`, seconds `46.00`, LSTM delta `-0.1188`

Top all feature movements:
- `lag_04__CT_place_STORAGEROOM`: contribution `-0.010490`
- `lag_12__CT_place_LOBBY`: contribution `-0.010310`
- `lag_05__CT_place_CONSTRUCTION`: contribution `-0.007181`
- `lag_06__CT_shots_fired_sum`: contribution `-0.006461`
- `lag_12__CT_place_STORAGEROOM`: contribution `-0.006231`

Top utility-only movements:
- `lag_09__T5__flash_duration`: contribution `-0.003071`
- `lag_00__CT_flash_duration_sum`: contribution `-0.002941`

### tick `184182`, seconds `36.00`, LSTM delta `+0.1091`

Top all feature movements:
- `lag_12__CT_place_LOBBY`: contribution `+0.010310`
- `lag_06__CT5__shots_fired`: contribution `+0.005566`
- `lag_00__kill_diff_last_3s`: contribution `+0.004783`
- `lag_06__CT_shots_fired_sum`: contribution `+0.004522`
- `lag_05__CT_place_CANAL`: contribution `+0.004335`

Top utility-only movements:
- No utility movement among the top local contributors.
