# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-lynn-vision-vs-housebets-bo3-GrWDn9AJOxYQcZMXkSI-Tw/lynn-vision-vs-housebets-m2-dust2.csv`
- round_num: `13`

## Largest probability jumps

- tick `118137`, seconds `38.50`, LSTM `0.7241`, delta `+0.2371`
- tick `118809`, seconds `49.00`, LSTM `0.5020`, delta `+0.2253`
- tick `118713`, seconds `47.50`, LSTM `0.3110`, delta `-0.2024`
- tick `120633`, seconds `77.50`, LSTM `0.1405`, delta `-0.1963`
- tick `118393`, seconds `42.50`, LSTM `0.3653`, delta `-0.1360`
- tick `118425`, seconds `43.00`, LSTM `0.4948`, delta `+0.1295`
- tick `118201`, seconds `39.50`, LSTM `0.5420`, delta `-0.1293`
- tick `118745`, seconds `48.00`, LSTM `0.2198`, delta `-0.0913`
- tick `118777`, seconds `48.50`, LSTM `0.2767`, delta `+0.0569`
- tick `118169`, seconds `39.00`, LSTM `0.6713`, delta `-0.0528`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003092`, |coef| `0.003092`
- `lag_00__damage_diff_last_5s`: coefficient `0.002841`, |coef| `0.002841`
- `lag_00__T_kills_last_3s`: coefficient `-0.002643`, |coef| `0.002643`
- `lag_00__CT3__flash_duration`: coefficient `0.002326`, |coef| `0.002326`
- `lag_06__CT_place_HOLE`: coefficient `-0.002126`, |coef| `0.002126`
- `lag_07__CT_place_HOLE`: coefficient `-0.002005`, |coef| `0.002005`
- `lag_00__T_place_EXTENDEDA`: coefficient `-0.001903`, |coef| `0.001903`
- `lag_12__T_place_CATWALK`: coefficient `-0.001747`, |coef| `0.001747`
- `lag_03__T_place_SHORTSTAIRS`: coefficient `-0.001708`, |coef| `0.001708`
- `lag_00__T_damage_last_5s`: coefficient `-0.001686`, |coef| `0.001686`
- `lag_15__CT_place_LONGDOORS`: coefficient `0.001609`, |coef| `0.001609`
- `lag_14__T5__duck_amount`: coefficient `-0.001597`, |coef| `0.001597`
- `lag_15__T5__duck_amount`: coefficient `-0.001581`, |coef| `0.001581`
- `lag_13__T_duck_amount_mean`: coefficient `-0.001568`, |coef| `0.001568`
- `lag_15__CT_place_LONGA`: coefficient `-0.001562`, |coef| `0.001562`

## Top 10 utility ridge features

- `lag_00__CT3__flash_duration`: coefficient `0.002326` (raises CT win probability)
- `lag_02__T3__flash_duration`: coefficient `0.001486` (raises CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `0.001484` (raises CT win probability)
- `lag_07__T5__flash_duration`: coefficient `0.001376` (raises CT win probability)
- `lag_10__T5__flash_duration`: coefficient `-0.001255` (lowers CT win probability)
- `lag_00__T4__flash_duration`: coefficient `-0.001177` (lowers CT win probability)
- `lag_02__T4__flash_duration`: coefficient `0.001169` (raises CT win probability)
- `lag_02__T_flash_duration_sum`: coefficient `0.001148` (raises CT win probability)
- `lag_12__T1__flash_duration`: coefficient `-0.001031` (lowers CT win probability)
- `lag_00__CT5__flash_duration`: coefficient `0.000978` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003092` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002841` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002643` (lowers CT win probability)
- `lag_06__CT_place_HOLE`: coefficient `-0.002126` (lowers CT win probability)
- `lag_07__CT_place_HOLE`: coefficient `-0.002005` (lowers CT win probability)
- `lag_00__T_place_EXTENDEDA`: coefficient `-0.001903` (lowers CT win probability)
- `lag_12__T_place_CATWALK`: coefficient `-0.001747` (lowers CT win probability)
- `lag_03__T_place_SHORTSTAIRS`: coefficient `-0.001708` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001686` (lowers CT win probability)
- `lag_15__CT_place_LONGDOORS`: coefficient `0.001609` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `118137`, seconds `38.50`, LSTM delta `+0.2371`

Top all feature movements:
- `lag_00__CT3__flash_duration`: contribution `+0.018664`
- `lag_02__T3__flash_duration`: contribution `+0.009777`
- `lag_02__T_flashed_players`: contribution `+0.008039`
- `lag_01__CT_place_SHORTSTAIRS`: contribution `+0.007799`
- `lag_04__T_place_CATWALK`: contribution `+0.007616`

Top utility-only movements:
- `lag_00__CT3__flash_duration`: contribution `+0.018664`
- `lag_02__T3__flash_duration`: contribution `+0.009777`
- `lag_00__CT_flash_duration_sum`: contribution `+0.007296`
- `lag_02__T_flash_duration_sum`: contribution `+0.005904`
- `lag_00__T4__flash_duration`: contribution `+0.005659`

### tick `118809`, seconds `49.00`, LSTM delta `+0.2253`

Top all feature movements:
- `lag_07__CT_place_HOLE`: contribution `+0.022386`
- `lag_09__CT_place_HOLE`: contribution `+0.014632`
- `lag_00__kill_diff_last_3s`: contribution `+0.007442`
- `lag_03__T_place_SHORTSTAIRS`: contribution `+0.007176`
- `lag_13__CT_place_SHORTSTAIRS`: contribution `+0.007021`

Top utility-only movements:
- `lag_10__T5__flash_duration`: contribution `+0.006842`
- `lag_13__CT3__flash_duration`: contribution `+0.006208`
- `lag_12__T1__flash_duration`: contribution `+0.004916`

### tick `118713`, seconds `47.50`, LSTM delta `-0.2024`

Top all feature movements:
- `lag_06__CT_place_HOLE`: contribution `-0.023736`
- `lag_04__CT_place_HOLE`: contribution `-0.009481`
- `lag_00__T_place_EXTENDEDA`: contribution `-0.009437`
- `lag_00__T_kills_last_3s`: contribution `-0.008374`
- `lag_07__T5__flash_duration`: contribution `-0.007500`

Top utility-only movements:
- `lag_07__T5__flash_duration`: contribution `-0.007500`
- `lag_10__CT3__flash_duration`: contribution `-0.005468`
- `lag_09__T1__flash_duration`: contribution `-0.003758`
- `lag_08__T3__flash_duration`: contribution `-0.002400`

### tick `120633`, seconds `77.50`, LSTM delta `-0.1963`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.008374`
- `lag_00__kill_diff_last_3s`: contribution `-0.007442`
- `lag_08__T_duck_amount_mean`: contribution `-0.007378`
- `lag_15__CT_place_LONGDOORS`: contribution `-0.007046`
- `lag_06__CT2__duck_amount`: contribution `-0.004807`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `118393`, seconds `42.50`, LSTM delta `-0.1360`

Top all feature movements:
- `lag_00__CT3__flash_duration`: contribution `-0.015127`
- `lag_12__T_place_CATWALK`: contribution `-0.010055`
- `lag_13__CT_place_SHORTSTAIRS`: contribution `-0.007021`
- `lag_00__damage_diff_last_5s`: contribution `-0.005640`
- `lag_10__T3__flash_duration`: contribution `-0.005426`

Top utility-only movements:
- `lag_00__CT3__flash_duration`: contribution `-0.015127`
- `lag_10__T3__flash_duration`: contribution `-0.005426`
- `lag_10__T_flash_duration_sum`: contribution `-0.004434`
- `lag_00__CT_flash_duration_sum`: contribution `-0.004317`
- `lag_08__CT3__flash_duration`: contribution `-0.003077`
