# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-the-mongolz-bo3-7VmOOQFfF_Xgx4vOG4cYIY/vitality-vs-the-mongolz-m3-inferno.csv`
- round_num: `7`

## Largest probability jumps

- tick `58648`, seconds `66.00`, LSTM `0.2507`, delta `+0.1249`
- tick `55416`, seconds `15.50`, LSTM `0.0799`, delta `-0.1209`
- tick `58744`, seconds `67.50`, LSTM `0.1041`, delta `-0.1063`
- tick `58616`, seconds `65.50`, LSTM `0.1258`, delta `+0.0800`
- tick `54456`, seconds `0.50`, LSTM `0.0704`, delta `-0.0725`
- tick `58680`, seconds `66.50`, LSTM `0.2050`, delta `-0.0457`
- tick `58936`, seconds `70.50`, LSTM `0.0476`, delta `-0.0444`
- tick `57816`, seconds `53.00`, LSTM `0.0182`, delta `-0.0424`
- tick `55384`, seconds `15.00`, LSTM `0.2008`, delta `+0.0371`
- tick `54488`, seconds `1.00`, LSTM `0.0976`, delta `+0.0272`

## Top 15 local ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.001226`, |coef| `0.001226`
- `lag_09__T_place_ARCH`: coefficient `0.001027`, |coef| `0.001027`
- `lag_00__kill_diff_last_3s`: coefficient `0.001012`, |coef| `0.001012`
- `lag_00__T_place_BALCONY`: coefficient `-0.000992`, |coef| `0.000992`
- `lag_02__CT_place_LIBRARY`: coefficient `-0.000970`, |coef| `0.000970`
- `lag_14__CT_place_RUINS`: coefficient `0.000953`, |coef| `0.000953`
- `lag_00__damage_diff_last_5s`: coefficient `0.000878`, |coef| `0.000878`
- `lag_12__CT3__duck_amount`: coefficient `0.000813`, |coef| `0.000813`
- `lag_03__T_place_ARCH`: coefficient `-0.000784`, |coef| `0.000784`
- `lag_00__T_kills_last_3s`: coefficient `-0.000762`, |coef| `0.000762`
- `lag_10__T_active_infernos`: coefficient `-0.000761`, |coef| `0.000761`
- `lag_02__CT_shots_fired_sum`: coefficient `0.000738`, |coef| `0.000738`
- `lag_03__T2__flash_duration`: coefficient `0.000715`, |coef| `0.000715`
- `lag_00__CT2__shots_fired`: coefficient `0.000705`, |coef| `0.000705`
- `lag_10__T_place_ARCH`: coefficient `0.000684`, |coef| `0.000684`

## Top 10 utility ridge features

- `lag_10__T_active_infernos`: coefficient `-0.000761` (lowers CT win probability)
- `lag_03__T2__flash_duration`: coefficient `0.000715` (raises CT win probability)
- `lag_10__T_B_site_active_infernos`: coefficient `-0.000565` (lowers CT win probability)
- `lag_10__active_infernos_total`: coefficient `-0.000522` (lowers CT win probability)
- `lag_07__T5__molly`: coefficient `-0.000510` (lowers CT win probability)
- `lag_02__T5__flash_duration`: coefficient `0.000495` (raises CT win probability)
- `lag_10__T_A_site_active_infernos`: coefficient `-0.000493` (lowers CT win probability)
- `lag_00__CT3__smoke`: coefficient `0.000487` (raises CT win probability)
- `lag_00__CT_smokes_last_5s`: coefficient `0.000484` (raises CT win probability)
- `lag_15__T3__molly`: coefficient `0.000477` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.001226` (raises CT win probability)
- `lag_09__T_place_ARCH`: coefficient `0.001027` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001012` (raises CT win probability)
- `lag_00__T_place_BALCONY`: coefficient `-0.000992` (lowers CT win probability)
- `lag_02__CT_place_LIBRARY`: coefficient `-0.000970` (lowers CT win probability)
- `lag_14__CT_place_RUINS`: coefficient `0.000953` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000878` (raises CT win probability)
- `lag_12__CT3__duck_amount`: coefficient `0.000813` (raises CT win probability)
- `lag_03__T_place_ARCH`: coefficient `-0.000784` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.000762` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `58648`, seconds `66.00`, LSTM delta `+0.1249`

Top all feature movements:
- `lag_09__T_place_ARCH`: contribution `+0.009555`
- `lag_03__T_place_ARCH`: contribution `+0.007298`
- `lag_10__T_place_ARCH`: contribution `+0.006361`
- `lag_00__CT_shots_fired_sum`: contribution `+0.005964`
- `lag_03__T2__flash_duration`: contribution `+0.004133`

Top utility-only movements:
- `lag_03__T2__flash_duration`: contribution `+0.004133`
- `lag_03__T5__flash_duration`: contribution `+0.002682`
- `lag_03__T_flash_duration_sum`: contribution `+0.002550`
- `lag_00__T2__flash_duration`: contribution `+0.002344`
- `lag_01__T5__flash_duration`: contribution `+0.001741`

### tick `55416`, seconds `15.50`, LSTM delta `-0.1209`

Top all feature movements:
- `lag_02__CT_place_LIBRARY`: contribution `-0.006221`
- `lag_14__CT_place_RUINS`: contribution `-0.003331`
- `lag_10__T_active_infernos`: contribution `-0.003171`
- `lag_12__CT3__duck_amount`: contribution `-0.003027`
- `lag_00__kill_diff_last_3s`: contribution `-0.002437`

Top utility-only movements:
- `lag_10__T_active_infernos`: contribution `-0.003171`
- `lag_10__T_B_site_active_infernos`: contribution `-0.001598`
- `lag_10__active_infernos_total`: contribution `-0.001500`
- `lag_10__T_A_site_active_infernos`: contribution `-0.001466`

### tick `58744`, seconds `67.50`, LSTM delta `-0.1063`

Top all feature movements:
- `lag_02__CT_shots_fired_sum`: contribution `-0.008712`
- `lag_02__CT2__shots_fired`: contribution `-0.004588`
- `lag_03__T2__flash_duration`: contribution `-0.004133`
- `lag_06__T_flashed_players`: contribution `-0.003957`
- `lag_12__T_place_ARCH`: contribution `+0.003944`

Top utility-only movements:
- `lag_03__T2__flash_duration`: contribution `-0.004133`
- `lag_06__T2__flash_duration`: contribution `-0.002695`
- `lag_06__T_flash_duration_sum`: contribution `-0.002398`
- `lag_06__T5__flash_duration`: contribution `-0.002053`

### tick `58616`, seconds `65.50`, LSTM delta `+0.0800`

Top all feature movements:
- `lag_09__T_place_ARCH`: contribution `+0.009555`
- `lag_03__T_place_ARCH`: contribution `+0.007298`
- `lag_02__CT_place_LIBRARY`: contribution `+0.006221`
- `lag_00__CT_shots_fired_sum`: contribution `+0.005112`
- `lag_02__T_flashed_players`: contribution `+0.003568`

Top utility-only movements:
- `lag_02__T5__flash_duration`: contribution `+0.002948`
- `lag_02__T_flash_duration_sum`: contribution `+0.002156`
- `lag_00__T5__flash_duration`: contribution `+0.002041`
- `lag_02__T2__flash_duration`: contribution `+0.001832`

### tick `54456`, seconds `0.50`, LSTM delta `-0.0725`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.002928`
- `lag_01__T_closest_enemy_dist`: contribution `-0.002296`
- `lag_01__T_place_TSPAWN`: contribution `-0.002267`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.002247`
- `lag_01__centroid_distance_xy`: contribution `-0.001922`

Top utility-only movements:
- `lag_01__molly_inv_diff`: contribution `-0.001106`
- `lag_01__smoke_inv_diff`: contribution `-0.001052`
- `lag_01__utility_inv_diff`: contribution `-0.000856`
- `lag_01__T_smoke_inv`: contribution `-0.000845`
- `lag_01__CT4__flash`: contribution `-0.000810`
