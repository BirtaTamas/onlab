# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-legacy-bo3-o9hWAn-ugamRsSw8ngEfHF/astralis-vs-legacy-m3-ancient.csv`
- round_num: `11`

## Largest probability jumps

- tick `82370`, seconds `13.00`, LSTM `0.2094`, delta `-0.2097`
- tick `82402`, seconds `13.50`, LSTM `0.1589`, delta `-0.0504`
- tick `82626`, seconds `17.00`, LSTM `0.1314`, delta `-0.0393`
- tick `82210`, seconds `10.50`, LSTM `0.4378`, delta `-0.0288`
- tick `81634`, seconds `1.50`, LSTM `0.4841`, delta `-0.0229`
- tick `83490`, seconds `30.50`, LSTM `0.1311`, delta `-0.0228`
- tick `89826`, seconds `129.50`, LSTM `0.0307`, delta `+0.0226`
- tick `83778`, seconds `35.00`, LSTM `0.1003`, delta `-0.0217`
- tick `82082`, seconds `8.50`, LSTM `0.4391`, delta `-0.0200`
- tick `86018`, seconds `70.00`, LSTM `0.1306`, delta `+0.0200`

## Top 15 local ridge features

- `lag_00__CT_place_UNKNOWN`: coefficient `0.003568`, |coef| `0.003568`
- `lag_14__T_he_last_5s`: coefficient `0.002021`, |coef| `0.002021`
- `lag_12__T_place_WATER`: coefficient `0.001515`, |coef| `0.001515`
- `lag_00__CT_flashed_players`: coefficient `-0.001239`, |coef| `0.001239`
- `lag_02__CT_place_UNKNOWN`: coefficient `0.001170`, |coef| `0.001170`
- `lag_10__T3__flash_duration`: coefficient `-0.001083`, |coef| `0.001083`
- `lag_10__CT_flashed_players`: coefficient `-0.000954`, |coef| `0.000954`
- `lag_04__CT5__flash_duration`: coefficient `-0.000925`, |coef| `0.000925`
- `lag_15__T_place_WATER`: coefficient `-0.000836`, |coef| `0.000836`
- `lag_06__T2__duck_amount`: coefficient `-0.000806`, |coef| `0.000806`
- `lag_00__T_kills_last_3s`: coefficient `-0.000773`, |coef| `0.000773`
- `lag_00__kill_diff_last_3s`: coefficient `0.000743`, |coef| `0.000743`
- `lag_09__CT_place_TOPOFMID`: coefficient `0.000720`, |coef| `0.000720`
- `lag_04__T_place_TSIDELOWER`: coefficient `-0.000717`, |coef| `0.000717`
- `lag_14__T_place_TUNNEL`: coefficient `0.000715`, |coef| `0.000715`

## Top 10 utility ridge features

- `lag_14__T_he_last_5s`: coefficient `0.002021` (raises CT win probability)
- `lag_10__T3__flash_duration`: coefficient `-0.001083` (lowers CT win probability)
- `lag_04__CT5__flash_duration`: coefficient `-0.000925` (lowers CT win probability)
- `lag_01__CT1__flash_duration`: coefficient `-0.000707` (lowers CT win probability)
- `lag_00__CT2__utility_total`: coefficient `0.000670` (raises CT win probability)
- `lag_00__CT2__smoke`: coefficient `0.000615` (raises CT win probability)
- `lag_09__CT1__flash_duration`: coefficient `-0.000593` (lowers CT win probability)
- `lag_05__CT_B_site_active_infernos`: coefficient `-0.000564` (lowers CT win probability)
- `lag_01__CT_flash_duration_sum`: coefficient `-0.000557` (lowers CT win probability)
- `lag_10__CT4__flash_duration`: coefficient `-0.000557` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_UNKNOWN`: coefficient `0.003568` (raises CT win probability)
- `lag_12__T_place_WATER`: coefficient `0.001515` (raises CT win probability)
- `lag_00__CT_flashed_players`: coefficient `-0.001239` (lowers CT win probability)
- `lag_02__CT_place_UNKNOWN`: coefficient `0.001170` (raises CT win probability)
- `lag_10__CT_flashed_players`: coefficient `-0.000954` (lowers CT win probability)
- `lag_15__T_place_WATER`: coefficient `-0.000836` (lowers CT win probability)
- `lag_06__T2__duck_amount`: coefficient `-0.000806` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.000773` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000743` (raises CT win probability)
- `lag_09__CT_place_TOPOFMID`: coefficient `0.000720` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `82370`, seconds `13.00`, LSTM delta `-0.2097`

Top all feature movements:
- `lag_14__T_he_last_5s`: contribution `-0.026375`
- `lag_12__T_place_WATER`: contribution `-0.017297`
- `lag_00__CT_flashed_players`: contribution `-0.008140`
- `lag_10__T3__flash_duration`: contribution `-0.007785`
- `lag_10__CT_flashed_players`: contribution `-0.006266`

Top utility-only movements:
- `lag_14__T_he_last_5s`: contribution `-0.026375`
- `lag_10__T3__flash_duration`: contribution `-0.007785`
- `lag_04__CT5__flash_duration`: contribution `-0.006089`

### tick `82402`, seconds `13.50`, LSTM delta `-0.0504`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.004970`
- `lag_01__CT_flashed_players`: contribution `-0.004513`
- `lag_15__T_he_last_5s`: contribution `-0.003133`
- `lag_00__CT_flashed_players`: contribution `+0.002713`
- `lag_09__CT_place_TOPOFMID`: contribution `-0.002614`

Top utility-only movements:
- `lag_15__T_he_last_5s`: contribution `-0.003133`
- `lag_05__CT_B_site_active_infernos`: contribution `-0.001939`
- `lag_05__CT5__flash_duration`: contribution `-0.001770`

### tick `82626`, seconds `17.00`, LSTM delta `-0.0393`

Top all feature movements:
- `lag_03__T3__flash_duration`: contribution `-0.003285`
- `lag_12__CT5__flash_duration`: contribution `-0.002717`
- `lag_00__CT_flashed_players`: contribution `+0.002713`
- `lag_07__T_shots_fired_sum`: contribution `-0.002361`
- `lag_10__CT_flashed_players`: contribution `+0.002089`

Top utility-only movements:
- `lag_03__T3__flash_duration`: contribution `-0.003285`
- `lag_12__CT5__flash_duration`: contribution `-0.002717`
- `lag_00__CT_flash_duration_sum`: contribution `+0.001397`
- `lag_01__CT_active_infernos`: contribution `-0.001184`
- `lag_12__CT_flash_duration_sum`: contribution `-0.001009`

### tick `82210`, seconds `10.50`, LSTM delta `-0.0288`

Top all feature movements:
- `lag_09__T_he_last_5s`: contribution `-0.006553`
- `lag_07__T_place_WATER`: contribution `-0.006545`
- `lag_09__T_place_TUNNEL`: contribution `-0.003257`
- `lag_01__CT_flashed_players`: contribution `+0.003009`
- `lag_15__T_place_TUNNEL`: contribution `+0.002784`

Top utility-only movements:
- `lag_09__T_he_last_5s`: contribution `-0.006553`
- `lag_05__T3__flash_duration`: contribution `-0.001866`
- `lag_01__CT1__flash_duration`: contribution `+0.001043`

### tick `81634`, seconds `1.50`, LSTM delta `-0.0229`

Top all feature movements:
- `lag_00__CT_place_UNKNOWN`: contribution `-0.050122`
- `lag_03__CT_place_UNKNOWN`: contribution `+0.011656`
- `lag_01__T_he_last_5s`: contribution `+0.003450`
- `lag_01__CT_place_UNKNOWN`: contribution `+0.003298`
- `lag_02__CT_velocity_mean`: contribution `+0.000744`

Top utility-only movements:
- `lag_01__T_he_last_5s`: contribution `+0.003450`
- `lag_03__CT5__utility_total`: contribution `+0.000577`
- `lag_03__T1__utility_total`: contribution `+0.000540`
- `lag_03__T4__utility_total`: contribution `+0.000526`
- `lag_03__CT1__smoke`: contribution `+0.000468`
