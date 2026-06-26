# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-lynn-vision-bo3-80tf5tBYONxHYQuFp0AoSQ/legacy-vs-lynn-vision-m1-dust2.csv`
- round_num: `14`

## Largest probability jumps

- tick `126396`, seconds `101.00`, LSTM `0.7576`, delta `+0.2691`
- tick `125564`, seconds `88.00`, LSTM `0.2863`, delta `-0.2507`
- tick `126140`, seconds `97.00`, LSTM `0.4292`, delta `+0.2460`
- tick `123772`, seconds `60.00`, LSTM `0.4792`, delta `-0.2347`
- tick `125820`, seconds `92.00`, LSTM `0.5859`, delta `+0.1974`
- tick `126012`, seconds `95.00`, LSTM `0.1786`, delta `-0.1893`
- tick `125436`, seconds `86.00`, LSTM `0.6812`, delta `+0.1882`
- tick `123708`, seconds `59.00`, LSTM `0.7192`, delta `+0.1783`
- tick `125500`, seconds `87.00`, LSTM `0.5597`, delta `-0.1404`
- tick `123580`, seconds `57.00`, LSTM `0.4847`, delta `+0.1258`

## Top 15 local ridge features

- `lag_00__CT_defusing_count`: coefficient `0.005191`, |coef| `0.005191`
- `lag_00__kill_diff_last_3s`: coefficient `0.003703`, |coef| `0.003703`
- `lag_08__T_flash_alpha_mean`: coefficient `-0.003244`, |coef| `0.003244`
- `lag_00__CT_kills_last_3s`: coefficient `0.002754`, |coef| `0.002754`
- `lag_00__CT_velocity_mean`: coefficient `-0.002678`, |coef| `0.002678`
- `lag_00__CT_place_HOLE`: coefficient `0.002451`, |coef| `0.002451`
- `lag_02__CT_kills_last_3s`: coefficient `-0.002300`, |coef| `0.002300`
- `lag_00__damage_diff_last_5s`: coefficient `0.002291`, |coef| `0.002291`
- `lag_05__T_place_BDOORS`: coefficient `0.002284`, |coef| `0.002284`
- `lag_02__CT_shots_fired_sum`: coefficient `0.002132`, |coef| `0.002132`
- `lag_00__T1__duck_amount`: coefficient `-0.001890`, |coef| `0.001890`
- `lag_00__T_kills_last_3s`: coefficient `-0.001853`, |coef| `0.001853`
- `lag_08__centroid_distance_xy`: coefficient `0.001787`, |coef| `0.001787`
- `lag_00__T_damage_last_5s`: coefficient `-0.001758`, |coef| `0.001758`
- `lag_01__damage_diff_last_5s`: coefficient `0.001743`, |coef| `0.001743`

## Top 10 utility ridge features

- `lag_08__T_flash_alpha_mean`: coefficient `-0.003244` (lowers CT win probability)
- `lag_15__T1__flash_duration`: coefficient `-0.001725` (lowers CT win probability)
- `lag_13__T1__flash_duration`: coefficient `-0.001558` (lowers CT win probability)
- `lag_12__T1__flash_duration`: coefficient `-0.001538` (lowers CT win probability)
- `lag_08__T1__flash_duration`: coefficient `-0.001288` (lowers CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.001214` (lowers CT win probability)
- `lag_14__T1__flash_duration`: coefficient `-0.001203` (lowers CT win probability)
- `lag_09__CT3__flash_duration`: coefficient `-0.001181` (lowers CT win probability)
- `lag_07__T_flash_alpha_mean`: coefficient `-0.001054` (lowers CT win probability)
- `lag_00__T_B_site_active_infernos`: coefficient `-0.001020` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_defusing_count`: coefficient `0.005191` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003703` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002754` (raises CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.002678` (lowers CT win probability)
- `lag_00__CT_place_HOLE`: coefficient `0.002451` (raises CT win probability)
- `lag_02__CT_kills_last_3s`: coefficient `-0.002300` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002291` (raises CT win probability)
- `lag_05__T_place_BDOORS`: coefficient `0.002284` (raises CT win probability)
- `lag_02__CT_shots_fired_sum`: coefficient `0.002132` (raises CT win probability)
- `lag_00__T1__duck_amount`: coefficient `-0.001890` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `126396`, seconds `101.00`, LSTM delta `+0.2691`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `+0.050322`
- `lag_08__T_flash_alpha_mean`: contribution `+0.019683`
- `lag_00__CT_velocity_mean`: contribution `+0.009862`
- `lag_02__CT_kills_last_3s`: contribution `+0.006641`
- `lag_08__centroid_distance_xy`: contribution `+0.005850`

Top utility-only movements:
- `lag_08__T_flash_alpha_mean`: contribution `+0.019683`

### tick `125564`, seconds `88.00`, LSTM delta `-0.2507`

Top all feature movements:
- `lag_04__T_place_HOLE`: contribution `-0.039448`
- `lag_15__T_place_HOLE`: contribution `-0.036315`
- `lag_00__CT_place_HOLE`: contribution `-0.027368`
- `lag_05__CT_place_HOLE`: contribution `-0.015068`
- `lag_04__CT_place_HOLE`: contribution `+0.011007`

Top utility-only movements:
- `lag_04__CT3__flash_duration`: contribution `-0.005091`
- `lag_04__CT_flash_duration_sum`: contribution `-0.003426`

### tick `126140`, seconds `97.00`, LSTM delta `+0.2460`

Top all feature movements:
- `lag_08__CT_place_HOLE`: contribution `+0.016078`
- `lag_13__CT_place_HOLE`: contribution `+0.009656`
- `lag_00__kill_diff_last_3s`: contribution `+0.008914`
- `lag_08__CT_shots_fired_sum`: contribution `+0.008660`
- `lag_00__CT_velocity_mean`: contribution `+0.008481`

Top utility-only movements:
- `lag_08__T1__flash_duration`: contribution `+0.008430`
- `lag_00__T_flash_alpha_mean`: contribution `+0.007368`
- `lag_09__CT3__flash_duration`: contribution `+0.006630`

### tick `123772`, seconds `60.00`, LSTM delta `-0.2347`

Top all feature movements:
- `lag_00__CT_place_HOLE`: contribution `-0.027368`
- `lag_02__CT_shots_fired_sum`: contribution `-0.020737`
- `lag_00__kill_diff_last_3s`: contribution `-0.017828`
- `lag_07__T_place_BDOORS`: contribution `-0.013251`
- `lag_02__CT4__shots_fired`: contribution `-0.008038`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `125820`, seconds `92.00`, LSTM delta `+0.1974`

Top all feature movements:
- `lag_12__T_place_HOLE`: contribution `+0.039346`
- `lag_08__CT_place_HOLE`: contribution `+0.016078`
- `lag_14__CT_place_HOLE`: contribution `+0.015918`
- `lag_03__CT_place_HOLE`: contribution `+0.014992`
- `lag_12__CT_place_BDOORS`: contribution `+0.011285`

Top utility-only movements:
- `lag_12__CT3__flash_duration`: contribution `+0.006159`
- `lag_12__CT_flash_duration_sum`: contribution `+0.005428`
- `lag_10__T1__flash_duration`: contribution `+0.003416`
