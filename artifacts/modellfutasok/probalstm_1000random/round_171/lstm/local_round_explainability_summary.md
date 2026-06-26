# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-betboom-vs-nemiga-train-khA7BVyAiKBjWcyTrFzube/betboom-vs-nemiga-train.csv`
- round_num: `10`

## Largest probability jumps

- tick `81895`, seconds `0.50`, LSTM `0.0231`, delta `-0.0418`
- tick `82759`, seconds `14.00`, LSTM `0.0156`, delta `-0.0106`
- tick `83335`, seconds `23.00`, LSTM `0.0072`, delta `-0.0064`
- tick `81927`, seconds `1.00`, LSTM `0.0168`, delta `-0.0063`
- tick `82023`, seconds `2.50`, LSTM `0.0222`, delta `+0.0058`
- tick `82823`, seconds `15.00`, LSTM `0.0195`, delta `+0.0054`
- tick `82727`, seconds `13.50`, LSTM `0.0262`, delta `+0.0044`
- tick `82055`, seconds `3.00`, LSTM `0.0263`, delta `+0.0041`
- tick `82471`, seconds `9.50`, LSTM `0.0210`, delta `-0.0039`
- tick `82247`, seconds `6.00`, LSTM `0.0222`, delta `-0.0036`

## Top 15 local ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000367`, |coef| `0.000367`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000331`, |coef| `0.000331`
- `lag_00__CT_velocity_mean`: coefficient `-0.000303`, |coef| `0.000303`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000301`, |coef| `0.000301`
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000301`, |coef| `0.000301`
- `lag_01__centroid_distance_xy`: coefficient `-0.000290`, |coef| `0.000290`
- `lag_00__T_velocity_mean`: coefficient `-0.000242`, |coef| `0.000242`
- `lag_01__utility_inv_diff`: coefficient `0.000238`, |coef| `0.000238`
- `lag_01__molly_inv_diff`: coefficient `0.000215`, |coef| `0.000215`
- `lag_01__flash_inv_diff`: coefficient `0.000213`, |coef| `0.000213`
- `lag_01__T1__has_bomb`: coefficient `-0.000193`, |coef| `0.000193`
- `lag_01__T1__utility_total`: coefficient `-0.000190`, |coef| `0.000190`
- `lag_00__T4__smoke`: coefficient `0.000186`, |coef| `0.000186`
- `lag_01__armor_diff`: coefficient `0.000185`, |coef| `0.000185`
- `lag_01__T1__flash`: coefficient `-0.000176`, |coef| `0.000176`

## Top 10 utility ridge features

- `lag_01__utility_inv_diff`: coefficient `0.000238` (raises CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000215` (raises CT win probability)
- `lag_01__flash_inv_diff`: coefficient `0.000213` (raises CT win probability)
- `lag_01__T1__utility_total`: coefficient `-0.000190` (lowers CT win probability)
- `lag_00__T4__smoke`: coefficient `0.000186` (raises CT win probability)
- `lag_01__T1__flash`: coefficient `-0.000176` (lowers CT win probability)
- `lag_01__T5__utility_total`: coefficient `-0.000172` (lowers CT win probability)
- `lag_01__T_molly_inv`: coefficient `-0.000169` (lowers CT win probability)
- `lag_01__T_utility_inv`: coefficient `-0.000159` (lowers CT win probability)
- `lag_01__T_flash_inv`: coefficient `-0.000149` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000367` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000331` (lowers CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000303` (lowers CT win probability)
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000301` (lowers CT win probability)
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000301` (lowers CT win probability)
- `lag_01__centroid_distance_xy`: coefficient `-0.000290` (lowers CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000242` (lowers CT win probability)
- `lag_01__T1__has_bomb`: coefficient `-0.000193` (lowers CT win probability)
- `lag_01__armor_diff`: coefficient `0.000185` (raises CT win probability)
- `lag_01__equip_diff`: coefficient `0.000162` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `81895`, seconds `0.50`, LSTM delta `-0.0418`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.001757`
- `lag_01__T_place_TSPAWN`: contribution `-0.001465`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.001231`
- `lag_01__T_closest_enemy_dist`: contribution `-0.001202`
- `lag_01__centroid_distance_xy`: contribution `-0.001114`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `-0.000734`
- `lag_01__molly_inv_diff`: contribution `-0.000600`
- `lag_01__flash_inv_diff`: contribution `-0.000571`
- `lag_01__T1__utility_total`: contribution `-0.000428`
- `lag_00__T4__smoke`: contribution `-0.000405`

### tick `82759`, seconds `14.00`, LSTM delta `-0.0106`

Top all feature movements:
- `lag_02__CT_place_ELECTRICALBOX`: contribution `-0.000858`
- `lag_11__CT_flashed_players`: contribution `-0.000823`
- `lag_09__CT_place_ELECTRICALBOX`: contribution `-0.000751`
- `lag_11__CT_flash_duration_sum`: contribution `-0.000508`
- `lag_11__CT_place_CONNECTOR`: contribution `-0.000434`

Top utility-only movements:
- `lag_11__CT_flash_duration_sum`: contribution `-0.000508`
- `lag_11__CT3__flash_duration`: contribution `-0.000249`
- `lag_11__CT5__flash_duration`: contribution `-0.000147`

### tick `83335`, seconds `23.00`, LSTM delta `-0.0064`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.000397`
- `lag_15__CT3__flash_duration`: contribution `-0.000283`
- `lag_15__T_place_BACKOFB`: contribution `-0.000227`
- `lag_10__T_place_BACKOFB`: contribution `-0.000224`
- `lag_00__T5__is_scoped`: contribution `-0.000206`

Top utility-only movements:
- `lag_15__CT3__flash_duration`: contribution `-0.000283`
- `lag_12__T_A_site_active_infernos`: contribution `-0.000140`
- `lag_00__CT3__smoke`: contribution `-0.000101`

### tick `81927`, seconds `1.00`, LSTM delta `-0.0063`

Top all feature movements:
- `lag_02__CT_place_CTSPAWN`: contribution `-0.000633`
- `lag_02__T_place_TSPAWN`: contribution `-0.000501`
- `lag_02__T_closest_enemy_dist`: contribution `-0.000441`
- `lag_02__CT_closest_enemy_dist`: contribution `-0.000440`
- `lag_02__centroid_distance_xy`: contribution `-0.000411`

Top utility-only movements:
- `lag_02__utility_inv_diff`: contribution `-0.000286`
- `lag_02__flash_inv_diff`: contribution `-0.000224`
- `lag_02__molly_inv_diff`: contribution `-0.000219`
- `lag_02__T1__utility_total`: contribution `-0.000183`
- `lag_02__T1__flash`: contribution `-0.000158`

### tick `82023`, seconds `2.50`, LSTM delta `+0.0058`

Top all feature movements:
- `lag_00__CT_place_ENTRANCE`: contribution `+0.001747`
- `lag_05__T_place_TSPAWN`: contribution `+0.000207`
- `lag_05__CT_place_CTSPAWN`: contribution `+0.000206`
- `lag_05__CT_closest_enemy_dist`: contribution `+0.000151`
- `lag_02__CT_place_CTSPAWN`: contribution `+0.000138`

Top utility-only movements:
- `lag_05__T5__flash`: contribution `+0.000056`
- `lag_05__molly_inv_diff`: contribution `+0.000043`
- `lag_05__T_molly_inv`: contribution `+0.000040`
