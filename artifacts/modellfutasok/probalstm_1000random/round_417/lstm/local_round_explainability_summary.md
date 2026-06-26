# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-3dmax-vs-faze-bo3-oPmZd_fSjJ93cG46OkNLxM/3dmax-vs-faze-m2-ancient.csv`
- round_num: `3`

## Largest probability jumps

- tick `14646`, seconds `0.50`, LSTM `0.0175`, delta `-0.0320`
- tick `16694`, seconds `32.50`, LSTM `0.0038`, delta `-0.0135`
- tick `16662`, seconds `32.00`, LSTM `0.0173`, delta `+0.0086`
- tick `14678`, seconds `1.00`, LSTM `0.0111`, delta `-0.0065`
- tick `15350`, seconds `11.50`, LSTM `0.0263`, delta `+0.0058`
- tick `15318`, seconds `11.00`, LSTM `0.0204`, delta `+0.0049`
- tick `16214`, seconds `25.00`, LSTM `0.0065`, delta `-0.0046`
- tick `16534`, seconds `30.00`, LSTM `0.0112`, delta `+0.0045`
- tick `15734`, seconds `17.50`, LSTM `0.0278`, delta `+0.0044`
- tick `15638`, seconds `16.00`, LSTM `0.0253`, delta `-0.0044`

## Top 15 local ridge features

- `lag_01__CT_place_UNKNOWN`: coefficient `-0.000675`, |coef| `0.000675`
- `lag_00__CT_place_UNKNOWN`: coefficient `0.000446`, |coef| `0.000446`
- `lag_10__T_shots_fired_sum`: coefficient `0.000127`, |coef| `0.000127`
- `lag_04__CT_place_UNKNOWN`: coefficient `-0.000127`, |coef| `0.000127`
- `lag_00__T_velocity_mean`: coefficient `-0.000119`, |coef| `0.000119`
- `lag_08__T_place_TSIDELOWER`: coefficient `0.000115`, |coef| `0.000115`
- `lag_00__T_place_SIDEENTRANCE`: coefficient `-0.000114`, |coef| `0.000114`
- `lag_07__T_place_SIDEENTRANCE`: coefficient `-0.000108`, |coef| `0.000108`
- `lag_00__CT_velocity_mean`: coefficient `-0.000106`, |coef| `0.000106`
- `lag_03__T2__duck_amount`: coefficient `0.000105`, |coef| `0.000105`
- `lag_00__damage_diff_last_5s`: coefficient `0.000099`, |coef| `0.000099`
- `lag_02__T_place_SIDEENTRANCE`: coefficient `-0.000099`, |coef| `0.000099`
- `lag_03__CT_place_UNKNOWN`: coefficient `-0.000099`, |coef| `0.000099`
- `lag_01__T_place_SIDEENTRANCE`: coefficient `-0.000097`, |coef| `0.000097`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000095`, |coef| `0.000095`

## Top 10 utility ridge features

- `lag_09__CT5__flash_duration`: coefficient `0.000090` (raises CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000082` (raises CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000080` (raises CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000067` (raises CT win probability)
- `lag_02__CT_utility_damage_last_5s`: coefficient `0.000063` (raises CT win probability)
- `lag_02__utility_damage_diff_last_5s`: coefficient `0.000059` (raises CT win probability)
- `lag_00__T_active_infernos`: coefficient `0.000056` (raises CT win probability)
- `lag_08__CT5__flash_duration`: coefficient `0.000053` (raises CT win probability)
- `lag_01__T_smoke_inv`: coefficient `-0.000052` (lowers CT win probability)
- `lag_01__flash_inv_diff`: coefficient `0.000052` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_UNKNOWN`: coefficient `-0.000675` (lowers CT win probability)
- `lag_00__CT_place_UNKNOWN`: coefficient `0.000446` (raises CT win probability)
- `lag_10__T_shots_fired_sum`: coefficient `0.000127` (raises CT win probability)
- `lag_04__CT_place_UNKNOWN`: coefficient `-0.000127` (lowers CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000119` (lowers CT win probability)
- `lag_08__T_place_TSIDELOWER`: coefficient `0.000115` (raises CT win probability)
- `lag_00__T_place_SIDEENTRANCE`: coefficient `-0.000114` (lowers CT win probability)
- `lag_07__T_place_SIDEENTRANCE`: coefficient `-0.000108` (lowers CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000106` (lowers CT win probability)
- `lag_03__T2__duck_amount`: coefficient `0.000105` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `14646`, seconds `0.50`, LSTM delta `-0.0320`

Top all feature movements:
- `lag_01__CT_place_UNKNOWN`: contribution `-0.023700`
- `lag_01__T_place_TSPAWN`: contribution `-0.000420`
- `lag_00__T_velocity_mean`: contribution `-0.000389`
- `lag_01__T_closest_enemy_dist`: contribution `-0.000347`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.000323`

Top utility-only movements:
- `lag_01__smoke_inv_diff`: contribution `-0.000261`
- `lag_01__utility_inv_diff`: contribution `-0.000246`
- `lag_01__molly_inv_diff`: contribution `-0.000186`
- `lag_01__T_smoke_inv`: contribution `-0.000119`
- `lag_01__T_molly_inv`: contribution `-0.000110`

### tick `16694`, seconds `32.50`, LSTM delta `-0.0135`

Top all feature movements:
- `lag_10__T_shots_fired_sum`: contribution `-0.001049`
- `lag_11__T_shots_fired_sum`: contribution `-0.000497`
- `lag_08__T_place_TSIDELOWER`: contribution `-0.000430`
- `lag_03__T2__duck_amount`: contribution `-0.000401`
- `lag_04__T_shots_fired_sum`: contribution `-0.000399`

Top utility-only movements:
- `lag_02__CT_utility_damage_last_5s`: contribution `-0.000356`
- `lag_04__T_utility_damage_last_5s`: contribution `-0.000284`
- `lag_02__utility_damage_diff_last_5s`: contribution `-0.000270`
- `lag_12__CT_utility_damage_last_5s`: contribution `-0.000189`

### tick `16662`, seconds `32.00`, LSTM delta `+0.0086`

Top all feature movements:
- `lag_10__T_shots_fired_sum`: contribution `+0.001049`
- `lag_07__T_place_SIDEENTRANCE`: contribution `+0.000528`
- `lag_09__T_shots_fired_sum`: contribution `+0.000521`
- `lag_02__T_place_SIDEENTRANCE`: contribution `+0.000484`
- `lag_04__T_shots_fired_sum`: contribution `+0.000399`

Top utility-only movements:
- `lag_03__T_utility_damage_last_5s`: contribution `+0.000247`
- `lag_11__CT_utility_damage_last_5s`: contribution `+0.000196`

### tick `14678`, seconds `1.00`, LSTM delta `-0.0065`

Top all feature movements:
- `lag_00__CT_place_UNKNOWN`: contribution `-0.006268`
- `lag_02__CT_place_UNKNOWN`: contribution `+0.001230`
- `lag_02__T2__duck_amount`: contribution `-0.000127`
- `lag_00__T2__duck_amount`: contribution `-0.000081`
- `lag_00__CT_velocity_mean`: contribution `-0.000077`

Top utility-only movements:
- `lag_00__T2__smoke`: contribution `-0.000054`
- `lag_02__utility_inv_diff`: contribution `-0.000048`
- `lag_02__smoke_inv_diff`: contribution `-0.000041`

### tick `15350`, seconds `11.50`, LSTM delta `+0.0058`

Top all feature movements:
- `lag_11__T_place_TUNNEL`: contribution `+0.001093`
- `lag_09__T_place_WATER`: contribution `+0.000965`
- `lag_14__CT_place_HOUSE`: contribution `+0.000452`
- `lag_11__T_place_WATER`: contribution `+0.000359`
- `lag_12__T_place_WATER`: contribution `+0.000357`

Top utility-only movements:
- No utility movement among the top local contributors.
