# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-liquid-vs-3dmax-bo3-k7r_vGkiL4eRhxKdRPUZx1/liquid-vs-3dmax-m2-ancient.csv`
- round_num: `14`

## Largest probability jumps

- tick `104989`, seconds `0.50`, LSTM `0.0086`, delta `-0.0223`
- tick `105341`, seconds `6.00`, LSTM `0.0115`, delta `-0.0042`
- tick `109565`, seconds `72.00`, LSTM `0.0037`, delta `-0.0037`
- tick `109373`, seconds `69.00`, LSTM `0.0064`, delta `-0.0034`
- tick `105085`, seconds `2.00`, LSTM `0.0161`, delta `+0.0032`
- tick `106429`, seconds `23.00`, LSTM `0.0145`, delta `+0.0031`
- tick `106685`, seconds `27.00`, LSTM `0.0138`, delta `-0.0028`
- tick `106365`, seconds `22.00`, LSTM `0.0102`, delta `+0.0025`
- tick `106461`, seconds `23.50`, LSTM `0.0169`, delta `+0.0024`
- tick `105053`, seconds `1.50`, LSTM `0.0129`, delta `+0.0023`

## Top 15 local ridge features

- `lag_01__CT_place_SIDEHALL`: coefficient `-0.000313`, |coef| `0.000313`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000157`, |coef| `0.000157`
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000132`, |coef| `0.000132`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000132`, |coef| `0.000132`
- `lag_01__T3__has_bomb`: coefficient `-0.000128`, |coef| `0.000128`
- `lag_01__CT5__duck_amount`: coefficient `-0.000127`, |coef| `0.000127`
- `lag_01__centroid_distance_xy`: coefficient `-0.000122`, |coef| `0.000122`
- `lag_00__CT5__duck_amount`: coefficient `0.000121`, |coef| `0.000121`
- `lag_01__smoke_inv_diff`: coefficient `0.000119`, |coef| `0.000119`
- `lag_01__bomb_events_last_5s`: coefficient `0.000115`, |coef| `0.000115`
- `lag_01__armor_diff`: coefficient `0.000112`, |coef| `0.000112`
- `lag_01__CT_place_BOMBSITEA`: coefficient `-0.000110`, |coef| `0.000110`
- `lag_01__CT_macro_A`: coefficient `-0.000110`, |coef| `0.000110`
- `lag_00__T_velocity_mean`: coefficient `-0.000102`, |coef| `0.000102`
- `lag_00__CT_velocity_mean`: coefficient `-0.000101`, |coef| `0.000101`

## Top 10 utility ridge features

- `lag_01__smoke_inv_diff`: coefficient `0.000119` (raises CT win probability)
- `lag_01__T_smoke_inv`: coefficient `-0.000083` (lowers CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000082` (raises CT win probability)
- `lag_01__T1__flash`: coefficient `-0.000075` (lowers CT win probability)
- `lag_10__CT_smokes_last_5s`: coefficient `-0.000065` (lowers CT win probability)
- `lag_01__T3__molly`: coefficient `-0.000063` (lowers CT win probability)
- `lag_01__T3__utility_total`: coefficient `-0.000062` (lowers CT win probability)
- `lag_01__T3__smoke`: coefficient `-0.000060` (lowers CT win probability)
- `lag_01__T5__smoke`: coefficient `-0.000059` (lowers CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000059` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_SIDEHALL`: coefficient `-0.000313` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000157` (lowers CT win probability)
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000132` (lowers CT win probability)
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000132` (lowers CT win probability)
- `lag_01__T3__has_bomb`: coefficient `-0.000128` (lowers CT win probability)
- `lag_01__CT5__duck_amount`: coefficient `-0.000127` (lowers CT win probability)
- `lag_01__centroid_distance_xy`: coefficient `-0.000122` (lowers CT win probability)
- `lag_00__CT5__duck_amount`: coefficient `0.000121` (raises CT win probability)
- `lag_01__bomb_events_last_5s`: coefficient `0.000115` (raises CT win probability)
- `lag_01__armor_diff`: coefficient `0.000112` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `104989`, seconds `0.50`, LSTM delta `-0.0223`

Top all feature movements:
- `lag_01__CT_place_SIDEHALL`: contribution `-0.002623`
- `lag_01__T_place_TSPAWN`: contribution `-0.000695`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.000488`
- `lag_01__T_closest_enemy_dist`: contribution `-0.000481`
- `lag_00__CT5__duck_amount`: contribution `-0.000458`

Top utility-only movements:
- `lag_01__smoke_inv_diff`: contribution `-0.000378`
- `lag_01__T_smoke_inv`: contribution `-0.000190`
- `lag_01__utility_inv_diff`: contribution `-0.000180`
- `lag_01__T1__flash`: contribution `-0.000154`

### tick `105341`, seconds `6.00`, LSTM delta `-0.0042`

Top all feature movements:
- `lag_10__CT_smokes_last_5s`: contribution `-0.001123`
- `lag_12__CT_place_SIDEHALL`: contribution `-0.000244`
- `lag_05__CT_place_HOUSE`: contribution `-0.000150`
- `lag_03__CT_place_ALLEY`: contribution `-0.000150`
- `lag_01__CT_macro_A`: contribution `+0.000144`

Top utility-only movements:
- `lag_10__CT_smokes_last_5s`: contribution `-0.001123`
- `lag_12__smoke_inv_diff`: contribution `-0.000065`

### tick `109565`, seconds `72.00`, LSTM delta `-0.0037`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.001634`
- `lag_00__T5__shots_fired`: contribution `-0.000452`
- `lag_00__T_place_ALLEY`: contribution `-0.000301`
- `lag_00__CT5__duck_amount`: contribution `+0.000296`
- `lag_14__T_place_HOUSE`: contribution `+0.000247`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `109373`, seconds `69.00`, LSTM delta `-0.0034`

Top all feature movements:
- `lag_01__T_place_ALLEY`: contribution `-0.000261`
- `lag_08__T_place_ALLEY`: contribution `-0.000251`
- `lag_04__T_place_CTSPAWN`: contribution `-0.000164`
- `lag_00__T3__is_walking`: contribution `-0.000157`
- `lag_09__T4__duck_amount`: contribution `-0.000144`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `105085`, seconds `2.00`, LSTM delta `+0.0032`

Top all feature movements:
- `lag_01__CT_place_SIDEHALL`: contribution `+0.001340`
- `lag_02__CT_smokes_last_5s`: contribution `+0.000456`
- `lag_04__CT_place_SIDEHALL`: contribution `+0.000224`
- `lag_02__CT_place_SIDEHALL`: contribution `+0.000140`
- `lag_04__T3__has_bomb`: contribution `-0.000105`

Top utility-only movements:
- `lag_02__CT_smokes_last_5s`: contribution `+0.000456`
