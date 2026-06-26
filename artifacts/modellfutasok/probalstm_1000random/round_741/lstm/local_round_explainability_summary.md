# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-3dmax-bo3-u02WLpVJ6Q22MzSL2B_-Tu/the-mongolz-vs-3dmax-m2-ancient.csv`
- round_num: `3`

## Largest probability jumps

- tick `15243`, seconds `0.50`, LSTM `0.0231`, delta `-0.0389`
- tick `16011`, seconds `12.50`, LSTM `0.0104`, delta `-0.0134`
- tick `15275`, seconds `1.00`, LSTM `0.0153`, delta `-0.0078`
- tick `15307`, seconds `1.50`, LSTM `0.0112`, delta `-0.0041`
- tick `15915`, seconds `11.00`, LSTM `0.0207`, delta `+0.0039`
- tick `15883`, seconds `10.50`, LSTM `0.0168`, delta `-0.0037`
- tick `15947`, seconds `11.50`, LSTM `0.0239`, delta `+0.0032`
- tick `15755`, seconds `8.50`, LSTM `0.0203`, delta `+0.0030`
- tick `15691`, seconds `7.50`, LSTM `0.0174`, delta `+0.0016`
- tick `16235`, seconds `16.00`, LSTM `0.0087`, delta `-0.0015`

## Top 15 local ridge features

- `lag_01__CT_place_UNKNOWN`: coefficient `-0.000655`, |coef| `0.000655`
- `lag_00__CT_place_UNKNOWN`: coefficient `0.000407`, |coef| `0.000407`
- `lag_04__CT_place_UNKNOWN`: coefficient `-0.000346`, |coef| `0.000346`
- `lag_07__CT_place_UNKNOWN`: coefficient `-0.000152`, |coef| `0.000152`
- `lag_00__CT_velocity_mean`: coefficient `-0.000133`, |coef| `0.000133`
- `lag_13__T_place_WATER`: coefficient `0.000124`, |coef| `0.000124`
- `lag_00__T_velocity_mean`: coefficient `-0.000124`, |coef| `0.000124`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000120`, |coef| `0.000120`
- `lag_02__CT_place_UNKNOWN`: coefficient `-0.000113`, |coef| `0.000113`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000106`, |coef| `0.000106`
- `lag_05__CT_place_MAINHALL`: coefficient `-0.000103`, |coef| `0.000103`
- `lag_01__armor_diff`: coefficient `0.000102`, |coef| `0.000102`
- `lag_04__CT_place_MAINHALL`: coefficient `-0.000102`, |coef| `0.000102`
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000102`, |coef| `0.000102`
- `lag_01__centroid_distance_xy`: coefficient `-0.000095`, |coef| `0.000095`

## Top 10 utility ridge features

- `lag_00__T2__smoke`: coefficient `0.000087` (raises CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000086` (raises CT win probability)
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.000082` (lowers CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000077` (raises CT win probability)
- `lag_01__T_smoke_inv`: coefficient `-0.000069` (lowers CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000065` (raises CT win probability)
- `lag_01__T_molly_inv`: coefficient `-0.000064` (lowers CT win probability)
- `lag_01__T1__smoke`: coefficient `-0.000059` (lowers CT win probability)
- `lag_01__T2__molly`: coefficient `-0.000056` (lowers CT win probability)
- `lag_01__T3__smoke`: coefficient `-0.000053` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_UNKNOWN`: coefficient `-0.000655` (lowers CT win probability)
- `lag_00__CT_place_UNKNOWN`: coefficient `0.000407` (raises CT win probability)
- `lag_04__CT_place_UNKNOWN`: coefficient `-0.000346` (lowers CT win probability)
- `lag_07__CT_place_UNKNOWN`: coefficient `-0.000152` (lowers CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000133` (lowers CT win probability)
- `lag_13__T_place_WATER`: coefficient `0.000124` (raises CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000124` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000120` (lowers CT win probability)
- `lag_02__CT_place_UNKNOWN`: coefficient `-0.000113` (lowers CT win probability)
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000106` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `15243`, seconds `0.50`, LSTM delta `-0.0389`

Top all feature movements:
- `lag_01__CT_place_UNKNOWN`: contribution `-0.022984`
- `lag_01__T_place_TSPAWN`: contribution `-0.000532`
- `lag_00__CT_velocity_mean`: contribution `-0.000451`
- `lag_00__T_velocity_mean`: contribution `-0.000413`
- `lag_01__T_closest_enemy_dist`: contribution `-0.000387`

Top utility-only movements:
- `lag_01__molly_inv_diff`: contribution `-0.000240`
- `lag_00__T2__smoke`: contribution `-0.000192`
- `lag_01__utility_inv_diff`: contribution `-0.000187`
- `lag_01__T_smoke_inv`: contribution `-0.000157`
- `lag_01__T_molly_inv`: contribution `-0.000146`

### tick `16011`, seconds `12.50`, LSTM delta `-0.0134`

Top all feature movements:
- `lag_13__T_place_WATER`: contribution `-0.001419`
- `lag_15__T_place_TUNNEL`: contribution `-0.001021`
- `lag_05__CT_place_MAINHALL`: contribution `-0.000856`
- `lag_04__CT_place_MAINHALL`: contribution `-0.000845`
- `lag_15__T_place_WATER`: contribution `-0.000791`

Top utility-only movements:
- `lag_00__T_utility_damage_last_5s`: contribution `-0.000704`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.000281`

### tick `15275`, seconds `1.00`, LSTM delta `-0.0078`

Top all feature movements:
- `lag_02__CT_place_UNKNOWN`: contribution `-0.003972`
- `lag_00__CT_place_UNKNOWN`: contribution `-0.002856`
- `lag_02__T_place_TSPAWN`: contribution `-0.000135`
- `lag_02__armor_diff`: contribution `-0.000119`
- `lag_01__T_velocity_mean`: contribution `+0.000113`

Top utility-only movements:
- `lag_02__utility_inv_diff`: contribution `-0.000067`
- `lag_02__molly_inv_diff`: contribution `-0.000064`
- `lag_02__smoke_inv_diff`: contribution `-0.000059`
- `lag_02__T_smoke_inv`: contribution `-0.000058`
- `lag_01__T2__smoke`: contribution `+0.000049`

### tick `15307`, seconds `1.50`, LSTM delta `-0.0041`

Top all feature movements:
- `lag_00__CT_place_UNKNOWN`: contribution `-0.011425`
- `lag_01__CT_place_UNKNOWN`: contribution `+0.004601`
- `lag_03__CT_place_UNKNOWN`: contribution `+0.000757`
- `lag_03__CT_velocity_mean`: contribution `+0.000089`
- `lag_03__T_velocity_mean`: contribution `+0.000068`

Top utility-only movements:
- `lag_03__T2__smoke`: contribution `+0.000057`

### tick `15915`, seconds `11.00`, LSTM delta `+0.0039`

Top all feature movements:
- `lag_10__T_place_WATER`: contribution `+0.000821`
- `lag_12__T_place_WATER`: contribution `+0.000448`
- `lag_12__T_place_TUNNEL`: contribution `+0.000358`
- `lag_15__CT_place_HOUSE`: contribution `+0.000304`
- `lag_13__CT_place_HOUSE`: contribution `+0.000268`

Top utility-only movements:
- No utility movement among the top local contributors.
