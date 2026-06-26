# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-aurora-vs-falcons-bo3-5oHSxtVT-5F3Op7ZcgBMjW/aurora-vs-falcons-m1-inferno.csv`
- round_num: `8`

## Largest probability jumps

- tick `71958`, seconds `76.00`, LSTM `0.1617`, delta `-0.2216`
- tick `71350`, seconds `66.50`, LSTM `0.3463`, delta `+0.2129`
- tick `72022`, seconds `77.00`, LSTM `0.0198`, delta `-0.1028`
- tick `71670`, seconds `71.50`, LSTM `0.3461`, delta `-0.0614`
- tick `67126`, seconds `0.50`, LSTM `0.0312`, delta `-0.0443`
- tick `71990`, seconds `76.50`, LSTM `0.1226`, delta `-0.0391`
- tick `71798`, seconds `73.50`, LSTM `0.3702`, delta `+0.0359`
- tick `71574`, seconds `70.00`, LSTM `0.3670`, delta `+0.0351`
- tick `69782`, seconds `42.00`, LSTM `0.0884`, delta `-0.0322`
- tick `71382`, seconds `67.00`, LSTM `0.3777`, delta `+0.0314`

## Top 15 local ridge features

- `lag_00__damage_diff_last_5s`: coefficient `0.002414`, |coef| `0.002414`
- `lag_00__kill_diff_last_3s`: coefficient `0.002385`, |coef| `0.002385`
- `lag_03__T_place_ARCH`: coefficient `0.002320`, |coef| `0.002320`
- `lag_09__T_place_ARCH`: coefficient `-0.002313`, |coef| `0.002313`
- `lag_12__T_place_BACKALLEY`: coefficient `0.001916`, |coef| `0.001916`
- `lag_14__CT_place_RUINS`: coefficient `0.001881`, |coef| `0.001881`
- `lag_06__T_place_ARCH`: coefficient `-0.001770`, |coef| `0.001770`
- `lag_04__T_place_ARCH`: coefficient `0.001713`, |coef| `0.001713`
- `lag_03__T_place_CTSPAWN`: coefficient `-0.001688`, |coef| `0.001688`
- `lag_08__T4__has_bomb`: coefficient `0.001627`, |coef| `0.001627`
- `lag_08__bomb_events_last_5s`: coefficient `0.001626`, |coef| `0.001626`
- `lag_00__CT_kills_last_3s`: coefficient `0.001557`, |coef| `0.001557`
- `lag_01__T_place_BACKALLEY`: coefficient `-0.001527`, |coef| `0.001527`
- `lag_08__T5__has_bomb`: coefficient `-0.001434`, |coef| `0.001434`
- `lag_00__T5__alive`: coefficient `-0.001431`, |coef| `0.001431`

## Top 10 utility ridge features

- `lag_00__T5__molly`: coefficient `-0.001297` (lowers CT win probability)
- `lag_02__T1__smoke`: coefficient `-0.001181` (lowers CT win probability)
- `lag_01__T5__molly`: coefficient `-0.000780` (lowers CT win probability)
- `lag_01__T1__smoke`: coefficient `-0.000691` (lowers CT win probability)
- `lag_00__CT1__flash`: coefficient `0.000667` (raises CT win probability)
- `lag_07__T_A_site_active_smokes`: coefficient `-0.000645` (lowers CT win probability)
- `lag_05__T1__smoke`: coefficient `0.000589` (raises CT win probability)
- `lag_14__T1__flash`: coefficient `0.000570` (raises CT win probability)
- `lag_06__T1__smoke`: coefficient `-0.000560` (lowers CT win probability)
- `lag_04__T_A_site_active_smokes`: coefficient `-0.000551` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__damage_diff_last_5s`: coefficient `0.002414` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002385` (raises CT win probability)
- `lag_03__T_place_ARCH`: coefficient `0.002320` (raises CT win probability)
- `lag_09__T_place_ARCH`: coefficient `-0.002313` (lowers CT win probability)
- `lag_12__T_place_BACKALLEY`: coefficient `0.001916` (raises CT win probability)
- `lag_14__CT_place_RUINS`: coefficient `0.001881` (raises CT win probability)
- `lag_06__T_place_ARCH`: coefficient `-0.001770` (lowers CT win probability)
- `lag_04__T_place_ARCH`: coefficient `0.001713` (raises CT win probability)
- `lag_03__T_place_CTSPAWN`: coefficient `-0.001688` (lowers CT win probability)
- `lag_08__T4__has_bomb`: coefficient `0.001627` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `71958`, seconds `76.00`, LSTM delta `-0.2216`

Top all feature movements:
- `lag_03__T_place_ARCH`: contribution `-0.021583`
- `lag_09__T_place_ARCH`: contribution `-0.021516`
- `lag_06__T_place_ARCH`: contribution `-0.016471`
- `lag_00__CT_place_QUAD`: contribution `-0.011089`
- `lag_03__T_place_CTSPAWN`: contribution `-0.008052`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `71350`, seconds `66.50`, LSTM delta `+0.2129`

Top all feature movements:
- `lag_08__bomb_events_last_5s`: contribution `+0.006793`
- `lag_14__CT_place_RUINS`: contribution `+0.006571`
- `lag_12__T_place_BACKALLEY`: contribution `+0.005797`
- `lag_00__kill_diff_last_3s`: contribution `+0.005739`
- `lag_00__damage_diff_last_5s`: contribution `+0.005446`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `72022`, seconds `77.00`, LSTM delta `-0.1028`

Top all feature movements:
- `lag_08__T_place_ARCH`: contribution `-0.013206`
- `lag_00__CT_place_QUAD`: contribution `+0.011089`
- `lag_02__CT_place_QUAD`: contribution `-0.010124`
- `lag_05__T_place_ARCH`: contribution `-0.006384`
- `lag_00__kill_diff_last_3s`: contribution `-0.005739`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `71670`, seconds `71.50`, LSTM delta `-0.0614`

Top all feature movements:
- `lag_08__bomb_events_last_5s`: contribution `-0.006793`
- `lag_00__damage_diff_last_5s`: contribution `-0.005446`
- `lag_00__T_place_ARCH`: contribution `+0.005287`
- `lag_11__T_place_BACKALLEY`: contribution `-0.003573`
- `lag_00__CT_damage_last_5s`: contribution `-0.003064`

Top utility-only movements:
- `lag_12__T1__smoke`: contribution `-0.001046`

### tick `67126`, seconds `0.50`, LSTM delta `-0.0443`

Top all feature movements:
- `lag_01__T_closest_enemy_dist`: contribution `-0.002038`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.001675`
- `lag_00__CT_velocity_mean`: contribution `-0.001405`
- `lag_01__centroid_distance_xy`: contribution `-0.001390`
- `lag_01__T_place_TSPAWN`: contribution `-0.001344`

Top utility-only movements:
- `lag_01__T5__molly`: contribution `-0.001232`
- `lag_01__T1__smoke`: contribution `-0.001026`
- `lag_01__smoke_inv_diff`: contribution `-0.000945`
- `lag_01__molly_inv_diff`: contribution `-0.000908`
- `lag_01__T_smoke_inv`: contribution `-0.000850`
