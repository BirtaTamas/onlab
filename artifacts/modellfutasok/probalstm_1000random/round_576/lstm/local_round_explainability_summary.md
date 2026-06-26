# Local Round Explainability

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-gamerlegion-vs-the-mongolz-bo3-zdjI5BKx0DIgDYoNAnfKpI/gamerlegion-vs-the-mongolz-m2-mirage.csv`
- round_num: `3`

## Largest probability jumps

- tick `10473`, seconds `0.50`, LSTM `0.0135`, delta `-0.0311`
- tick `10505`, seconds `1.00`, LSTM `0.0087`, delta `-0.0048`
- tick `11145`, seconds `11.00`, LSTM `0.0067`, delta `-0.0028`
- tick `11017`, seconds `9.00`, LSTM `0.0099`, delta `-0.0022`
- tick `10985`, seconds `8.50`, LSTM `0.0121`, delta `+0.0022`
- tick `11497`, seconds `16.50`, LSTM `0.0047`, delta `-0.0019`
- tick `11209`, seconds `12.00`, LSTM `0.0097`, delta `+0.0016`
- tick `11177`, seconds `11.50`, LSTM `0.0082`, delta `+0.0015`
- tick `11305`, seconds `13.50`, LSTM `0.0090`, delta `-0.0014`
- tick `10921`, seconds `7.50`, LSTM `0.0094`, delta `+0.0014`

## Top 15 local ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000285`, |coef| `0.000285`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000259`, |coef| `0.000259`
- `lag_00__CT_velocity_mean`: coefficient `-0.000206`, |coef| `0.000206`
- `lag_01__smoke_inv_diff`: coefficient `0.000185`, |coef| `0.000185`
- `lag_01__armor_diff`: coefficient `0.000179`, |coef| `0.000179`
- `lag_01__molly_inv_diff`: coefficient `0.000178`, |coef| `0.000178`
- `lag_00__T_velocity_mean`: coefficient `-0.000169`, |coef| `0.000169`
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000164`, |coef| `0.000164`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000163`, |coef| `0.000163`
- `lag_01__utility_inv_diff`: coefficient `0.000162`, |coef| `0.000162`
- `lag_01__T5__has_bomb`: coefficient `-0.000153`, |coef| `0.000153`
- `lag_01__CT_armor_sum`: coefficient `0.000151`, |coef| `0.000151`
- `lag_00__T4__is_walking`: coefficient `0.000149`, |coef| `0.000149`
- `lag_01__centroid_distance_xy`: coefficient `-0.000147`, |coef| `0.000147`
- `lag_01__T1__utility_total`: coefficient `-0.000146`, |coef| `0.000146`

## Top 10 utility ridge features

- `lag_01__smoke_inv_diff`: coefficient `0.000185` (raises CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000178` (raises CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000162` (raises CT win probability)
- `lag_01__T1__utility_total`: coefficient `-0.000146` (lowers CT win probability)
- `lag_01__T_molly_inv`: coefficient `-0.000144` (lowers CT win probability)
- `lag_00__T4__smoke`: coefficient `0.000141` (raises CT win probability)
- `lag_01__T1__flash`: coefficient `-0.000133` (lowers CT win probability)
- `lag_01__T_smoke_inv`: coefficient `-0.000126` (lowers CT win probability)
- `lag_01__T2__molly`: coefficient `-0.000104` (lowers CT win probability)
- `lag_01__T_utility_inv`: coefficient `-0.000104` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000285` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000259` (lowers CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000206` (lowers CT win probability)
- `lag_01__armor_diff`: coefficient `0.000179` (raises CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000169` (lowers CT win probability)
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000164` (lowers CT win probability)
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000163` (lowers CT win probability)
- `lag_01__T5__has_bomb`: coefficient `-0.000153` (lowers CT win probability)
- `lag_01__CT_armor_sum`: coefficient `0.000151` (raises CT win probability)
- `lag_00__T4__is_walking`: coefficient `0.000149` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `10473`, seconds `0.50`, LSTM delta `-0.0311`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.001362`
- `lag_01__T_place_TSPAWN`: contribution `-0.001146`
- `lag_00__CT_velocity_mean`: contribution `-0.000713`
- `lag_01__smoke_inv_diff`: contribution `-0.000589`
- `lag_01__armor_diff`: contribution `-0.000503`

Top utility-only movements:
- `lag_01__smoke_inv_diff`: contribution `-0.000589`
- `lag_01__molly_inv_diff`: contribution `-0.000496`
- `lag_01__utility_inv_diff`: contribution `-0.000427`
- `lag_01__T1__utility_total`: contribution `-0.000330`
- `lag_01__T_molly_inv`: contribution `-0.000327`

### tick `10505`, seconds `1.00`, LSTM delta `-0.0048`

Top all feature movements:
- `lag_02__CT_place_CTSPAWN`: contribution `-0.000487`
- `lag_02__T_place_TSPAWN`: contribution `-0.000405`
- `lag_00__T_velocity_mean`: contribution `-0.000255`
- `lag_02__smoke_inv_diff`: contribution `-0.000241`
- `lag_02__armor_diff`: contribution `-0.000208`

Top utility-only movements:
- `lag_02__smoke_inv_diff`: contribution `-0.000241`
- `lag_02__molly_inv_diff`: contribution `-0.000204`
- `lag_02__utility_inv_diff`: contribution `-0.000178`
- `lag_02__T1__utility_total`: contribution `-0.000139`
- `lag_02__T_molly_inv`: contribution `-0.000135`

### tick `11145`, seconds `11.00`, LSTM delta `-0.0028`

Top all feature movements:
- `lag_04__T_utility_damage_last_5s`: contribution `-0.000486`
- `lag_04__utility_damage_diff_last_5s`: contribution `-0.000203`
- `lag_00__T_velocity_mean`: contribution `+0.000133`
- `lag_02__CT_place_JUNGLE`: contribution `-0.000120`
- `lag_05__CT_place_UNDERPASS`: contribution `-0.000111`

Top utility-only movements:
- `lag_04__T_utility_damage_last_5s`: contribution `-0.000486`
- `lag_04__utility_damage_diff_last_5s`: contribution `-0.000203`
- `lag_01__T2__utility_total`: contribution `+0.000074`
- `lag_01__T2__flash`: contribution `+0.000053`

### tick `11017`, seconds `9.00`, LSTM delta `-0.0022`

Top all feature movements:
- `lag_00__T_utility_damage_last_5s`: contribution `-0.000400`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.000160`
- `lag_00__CT_place_UNDERPASS`: contribution `-0.000114`
- `lag_02__CT_place_CTSPAWN`: contribution `+0.000106`
- `lag_14__T_place_SIDEALLEY`: contribution `-0.000104`

Top utility-only movements:
- `lag_00__T_utility_damage_last_5s`: contribution `-0.000400`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.000160`

### tick `10985`, seconds `8.50`, LSTM delta `+0.0022`

Top all feature movements:
- `lag_00__CT_velocity_mean`: contribution `+0.000333`
- `lag_01__CT_place_CTSPAWN`: contribution `+0.000296`
- `lag_00__CT_place_UNDERPASS`: contribution `+0.000114`
- `lag_02__CT_place_CTSPAWN`: contribution `+0.000106`
- `lag_01__CT_macro_OTHER`: contribution `+0.000076`

Top utility-only movements:
- No utility movement among the top local contributors.
