# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-ence-bo3-avzZyhQ46OR9GyYE_6NeM7/astralis-vs-ence-m3-overpass.csv`
- round_num: `19`

## Largest probability jumps

- tick `123391`, seconds `0.50`, LSTM `0.0198`, delta `-0.0303`
- tick `128479`, seconds `80.00`, LSTM `0.0139`, delta `-0.0108`
- tick `127999`, seconds `72.50`, LSTM `0.0374`, delta `+0.0102`
- tick `128415`, seconds `79.00`, LSTM `0.0268`, delta `-0.0089`
- tick `126015`, seconds `41.50`, LSTM `0.0202`, delta `+0.0082`
- tick `128095`, seconds `74.00`, LSTM `0.0380`, delta `+0.0079`
- tick `128223`, seconds `76.00`, LSTM `0.0414`, delta `+0.0077`
- tick `129023`, seconds `88.50`, LSTM `0.0040`, delta `-0.0076`
- tick `124127`, seconds `12.00`, LSTM `0.0313`, delta `+0.0074`
- tick `125215`, seconds `29.00`, LSTM `0.0214`, delta `-0.0063`

## Top 15 local ridge features

- `lag_01__CT_macro_A`: coefficient `-0.000266`, |coef| `0.000266`
- `lag_01__CT_place_BOMBSITEA`: coefficient `-0.000266`, |coef| `0.000266`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000198`, |coef| `0.000198`
- `lag_00__T_velocity_mean`: coefficient `-0.000197`, |coef| `0.000197`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000175`, |coef| `0.000175`
- `lag_00__CT_velocity_mean`: coefficient `-0.000169`, |coef| `0.000169`
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000169`, |coef| `0.000169`
- `lag_01__centroid_distance_xy`: coefficient `-0.000157`, |coef| `0.000157`
- `lag_01__T_round_start_equip_sum`: coefficient `-0.000156`, |coef| `0.000156`
- `lag_12__T3__is_scoped`: coefficient `-0.000137`, |coef| `0.000137`
- `lag_01__utility_inv_diff`: coefficient `0.000136`, |coef| `0.000136`
- `lag_00__CT_place_BRIDGE`: coefficient `-0.000135`, |coef| `0.000135`
- `lag_13__T3__is_scoped`: coefficient `-0.000135`, |coef| `0.000135`
- `lag_01__T1__Y`: coefficient `0.000130`, |coef| `0.000130`
- `lag_00__T3__is_scoped`: coefficient `0.000129`, |coef| `0.000129`

## Top 10 utility ridge features

- `lag_01__utility_inv_diff`: coefficient `0.000136` (raises CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000125` (raises CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000123` (raises CT win probability)
- `lag_01__T3__utility_total`: coefficient `-0.000111` (lowers CT win probability)
- `lag_01__T3__flash`: coefficient `-0.000111` (lowers CT win probability)
- `lag_01__CT1__flash`: coefficient `-0.000109` (lowers CT win probability)
- `lag_01__T_smoke_inv`: coefficient `-0.000109` (lowers CT win probability)
- `lag_01__T_utility_inv`: coefficient `-0.000107` (lowers CT win probability)
- `lag_01__T_molly_inv`: coefficient `-0.000096` (lowers CT win probability)
- `lag_01__T5__utility_total`: coefficient `-0.000096` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_macro_A`: coefficient `-0.000266` (lowers CT win probability)
- `lag_01__CT_place_BOMBSITEA`: coefficient `-0.000266` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000198` (lowers CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000197` (lowers CT win probability)
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000175` (lowers CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000169` (lowers CT win probability)
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000169` (lowers CT win probability)
- `lag_01__centroid_distance_xy`: coefficient `-0.000157` (lowers CT win probability)
- `lag_01__T_round_start_equip_sum`: coefficient `-0.000156` (lowers CT win probability)
- `lag_12__T3__is_scoped`: coefficient `-0.000137` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `123391`, seconds `0.50`, LSTM delta `-0.0303`

Top all feature movements:
- `lag_01__CT_macro_A`: contribution `-0.001539`
- `lag_01__CT_place_BOMBSITEA`: contribution `-0.001539`
- `lag_01__T_place_TSPAWN`: contribution `-0.000878`
- `lag_00__T_velocity_mean`: contribution `-0.000727`
- `lag_01__T_closest_enemy_dist`: contribution `-0.000643`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `-0.000388`
- `lag_01__molly_inv_diff`: contribution `-0.000343`
- `lag_01__smoke_inv_diff`: contribution `-0.000318`
- `lag_01__CT1__flash`: contribution `-0.000313`

### tick `128479`, seconds `80.00`, LSTM delta `-0.0108`

Top all feature movements:
- `lag_00__CT_place_PIPE`: contribution `-0.005787`
- `lag_12__T3__is_scoped`: contribution `-0.000881`
- `lag_15__T_place_TSTAIRS`: contribution `-0.000497`
- `lag_12__T_place_CONNECTOR`: contribution `-0.000406`
- `lag_00__CT_place_CANAL`: contribution `-0.000321`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `127999`, seconds `72.50`, LSTM delta `+0.0102`

Top all feature movements:
- `lag_00__T_place_TSTAIRS`: contribution `+0.000680`
- `lag_12__CT_place_WALKWAY`: contribution `+0.000534`
- `lag_00__T_place_ALLEY`: contribution `+0.000513`
- `lag_12__CT_place_WATER`: contribution `+0.000373`
- `lag_11__T_place_TUNNELS`: contribution `+0.000339`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `128415`, seconds `79.00`, LSTM delta `-0.0089`

Top all feature movements:
- `lag_10__T3__is_scoped`: contribution `-0.000662`
- `lag_10__T_place_CONNECTOR`: contribution `-0.000618`
- `lag_13__T_place_ALLEY`: contribution `-0.000498`
- `lag_00__T_place_UPPERPARK`: contribution `-0.000400`
- `lag_00__CT5__is_walking`: contribution `-0.000241`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `126015`, seconds `41.50`, LSTM delta `+0.0082`

Top all feature movements:
- `lag_00__CT_place_PIPE`: contribution `+0.005787`
- `lag_00__CT_place_CONSTRUCTION`: contribution `+0.000826`
- `lag_13__CT_place_PIPE`: contribution `+0.000553`
- `lag_11__T_place_FOUNTAIN`: contribution `-0.000268`
- `lag_12__T_place_TSTAIRS`: contribution `+0.000266`

Top utility-only movements:
- No utility movement among the top local contributors.
