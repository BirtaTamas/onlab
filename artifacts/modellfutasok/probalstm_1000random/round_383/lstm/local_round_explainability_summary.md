# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-g2-vs-falcons-bo3-VnJ8NRf6cDNnH9OuqiscGr/g2-vs-falcons-m1-ancient.csv`
- round_num: `6`

## Largest probability jumps

- tick `44111`, seconds `36.00`, LSTM `0.9117`, delta `+0.0988`
- tick `44399`, seconds `40.50`, LSTM `0.9615`, delta `+0.0370`
- tick `41839`, seconds `0.50`, LSTM `0.9401`, delta `+0.0358`
- tick `44303`, seconds `39.00`, LSTM `0.9096`, delta `-0.0319`
- tick `44047`, seconds `35.00`, LSTM `0.8376`, delta `-0.0304`
- tick `44015`, seconds `34.50`, LSTM `0.8681`, delta `-0.0273`
- tick `44079`, seconds `35.50`, LSTM `0.8129`, delta `-0.0247`
- tick `44271`, seconds `38.50`, LSTM `0.9415`, delta `+0.0223`
- tick `43759`, seconds `30.50`, LSTM `0.9233`, delta `-0.0154`
- tick `43983`, seconds `34.00`, LSTM `0.8954`, delta `-0.0148`

## Top 15 local ridge features

- `lag_11__T_place_TSIDELOWER`: coefficient `-0.000677`, |coef| `0.000677`
- `lag_02__T_place_RAMP`: coefficient `-0.000673`, |coef| `0.000673`
- `lag_05__T_flashed_players`: coefficient `0.000671`, |coef| `0.000671`
- `lag_01__CT_place_UNKNOWN`: coefficient `0.000663`, |coef| `0.000663`
- `lag_05__CT1__flash_duration`: coefficient `0.000638`, |coef| `0.000638`
- `lag_03__T_place_RAMP`: coefficient `-0.000595`, |coef| `0.000595`
- `lag_00__T_bomb_zone_count`: coefficient `0.000583`, |coef| `0.000583`
- `lag_01__T_place_RAMP`: coefficient `0.000536`, |coef| `0.000536`
- `lag_00__T_flashed_players`: coefficient `-0.000495`, |coef| `0.000495`
- `lag_00__CT_kills_last_3s`: coefficient `0.000481`, |coef| `0.000481`
- `lag_13__CT_A_site_active_infernos`: coefficient `-0.000459`, |coef| `0.000459`
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.000454`, |coef| `0.000454`
- `lag_00__T_macro_B`: coefficient `-0.000454`, |coef| `0.000454`
- `lag_08__T2__duck_amount`: coefficient `-0.000450`, |coef| `0.000450`
- `lag_03__CT3__is_scoped`: coefficient `-0.000439`, |coef| `0.000439`

## Top 10 utility ridge features

- `lag_05__CT1__flash_duration`: coefficient `0.000638` (raises CT win probability)
- `lag_13__CT_A_site_active_infernos`: coefficient `-0.000459` (lowers CT win probability)
- `lag_00__T5__flash_duration`: coefficient `-0.000370` (lowers CT win probability)
- `lag_00__T_flash_duration_sum`: coefficient `-0.000364` (lowers CT win probability)
- `lag_00__T3__flash_duration`: coefficient `-0.000348` (lowers CT win probability)
- `lag_05__T_flash_duration_sum`: coefficient `0.000342` (raises CT win probability)
- `lag_03__CT1__flash_duration`: coefficient `-0.000342` (lowers CT win probability)
- `lag_04__CT1__flash_duration`: coefficient `-0.000338` (lowers CT win probability)
- `lag_02__CT1__flash_duration`: coefficient `-0.000320` (lowers CT win probability)
- `lag_05__CT_flash_duration_sum`: coefficient `0.000304` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_11__T_place_TSIDELOWER`: coefficient `-0.000677` (lowers CT win probability)
- `lag_02__T_place_RAMP`: coefficient `-0.000673` (lowers CT win probability)
- `lag_05__T_flashed_players`: coefficient `0.000671` (raises CT win probability)
- `lag_01__CT_place_UNKNOWN`: coefficient `0.000663` (raises CT win probability)
- `lag_03__T_place_RAMP`: coefficient `-0.000595` (lowers CT win probability)
- `lag_00__T_bomb_zone_count`: coefficient `0.000583` (raises CT win probability)
- `lag_01__T_place_RAMP`: coefficient `0.000536` (raises CT win probability)
- `lag_00__T_flashed_players`: coefficient `-0.000495` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000481` (raises CT win probability)
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.000454` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `44111`, seconds `36.00`, LSTM delta `+0.0988`

Top all feature movements:
- `lag_05__T_flashed_players`: contribution `+0.005180`
- `lag_11__T_place_TSIDELOWER`: contribution `+0.005078`
- `lag_02__T_place_RAMP`: contribution `+0.004760`
- `lag_05__CT1__flash_duration`: contribution `+0.004236`
- `lag_00__T_bomb_zone_count`: contribution `+0.003396`

Top utility-only movements:
- `lag_05__CT1__flash_duration`: contribution `+0.004236`
- `lag_13__CT_A_site_active_infernos`: contribution `+0.001620`
- `lag_05__T_flash_duration_sum`: contribution `+0.001436`
- `lag_00__T_flash_duration_sum`: contribution `+0.001314`

### tick `44399`, seconds `40.50`, LSTM delta `+0.0370`

Top all feature movements:
- `lag_11__T_place_RAMP`: contribution `-0.002547`
- `lag_04__CT1__flash_duration`: contribution `+0.002242`
- `lag_00__CT_kills_last_3s`: contribution `+0.001387`
- `lag_09__T_flashed_players`: contribution `+0.001367`
- `lag_05__CT_shots_fired_sum`: contribution `+0.001353`

Top utility-only movements:
- `lag_04__CT1__flash_duration`: contribution `+0.002242`
- `lag_14__CT1__flash_duration`: contribution `+0.001151`
- `lag_00__T3__flash_duration`: contribution `+0.000894`

### tick `41839`, seconds `0.50`, LSTM delta `+0.0358`

Top all feature movements:
- `lag_01__CT_place_UNKNOWN`: contribution `+0.023278`
- `lag_01__T_place_TSPAWN`: contribution `+0.000472`
- `lag_01__utility_inv_diff`: contribution `+0.000386`
- `lag_01__smoke_inv_diff`: contribution `+0.000380`
- `lag_01__molly_inv_diff`: contribution `+0.000365`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `+0.000386`
- `lag_01__smoke_inv_diff`: contribution `+0.000380`
- `lag_01__molly_inv_diff`: contribution `+0.000365`
- `lag_01__CT_molly_inv`: contribution `+0.000299`
- `lag_01__CT_utility_inv`: contribution `+0.000241`

### tick `44303`, seconds `39.00`, LSTM delta `-0.0319`

Top all feature movements:
- `lag_02__T_place_RAMP`: contribution `-0.002380`
- `lag_00__T5__flash_duration`: contribution `-0.002145`
- `lag_08__T_place_RAMP`: contribution `+0.001919`
- `lag_00__T_flashed_players`: contribution `-0.001910`
- `lag_11__T_flashed_players`: contribution `-0.001487`

Top utility-only movements:
- `lag_00__T5__flash_duration`: contribution `-0.002145`
- `lag_00__T_flash_duration_sum`: contribution `-0.001248`
- `lag_00__T3__flash_duration`: contribution `-0.000894`
- `lag_11__CT1__flash_duration`: contribution `-0.000850`
- `lag_00__CT_flash_duration_sum`: contribution `-0.000787`

### tick `44047`, seconds `35.00`, LSTM delta `-0.0304`

Top all feature movements:
- `lag_02__T_place_RAMP`: contribution `+0.002380`
- `lag_03__CT1__flash_duration`: contribution `-0.002268`
- `lag_14__T_place_TSIDELOWER`: contribution `-0.002268`
- `lag_09__T_place_TSIDELOWER`: contribution `+0.001960`
- `lag_01__T_place_RAMP`: contribution `-0.001895`

Top utility-only movements:
- `lag_03__CT1__flash_duration`: contribution `-0.002268`
