# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-nrg-vs-vitality-bo3-7fVRmFXct71SEzAlz8IKvC/nrg-vs-vitality-m2-dust2.csv`
- round_num: `6`

## Largest probability jumps

- tick `45543`, seconds `53.00`, LSTM `0.0484`, delta `-0.3974`
- tick `45511`, seconds `52.50`, LSTM `0.4458`, delta `+0.2262`
- tick `45447`, seconds `51.50`, LSTM `0.1864`, delta `-0.0588`
- tick `45319`, seconds `49.50`, LSTM `0.2101`, delta `-0.0553`
- tick `45287`, seconds `49.00`, LSTM `0.2654`, delta `-0.0467`
- tick `45223`, seconds `48.00`, LSTM `0.2998`, delta `-0.0411`
- tick `45703`, seconds `55.50`, LSTM `0.0092`, delta `-0.0386`
- tick `44679`, seconds `39.50`, LSTM `0.3141`, delta `+0.0369`
- tick `45479`, seconds `52.00`, LSTM `0.2196`, delta `+0.0332`
- tick `42663`, seconds `8.00`, LSTM `0.4373`, delta `-0.0325`

## Top 15 local ridge features

- `lag_01__T3__flash_duration`: coefficient `0.002257`, |coef| `0.002257`
- `lag_10__T_place_BDOORS`: coefficient `-0.001922`, |coef| `0.001922`
- `lag_00__T_shots_fired_sum`: coefficient `0.001905`, |coef| `0.001905`
- `lag_07__T_place_BDOORS`: coefficient `-0.001766`, |coef| `0.001766`
- `lag_00__T5__shots_fired`: coefficient `0.001715`, |coef| `0.001715`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001686`, |coef| `0.001686`
- `lag_00__T1__is_scoped`: coefficient `0.001419`, |coef| `0.001419`
- `lag_00__T_kills_last_3s`: coefficient `-0.001416`, |coef| `0.001416`
- `lag_09__CT_place_ARAMP`: coefficient `0.001370`, |coef| `0.001370`
- `lag_00__T_place_BDOORS`: coefficient `-0.001363`, |coef| `0.001363`
- `lag_00__kill_diff_last_3s`: coefficient `0.001287`, |coef| `0.001287`
- `lag_00__CT3__flash_duration`: coefficient `0.001246`, |coef| `0.001246`
- `lag_11__T_flashed_players`: coefficient `0.001134`, |coef| `0.001134`
- `lag_00__damage_diff_last_5s`: coefficient `0.001120`, |coef| `0.001120`
- `lag_02__T3__flash_duration`: coefficient `-0.001104`, |coef| `0.001104`

## Top 10 utility ridge features

- `lag_01__T3__flash_duration`: coefficient `0.002257` (raises CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `0.001246` (raises CT win probability)
- `lag_02__T3__flash_duration`: coefficient `-0.001104` (lowers CT win probability)
- `lag_02__CT3__flash_duration`: coefficient `-0.001031` (lowers CT win probability)
- `lag_01__T_flash_duration_sum`: coefficient `0.000979` (raises CT win probability)
- `lag_01__CT3__flash_duration`: coefficient `0.000921` (raises CT win probability)
- `lag_00__CT2__smoke`: coefficient `0.000798` (raises CT win probability)
- `lag_09__CT_B_site_active_infernos`: coefficient `-0.000772` (lowers CT win probability)
- `lag_11__T4__flash_duration`: coefficient `0.000713` (raises CT win probability)
- `lag_00__T3__flash_duration`: coefficient `-0.000701` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_10__T_place_BDOORS`: coefficient `-0.001922` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `0.001905` (raises CT win probability)
- `lag_07__T_place_BDOORS`: coefficient `-0.001766` (lowers CT win probability)
- `lag_00__T5__shots_fired`: coefficient `0.001715` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001686` (raises CT win probability)
- `lag_00__T1__is_scoped`: coefficient `0.001419` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001416` (lowers CT win probability)
- `lag_09__CT_place_ARAMP`: coefficient `0.001370` (raises CT win probability)
- `lag_00__T_place_BDOORS`: coefficient `-0.001363` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001287` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `45543`, seconds `53.00`, LSTM delta `-0.3974`

Top all feature movements:
- `lag_10__T_place_BDOORS`: contribution `-0.024043`
- `lag_07__T_place_BDOORS`: contribution `-0.022093`
- `lag_01__T3__flash_duration`: contribution `-0.015421`
- `lag_00__T_shots_fired_sum`: contribution `-0.014285`
- `lag_01__T_place_BDOORS`: contribution `-0.011521`

Top utility-only movements:
- `lag_01__T3__flash_duration`: contribution `-0.015421`
- `lag_02__T3__flash_duration`: contribution `-0.007542`
- `lag_00__CT3__flash_duration`: contribution `-0.006757`
- `lag_02__CT3__flash_duration`: contribution `-0.005589`

### tick `45511`, seconds `52.50`, LSTM delta `+0.2262`

Top all feature movements:
- `lag_00__T_place_BDOORS`: contribution `+0.017052`
- `lag_01__T3__flash_duration`: contribution `+0.015421`
- `lag_08__T_place_BDOORS`: contribution `+0.010463`
- `lag_06__T_place_BDOORS`: contribution `+0.009112`
- `lag_00__T1__is_scoped`: contribution `+0.008108`

Top utility-only movements:
- `lag_01__T3__flash_duration`: contribution `+0.015421`
- `lag_01__CT3__flash_duration`: contribution `+0.004996`
- `lag_00__T3__flash_duration`: contribution `+0.004787`
- `lag_01__T_flash_duration_sum`: contribution `+0.002968`

### tick `45447`, seconds `51.50`, LSTM delta `-0.0588`

Top all feature movements:
- `lag_07__T_place_BDOORS`: contribution `-0.022093`
- `lag_06__T_place_BDOORS`: contribution `+0.009112`
- `lag_11__T1__is_scoped`: contribution `-0.005688`
- `lag_13__T_flashed_players`: contribution `-0.004505`
- `lag_14__T_place_MIDDOORS`: contribution `-0.004504`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `45319`, seconds `49.50`, LSTM delta `-0.0553`

Top all feature movements:
- `lag_00__T_place_BDOORS`: contribution `-0.017052`
- `lag_02__T_place_BDOORS`: contribution `-0.011845`
- `lag_02__T_place_MIDDOORS`: contribution `-0.004557`
- `lag_03__T_place_BDOORS`: contribution `-0.003584`
- `lag_15__T_place_MIDDOORS`: contribution `-0.002974`

Top utility-only movements:
- `lag_02__CT_B_site_active_infernos`: contribution `-0.001970`

### tick `45287`, seconds `49.00`, LSTM delta `-0.0467`

Top all feature movements:
- `lag_02__T_place_BDOORS`: contribution `-0.011845`
- `lag_01__T_place_BDOORS`: contribution `+0.011521`
- `lag_02__T_place_MIDDOORS`: contribution `-0.004557`
- `lag_14__T_place_MIDDOORS`: contribution `-0.004504`
- `lag_08__T_flashed_players`: contribution `-0.003801`

Top utility-only movements:
- `lag_01__CT_B_site_active_infernos`: contribution `-0.001175`
- `lag_00__T5__molly`: contribution `-0.000940`
