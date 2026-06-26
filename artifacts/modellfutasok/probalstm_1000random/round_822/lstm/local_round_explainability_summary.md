# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-vitality-vs-the-mongolz-bo3-JVS9HKMAkaZTRHkoiRSMP6/vitality-vs-the-mongolz-m1-mirage.csv`
- round_num: `15`

## Largest probability jumps

- tick `124904`, seconds `72.00`, LSTM `0.9266`, delta `+0.1773`
- tick `122888`, seconds `40.50`, LSTM `0.8812`, delta `+0.1662`
- tick `124808`, seconds `70.50`, LSTM `0.8568`, delta `+0.1584`
- tick `122696`, seconds `37.50`, LSTM `0.7612`, delta `+0.1478`
- tick `124680`, seconds `68.50`, LSTM `0.6858`, delta `-0.1302`
- tick `124872`, seconds `71.50`, LSTM `0.7493`, delta `-0.1168`
- tick `122856`, seconds `40.00`, LSTM `0.7150`, delta `-0.0675`
- tick `122920`, seconds `41.00`, LSTM `0.8304`, delta `-0.0508`
- tick `123144`, seconds `44.50`, LSTM `0.8826`, delta `+0.0494`
- tick `125224`, seconds `77.00`, LSTM `0.8387`, delta `-0.0423`

## Top 15 local ridge features

- `lag_15__T_flashes_last_5s`: coefficient `0.002071`, |coef| `0.002071`
- `lag_00__kill_diff_last_3s`: coefficient `0.001975`, |coef| `0.001975`
- `lag_00__CT_kills_last_3s`: coefficient `0.001799`, |coef| `0.001799`
- `lag_01__T_flashes_last_5s`: coefficient `0.001460`, |coef| `0.001460`
- `lag_05__T_flashes_last_5s`: coefficient `-0.001415`, |coef| `0.001415`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001366`, |coef| `0.001366`
- `lag_08__T_flashes_last_5s`: coefficient `-0.001347`, |coef| `0.001347`
- `lag_03__T_place_CONNECTOR`: coefficient `-0.001322`, |coef| `0.001322`
- `lag_11__T_flashes_last_5s`: coefficient `-0.001254`, |coef| `0.001254`
- `lag_07__T_place_CONNECTOR`: coefficient `0.001247`, |coef| `0.001247`
- `lag_06__T_shots_fired_sum`: coefficient `-0.001244`, |coef| `0.001244`
- `lag_06__T_place_UNDERPASS`: coefficient `-0.001183`, |coef| `0.001183`
- `lag_00__CT_place_TRUCK`: coefficient `0.001155`, |coef| `0.001155`
- `lag_00__CT5__is_walking`: coefficient `-0.001117`, |coef| `0.001117`
- `lag_01__T2__shots_fired`: coefficient `0.001105`, |coef| `0.001105`

## Top 10 utility ridge features

- `lag_15__T_flashes_last_5s`: coefficient `0.002071` (raises CT win probability)
- `lag_01__T_flashes_last_5s`: coefficient `0.001460` (raises CT win probability)
- `lag_05__T_flashes_last_5s`: coefficient `-0.001415` (lowers CT win probability)
- `lag_08__T_flashes_last_5s`: coefficient `-0.001347` (lowers CT win probability)
- `lag_11__T_flashes_last_5s`: coefficient `-0.001254` (lowers CT win probability)
- `lag_03__T2__flash_duration`: coefficient `0.000952` (raises CT win probability)
- `lag_02__T_flashes_last_5s`: coefficient `0.000838` (raises CT win probability)
- `lag_14__T_flashes_last_5s`: coefficient `0.000800` (raises CT win probability)
- `lag_00__T2__flash_duration`: coefficient `-0.000733` (lowers CT win probability)
- `lag_12__T_flashes_last_5s`: coefficient `-0.000642` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.001975` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001799` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001366` (lowers CT win probability)
- `lag_03__T_place_CONNECTOR`: coefficient `-0.001322` (lowers CT win probability)
- `lag_07__T_place_CONNECTOR`: coefficient `0.001247` (raises CT win probability)
- `lag_06__T_shots_fired_sum`: coefficient `-0.001244` (lowers CT win probability)
- `lag_06__T_place_UNDERPASS`: coefficient `-0.001183` (lowers CT win probability)
- `lag_00__CT_place_TRUCK`: coefficient `0.001155` (raises CT win probability)
- `lag_00__CT5__is_walking`: coefficient `-0.001117` (lowers CT win probability)
- `lag_01__T2__shots_fired`: coefficient `0.001105` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `124904`, seconds `72.00`, LSTM delta `+0.1773`

Top all feature movements:
- `lag_08__T_flashes_last_5s`: contribution `+0.012205`
- `lag_03__T_place_CONNECTOR`: contribution `+0.006404`
- `lag_00__CT_kills_last_3s`: contribution `+0.005194`
- `lag_00__T_shots_fired_sum`: contribution `+0.005121`
- `lag_00__kill_diff_last_3s`: contribution `+0.004755`

Top utility-only movements:
- `lag_08__T_flashes_last_5s`: contribution `+0.012205`

### tick `122888`, seconds `40.50`, LSTM delta `+0.1662`

Top all feature movements:
- `lag_06__T_shots_fired_sum`: contribution `+0.017725`
- `lag_07__T_shots_fired_sum`: contribution `+0.005855`
- `lag_04__T_shots_fired_sum`: contribution `+0.005563`
- `lag_02__T1__shots_fired`: contribution `+0.005389`
- `lag_06__T_place_UNDERPASS`: contribution `+0.004632`

Top utility-only movements:
- `lag_09__T2__flash_duration`: contribution `+0.004087`
- `lag_06__T2__flash_duration`: contribution `+0.004063`
- `lag_01__CT_utility_damage_last_5s`: contribution `+0.002822`
- `lag_01__utility_damage_diff_last_5s`: contribution `+0.001932`

### tick `124808`, seconds `70.50`, LSTM delta `+0.1584`

Top all feature movements:
- `lag_15__T_flashes_last_5s`: contribution `+0.018763`
- `lag_05__T_flashes_last_5s`: contribution `+0.012822`
- `lag_07__T_place_CONNECTOR`: contribution `+0.006037`
- `lag_00__CT_kills_last_3s`: contribution `+0.005194`
- `lag_00__kill_diff_last_3s`: contribution `+0.004755`

Top utility-only movements:
- `lag_15__T_flashes_last_5s`: contribution `+0.018763`
- `lag_05__T_flashes_last_5s`: contribution `+0.012822`

### tick `122696`, seconds `37.50`, LSTM delta `+0.1478`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.019461`
- `lag_03__T2__flash_duration`: contribution `+0.007054`
- `lag_00__T2__flash_duration`: contribution `+0.005427`
- `lag_00__CT_kills_last_3s`: contribution `+0.005194`
- `lag_00__kill_diff_last_3s`: contribution `+0.004755`

Top utility-only movements:
- `lag_03__T2__flash_duration`: contribution `+0.007054`
- `lag_00__T2__flash_duration`: contribution `+0.005427`

### tick `124680`, seconds `68.50`, LSTM delta `-0.1302`

Top all feature movements:
- `lag_01__T_flashes_last_5s`: contribution `-0.013226`
- `lag_11__T_flashes_last_5s`: contribution `-0.011363`
- `lag_03__T_place_CONNECTOR`: contribution `-0.006404`
- `lag_00__kill_diff_last_3s`: contribution `-0.004755`
- `lag_02__T_place_CONNECTOR`: contribution `-0.003695`

Top utility-only movements:
- `lag_01__T_flashes_last_5s`: contribution `-0.013226`
- `lag_11__T_flashes_last_5s`: contribution `-0.011363`
