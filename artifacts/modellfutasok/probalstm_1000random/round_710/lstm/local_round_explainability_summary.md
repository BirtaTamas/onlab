# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-faze-vs-aurora-bo3-OcxcOl9bFIHQQ2588nwUWG/faze-vs-aurora-m2-mirage.csv`
- round_num: `11`

## Largest probability jumps

- tick `77901`, seconds `58.50`, LSTM `0.9105`, delta `+0.3776`
- tick `77805`, seconds `57.00`, LSTM `0.6448`, delta `+0.3318`
- tick `77869`, seconds `58.00`, LSTM `0.5329`, delta `-0.2488`
- tick `77773`, seconds `56.50`, LSTM `0.3131`, delta `+0.1775`
- tick `77837`, seconds `57.50`, LSTM `0.7817`, delta `+0.1369`
- tick `74189`, seconds `0.50`, LSTM `0.0648`, delta `-0.0681`
- tick `77965`, seconds `59.50`, LSTM `0.8492`, delta `-0.0584`
- tick `76557`, seconds `37.50`, LSTM `0.2257`, delta `+0.0552`
- tick `78093`, seconds `61.50`, LSTM `0.8480`, delta `-0.0426`
- tick `75981`, seconds `28.50`, LSTM `0.1485`, delta `-0.0361`

## Top 15 local ridge features

- `lag_05__T2__is_scoped`: coefficient `0.003024`, |coef| `0.003024`
- `lag_00__CT_kills_last_3s`: coefficient `0.002781`, |coef| `0.002781`
- `lag_00__kill_diff_last_3s`: coefficient `0.002748`, |coef| `0.002748`
- `lag_00__T_place_CATWALK`: coefficient `-0.002295`, |coef| `0.002295`
- `lag_07__CT2__flash_duration`: coefficient `0.002250`, |coef| `0.002250`
- `lag_00__damage_diff_last_5s`: coefficient `0.001994`, |coef| `0.001994`
- `lag_00__CT_damage_last_5s`: coefficient `0.001878`, |coef| `0.001878`
- `lag_07__T_flashed_players`: coefficient `0.001873`, |coef| `0.001873`
- `lag_07__CT_place_LADDER`: coefficient `-0.001834`, |coef| `0.001834`
- `lag_15__T_utility_damage_last_5s`: coefficient `0.001821`, |coef| `0.001821`
- `lag_07__CT_place_SNIPERSNEST`: coefficient `0.001621`, |coef| `0.001621`
- `lag_05__T_flashed_players`: coefficient `-0.001604`, |coef| `0.001604`
- `lag_10__CT2__flash_duration`: coefficient `0.001548`, |coef| `0.001548`
- `lag_06__CT2__flash_duration`: coefficient `0.001544`, |coef| `0.001544`
- `lag_01__T2__is_scoped`: coefficient `-0.001532`, |coef| `0.001532`

## Top 10 utility ridge features

- `lag_07__CT2__flash_duration`: coefficient `0.002250` (raises CT win probability)
- `lag_15__T_utility_damage_last_5s`: coefficient `0.001821` (raises CT win probability)
- `lag_10__CT2__flash_duration`: coefficient `0.001548` (raises CT win probability)
- `lag_06__CT2__flash_duration`: coefficient `0.001544` (raises CT win probability)
- `lag_02__T5__flash_duration`: coefficient `0.001496` (raises CT win probability)
- `lag_00__T5__flash_duration`: coefficient `-0.001397` (lowers CT win probability)
- `lag_03__CT3__flash_duration`: coefficient `0.001377` (raises CT win probability)
- `lag_14__T_utility_damage_last_5s`: coefficient `0.001272` (raises CT win probability)
- `lag_08__CT2__flash_duration`: coefficient `0.001231` (raises CT win probability)
- `lag_00__T_utility_damage_last_5s`: coefficient `0.001222` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_05__T2__is_scoped`: coefficient `0.003024` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002781` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002748` (raises CT win probability)
- `lag_00__T_place_CATWALK`: coefficient `-0.002295` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001994` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001878` (raises CT win probability)
- `lag_07__T_flashed_players`: coefficient `0.001873` (raises CT win probability)
- `lag_07__CT_place_LADDER`: coefficient `-0.001834` (lowers CT win probability)
- `lag_07__CT_place_SNIPERSNEST`: coefficient `0.001621` (raises CT win probability)
- `lag_05__T_flashed_players`: coefficient `-0.001604` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `77901`, seconds `58.50`, LSTM delta `+0.3776`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.016057`
- `lag_01__T2__is_scoped`: contribution `+0.013505`
- `lag_10__CT_place_LADDER`: contribution `+0.013441`
- `lag_00__kill_diff_last_3s`: contribution `+0.013230`
- `lag_10__CT2__flash_duration`: contribution `+0.012140`

Top utility-only movements:
- `lag_10__CT2__flash_duration`: contribution `+0.012140`
- `lag_06__T5__flash_duration`: contribution `+0.006031`
- `lag_06__CT3__flash_duration`: contribution `+0.003609`
- `lag_06__CT_flash_duration_sum`: contribution `+0.003272`

### tick `77805`, seconds `57.00`, LSTM delta `+0.3318`

Top all feature movements:
- `lag_05__T2__is_scoped`: contribution `+0.026657`
- `lag_07__CT_place_LADDER`: contribution `+0.019071`
- `lag_07__CT2__flash_duration`: contribution `+0.017650`
- `lag_07__T_flashed_players`: contribution `+0.010840`
- `lag_07__CT_place_SNIPERSNEST`: contribution `+0.008680`

Top utility-only movements:
- `lag_07__CT2__flash_duration`: contribution `+0.017650`
- `lag_00__T5__flash_duration`: contribution `+0.007194`
- `lag_03__CT3__flash_duration`: contribution `+0.006290`
- `lag_00__T_utility_damage_last_5s`: contribution `+0.005407`
- `lag_07__CT_flash_duration_sum`: contribution `+0.003346`

### tick `77869`, seconds `58.00`, LSTM delta `-0.2488`

Top all feature movements:
- `lag_05__T2__is_scoped`: contribution `-0.026657`
- `lag_09__CT_place_LADDER`: contribution `-0.013795`
- `lag_01__T2__is_scoped`: contribution `-0.013505`
- `lag_00__T2__is_scoped`: contribution `-0.013342`
- `lag_02__T5__flash_duration`: contribution `-0.007703`

Top utility-only movements:
- `lag_02__T5__flash_duration`: contribution `-0.007703`
- `lag_09__CT2__flash_duration`: contribution `-0.005030`
- `lag_00__T_utility_damage_last_5s`: contribution `+0.003662`
- `lag_05__CT3__flash_duration`: contribution `-0.003047`

### tick `77773`, seconds `56.50`, LSTM delta `+0.1775`

Top all feature movements:
- `lag_06__CT_place_LADDER`: contribution `+0.014847`
- `lag_06__CT2__flash_duration`: contribution `+0.012111`
- `lag_00__CT_kills_last_3s`: contribution `+0.008029`
- `lag_02__T5__flash_duration`: contribution `+0.007703`
- `lag_06__CT_place_SNIPERSNEST`: contribution `+0.006872`

Top utility-only movements:
- `lag_06__CT2__flash_duration`: contribution `+0.012111`
- `lag_02__T5__flash_duration`: contribution `+0.007703`
- `lag_02__CT3__flash_duration`: contribution `+0.004568`
- `lag_06__CT_flash_duration_sum`: contribution `+0.004214`

### tick `77837`, seconds `57.50`, LSTM delta `+0.1369`

Top all feature movements:
- `lag_00__T2__is_scoped`: contribution `+0.013342`
- `lag_08__CT2__flash_duration`: contribution `+0.009655`
- `lag_08__CT_place_LADDER`: contribution `+0.007778`
- `lag_01__T1__duck_amount`: contribution `+0.005453`
- `lag_08__CT_place_SNIPERSNEST`: contribution `+0.004870`

Top utility-only movements:
- `lag_08__CT2__flash_duration`: contribution `+0.009655`
- `lag_04__T5__flash_duration`: contribution `+0.004274`
- `lag_01__T_utility_damage_last_5s`: contribution `+0.003798`
- `lag_04__CT3__flash_duration`: contribution `+0.003214`
- `lag_01__T5__flash_duration`: contribution `+0.002065`
