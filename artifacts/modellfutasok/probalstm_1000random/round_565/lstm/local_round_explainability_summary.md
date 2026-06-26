# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-tyloo-vs-vitality-bo3-aF98ikh3PjdqKlkdIJn9tC/tyloo-vs-vitality-m1-inferno.csv`
- round_num: `16`

## Largest probability jumps

- tick `130984`, seconds `33.50`, LSTM `0.3024`, delta `-0.3288`
- tick `130376`, seconds `24.00`, LSTM `0.3359`, delta `-0.2443`
- tick `130920`, seconds `32.50`, LSTM `0.6031`, delta `+0.2120`
- tick `131144`, seconds `36.00`, LSTM `0.0340`, delta `-0.1890`
- tick `131048`, seconds `34.50`, LSTM `0.2327`, delta `-0.0501`
- tick `130536`, seconds `26.50`, LSTM `0.2971`, delta `-0.0462`
- tick `131112`, seconds `35.50`, LSTM `0.2230`, delta `-0.0324`
- tick `129160`, seconds `5.00`, LSTM `0.5684`, delta `+0.0297`
- tick `130856`, seconds `31.50`, LSTM `0.3772`, delta `+0.0287`
- tick `130952`, seconds `33.00`, LSTM `0.6313`, delta `+0.0282`

## Top 15 local ridge features

- `lag_01__CT5__shots_fired`: coefficient `0.002895`, |coef| `0.002895`
- `lag_13__CT_place_BALCONY`: coefficient `-0.002624`, |coef| `0.002624`
- `lag_00__CT_place_BANANA`: coefficient `0.002481`, |coef| `0.002481`
- `lag_11__T2__flash_duration`: coefficient `-0.002437`, |coef| `0.002437`
- `lag_00__kill_diff_last_3s`: coefficient `0.002425`, |coef| `0.002425`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002342`, |coef| `0.002342`
- `lag_00__T_kills_last_3s`: coefficient `-0.002319`, |coef| `0.002319`
- `lag_15__T2__shots_fired`: coefficient `-0.002278`, |coef| `0.002278`
- `lag_13__T1__flash_duration`: coefficient `0.002086`, |coef| `0.002086`
- `lag_13__T2__flash_duration`: coefficient `0.002062`, |coef| `0.002062`
- `lag_11__T_flash_duration_sum`: coefficient `-0.001791`, |coef| `0.001791`
- `lag_07__T2__shots_fired`: coefficient `-0.001772`, |coef| `0.001772`
- `lag_14__T4__flash_duration`: coefficient `0.001766`, |coef| `0.001766`
- `lag_07__T_flashed_players`: coefficient `0.001764`, |coef| `0.001764`
- `lag_01__CT_shots_fired_sum`: coefficient `0.001763`, |coef| `0.001763`

## Top 10 utility ridge features

- `lag_11__T2__flash_duration`: coefficient `-0.002437` (lowers CT win probability)
- `lag_13__T1__flash_duration`: coefficient `0.002086` (raises CT win probability)
- `lag_13__T2__flash_duration`: coefficient `0.002062` (raises CT win probability)
- `lag_11__T_flash_duration_sum`: coefficient `-0.001791` (lowers CT win probability)
- `lag_14__T4__flash_duration`: coefficient `0.001766` (raises CT win probability)
- `lag_13__T_flash_duration_sum`: coefficient `0.001606` (raises CT win probability)
- `lag_11__T1__flash_duration`: coefficient `-0.001557` (lowers CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `0.001553` (raises CT win probability)
- `lag_01__T_B_site_active_infernos`: coefficient `-0.001338` (lowers CT win probability)
- `lag_02__T1__flash_duration`: coefficient `-0.001293` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT5__shots_fired`: coefficient `0.002895` (raises CT win probability)
- `lag_13__CT_place_BALCONY`: coefficient `-0.002624` (lowers CT win probability)
- `lag_00__CT_place_BANANA`: coefficient `0.002481` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002425` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002342` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002319` (lowers CT win probability)
- `lag_15__T2__shots_fired`: coefficient `-0.002278` (lowers CT win probability)
- `lag_07__T2__shots_fired`: coefficient `-0.001772` (lowers CT win probability)
- `lag_07__T_flashed_players`: coefficient `0.001764` (raises CT win probability)
- `lag_01__CT_shots_fired_sum`: coefficient `0.001763` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `130984`, seconds `33.50`, LSTM delta `-0.3288`

Top all feature movements:
- `lag_13__CT_place_BALCONY`: contribution `-0.016839`
- `lag_01__CT5__shots_fired`: contribution `-0.015307`
- `lag_13__T2__flash_duration`: contribution `-0.012865`
- `lag_13__T1__flash_duration`: contribution `-0.012790`
- `lag_14__T4__flash_duration`: contribution `-0.008272`

Top utility-only movements:
- `lag_13__T2__flash_duration`: contribution `-0.012865`
- `lag_13__T1__flash_duration`: contribution `-0.012790`
- `lag_14__T4__flash_duration`: contribution `-0.008272`
- `lag_13__T_flash_duration_sum`: contribution `-0.008199`

### tick `130376`, seconds `24.00`, LSTM delta `-0.2443`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.015619`
- `lag_11__T2__flash_duration`: contribution `-0.014484`
- `lag_00__CT2__flash_duration`: contribution `-0.012068`
- `lag_02__CT_place_QUAD`: contribution `-0.011228`
- `lag_02__T1__flash_duration`: contribution `-0.009922`

Top utility-only movements:
- `lag_11__T2__flash_duration`: contribution `-0.014484`
- `lag_00__CT2__flash_duration`: contribution `-0.012068`
- `lag_02__T1__flash_duration`: contribution `-0.009922`
- `lag_11__T_flash_duration_sum`: contribution `-0.005698`
- `lag_01__T_B_site_active_infernos`: contribution `-0.003784`

### tick `130920`, seconds `32.50`, LSTM delta `+0.2120`

Top all feature movements:
- `lag_15__T_shots_fired_sum`: contribution `+0.027974`
- `lag_15__T2__shots_fired`: contribution `+0.021441`
- `lag_11__T2__flash_duration`: contribution `+0.015205`
- `lag_11__T1__flash_duration`: contribution `+0.009549`
- `lag_11__T_flash_duration_sum`: contribution `+0.009145`

Top utility-only movements:
- `lag_11__T2__flash_duration`: contribution `+0.015205`
- `lag_11__T1__flash_duration`: contribution `+0.009549`
- `lag_11__T_flash_duration_sum`: contribution `+0.009145`
- `lag_12__T4__flash_duration`: contribution `+0.005026`

### tick `131144`, seconds `36.00`, LSTM delta `-0.1890`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.007346`
- `lag_00__CT_place_BANANA`: contribution `-0.007344`
- `lag_04__T1__shots_fired`: contribution `-0.007238`
- `lag_03__CT_place_LIBRARY`: contribution `-0.006463`
- `lag_00__kill_diff_last_3s`: contribution `-0.005836`

Top utility-only movements:
- `lag_01__T_B_site_active_infernos`: contribution `-0.003784`

### tick `131048`, seconds `34.50`, LSTM delta `-0.0501`

Top all feature movements:
- `lag_15__T2__shots_fired`: contribution `+0.008040`
- `lag_15__T_shots_fired_sum`: contribution `+0.007298`
- `lag_01__T_shots_fired_sum`: contribution `+0.007145`
- `lag_15__CT_place_BALCONY`: contribution `-0.006092`
- `lag_00__CT_place_LIBRARY`: contribution `-0.005572`

Top utility-only movements:
- `lag_15__T1__flash_duration`: contribution `-0.003228`
- `lag_15__T2__flash_duration`: contribution `-0.002801`
- `lag_15__T_flash_duration_sum`: contribution `-0.002702`
