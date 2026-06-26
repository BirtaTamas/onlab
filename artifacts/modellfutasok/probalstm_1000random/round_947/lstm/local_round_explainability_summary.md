# Local Round Explainability

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-aurora-vs-heroic-bo3-872ZDvS9tk2PrtGeXVe8dJ/aurora-vs-heroic-m1-train-p3.csv`
- round_num: `9`

## Largest probability jumps

- tick `71643`, seconds `89.00`, LSTM `0.8986`, delta `+0.2941`
- tick `71227`, seconds `82.50`, LSTM `0.7723`, delta `+0.2685`
- tick `71291`, seconds `83.50`, LSTM `0.8597`, delta `+0.1376`
- tick `71195`, seconds `82.00`, LSTM `0.5038`, delta `-0.1042`
- tick `71035`, seconds `79.50`, LSTM `0.6071`, delta `+0.0910`
- tick `70459`, seconds `70.50`, LSTM `0.4706`, delta `+0.0791`
- tick `69659`, seconds `58.00`, LSTM `0.3747`, delta `-0.0762`
- tick `69691`, seconds `58.50`, LSTM `0.3026`, delta `-0.0721`
- tick `66875`, seconds `14.50`, LSTM `0.4297`, delta `-0.0704`
- tick `71547`, seconds `87.50`, LSTM `0.6425`, delta `-0.0658`

## Top 15 local ridge features

- `lag_01__CT_place_TMAIN`: coefficient `-0.002615`, |coef| `0.002615`
- `lag_08__T_shots_fired_sum`: coefficient `-0.002570`, |coef| `0.002570`
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.002285`, |coef| `0.002285`
- `lag_08__T4__shots_fired`: coefficient `-0.002062`, |coef| `0.002062`
- `lag_01__T_utility_damage_last_5s`: coefficient `-0.001985`, |coef| `0.001985`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001549`, |coef| `0.001549`
- `lag_09__CT_place_ELECTRICALBOX`: coefficient `0.001510`, |coef| `0.001510`
- `lag_08__bomb_events_last_5s`: coefficient `0.001508`, |coef| `0.001508`
- `lag_13__CT5__is_scoped`: coefficient `0.001488`, |coef| `0.001488`
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.001449`, |coef| `0.001449`
- `lag_07__CT_place_ELECTRICALBOX`: coefficient `-0.001434`, |coef| `0.001434`
- `lag_15__CT5__is_scoped`: coefficient `0.001419`, |coef| `0.001419`
- `lag_00__damage_diff_last_5s`: coefficient `0.001398`, |coef| `0.001398`
- `lag_00__CT_kills_last_3s`: coefficient `0.001369`, |coef| `0.001369`
- `lag_10__CT2__flash_duration`: coefficient `-0.001360`, |coef| `0.001360`

## Top 10 utility ridge features

- `lag_00__T_utility_damage_last_5s`: coefficient `-0.002285` (lowers CT win probability)
- `lag_01__T_utility_damage_last_5s`: coefficient `-0.001985` (lowers CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.001449` (raises CT win probability)
- `lag_10__CT2__flash_duration`: coefficient `-0.001360` (lowers CT win probability)
- `lag_01__utility_damage_diff_last_5s`: coefficient `0.001274` (raises CT win probability)
- `lag_15__CT2__flash_duration`: coefficient `0.001261` (raises CT win probability)
- `lag_15__CT_flash_duration_sum`: coefficient `0.001168` (raises CT win probability)
- `lag_02__T_utility_damage_last_5s`: coefficient `-0.001137` (lowers CT win probability)
- `lag_09__T_A_site_active_infernos`: coefficient `-0.001029` (lowers CT win probability)
- `lag_13__CT_B_site_active_infernos`: coefficient `-0.000894` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_TMAIN`: coefficient `-0.002615` (lowers CT win probability)
- `lag_08__T_shots_fired_sum`: coefficient `-0.002570` (lowers CT win probability)
- `lag_08__T4__shots_fired`: coefficient `-0.002062` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001549` (raises CT win probability)
- `lag_09__CT_place_ELECTRICALBOX`: coefficient `0.001510` (raises CT win probability)
- `lag_08__bomb_events_last_5s`: coefficient `0.001508` (raises CT win probability)
- `lag_13__CT5__is_scoped`: coefficient `0.001488` (raises CT win probability)
- `lag_07__CT_place_ELECTRICALBOX`: coefficient `-0.001434` (lowers CT win probability)
- `lag_15__CT5__is_scoped`: coefficient `0.001419` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001398` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `71643`, seconds `89.00`, LSTM delta `+0.2941`

Top all feature movements:
- `lag_08__T_shots_fired_sum`: contribution `+0.032753`
- `lag_08__T4__shots_fired`: contribution `+0.021651`
- `lag_09__CT_place_ELECTRICALBOX`: contribution `+0.017557`
- `lag_07__CT_place_ELECTRICALBOX`: contribution `+0.016665`
- `lag_14__CT_place_TMAIN`: contribution `+0.014178`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `71227`, seconds `82.50`, LSTM delta `+0.2685`

Top all feature movements:
- `lag_01__CT_place_TMAIN`: contribution `+0.028974`
- `lag_00__CT_shots_fired_sum`: contribution `+0.005382`
- `lag_01__CT5__duck_amount`: contribution `+0.004056`
- `lag_02__CT5__is_scoped`: contribution `+0.003988`
- `lag_15__CT2__duck_amount`: contribution `+0.003911`

Top utility-only movements:
- `lag_13__CT_B_site_active_infernos`: contribution `+0.003072`
- `lag_09__T_A_site_active_infernos`: contribution `+0.003064`
- `lag_13__CT_A_site_active_infernos`: contribution `+0.002949`
- `lag_10__CT2__flash_duration`: contribution `+0.002837`

### tick `71291`, seconds `83.50`, LSTM delta `+0.1376`

Top all feature movements:
- `lag_03__CT_place_TMAIN`: contribution `+0.008822`
- `lag_00__CT_shots_fired_sum`: contribution `+0.005382`
- `lag_02__CT_shots_fired_sum`: contribution `+0.002813`
- `lag_04__CT5__is_scoped`: contribution `+0.002746`
- `lag_00__damage_diff_last_5s`: contribution `+0.002712`

Top utility-only movements:
- `lag_03__T_B_site_active_infernos`: contribution `+0.001837`

### tick `71195`, seconds `82.00`, LSTM delta `-0.1042`

Top all feature movements:
- `lag_15__CT2__flash_duration`: contribution `-0.005824`
- `lag_04__T2__is_scoped`: contribution `-0.005357`
- `lag_00__CT_place_TMAIN`: contribution `-0.004731`
- `lag_00__CT_shots_fired_sum`: contribution `+0.004305`
- `lag_15__CT_flash_duration_sum`: contribution `-0.004226`

Top utility-only movements:
- `lag_15__CT2__flash_duration`: contribution `-0.005824`
- `lag_15__CT_flash_duration_sum`: contribution `-0.004226`
- `lag_15__CT5__flash_duration`: contribution `-0.002436`

### tick `71035`, seconds `79.50`, LSTM delta `+0.0910`

Top all feature movements:
- `lag_10__CT2__flash_duration`: contribution `+0.006280`
- `lag_00__CT_kills_last_3s`: contribution `+0.003952`
- `lag_00__kill_diff_last_3s`: contribution `+0.003222`
- `lag_08__bomb_events_last_5s`: contribution `+0.003150`
- `lag_00__damage_diff_last_5s`: contribution `+0.003090`

Top utility-only movements:
- `lag_10__CT2__flash_duration`: contribution `+0.006280`
- `lag_10__CT_flash_duration_sum`: contribution `+0.002694`
- `lag_03__T_B_site_active_infernos`: contribution `+0.001837`
- `lag_10__CT5__flash_duration`: contribution `+0.001701`
