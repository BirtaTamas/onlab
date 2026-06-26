# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-falcons-bo3-yayytstbo8IxTFlUpfbUPR/mouz-vs-falcons-m1-train.csv`
- round_num: `17`

## Largest probability jumps

- tick `149292`, seconds `118.00`, LSTM `0.6819`, delta `+0.2738`
- tick `149452`, seconds `120.50`, LSTM `0.8915`, delta `+0.2510`
- tick `147372`, seconds `88.00`, LSTM `0.8925`, delta `+0.2497`
- tick `149164`, seconds `116.00`, LSTM `0.4197`, delta `-0.1880`
- tick `146988`, seconds `82.00`, LSTM `0.6225`, delta `+0.1465`
- tick `147020`, seconds `82.50`, LSTM `0.7343`, delta `+0.1118`
- tick `147084`, seconds `83.50`, LSTM `0.6828`, delta `-0.1020`
- tick `149356`, seconds `119.00`, LSTM `0.6918`, delta `+0.0746`
- tick `148364`, seconds `103.50`, LSTM `0.7864`, delta `-0.0652`
- tick `149324`, seconds `118.50`, LSTM `0.6172`, delta `-0.0648`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003254`, |coef| `0.003254`
- `lag_07__T_shots_fired_sum`: coefficient `-0.002792`, |coef| `0.002792`
- `lag_00__CT_kills_last_3s`: coefficient `0.002677`, |coef| `0.002677`
- `lag_09__T_utility_damage_last_5s`: coefficient `-0.002164`, |coef| `0.002164`
- `lag_10__T_utility_damage_last_5s`: coefficient `-0.002109`, |coef| `0.002109`
- `lag_00__damage_diff_last_5s`: coefficient `0.002030`, |coef| `0.002030`
- `lag_00__CT_defusing_count`: coefficient `0.001940`, |coef| `0.001940`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001922`, |coef| `0.001922`
- `lag_04__T_utility_damage_last_5s`: coefficient `-0.001892`, |coef| `0.001892`
- `lag_05__T_shots_fired_sum`: coefficient `-0.001844`, |coef| `0.001844`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.001598`, |coef| `0.001598`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001532`, |coef| `0.001532`
- `lag_07__T1__shots_fired`: coefficient `-0.001529`, |coef| `0.001529`
- `lag_14__T_utility_damage_last_5s`: coefficient `0.001426`, |coef| `0.001426`
- `lag_09__utility_damage_diff_last_5s`: coefficient `0.001419`, |coef| `0.001419`

## Top 10 utility ridge features

- `lag_09__T_utility_damage_last_5s`: coefficient `-0.002164` (lowers CT win probability)
- `lag_10__T_utility_damage_last_5s`: coefficient `-0.002109` (lowers CT win probability)
- `lag_04__T_utility_damage_last_5s`: coefficient `-0.001892` (lowers CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.001598` (lowers CT win probability)
- `lag_14__T_utility_damage_last_5s`: coefficient `0.001426` (raises CT win probability)
- `lag_09__utility_damage_diff_last_5s`: coefficient `0.001419` (raises CT win probability)
- `lag_10__utility_damage_diff_last_5s`: coefficient `0.001354` (raises CT win probability)
- `lag_04__utility_damage_diff_last_5s`: coefficient `0.001202` (raises CT win probability)
- `lag_05__T_utility_damage_last_5s`: coefficient `-0.001193` (lowers CT win probability)
- `lag_01__CT4__flash_duration`: coefficient `0.000957` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003254` (raises CT win probability)
- `lag_07__T_shots_fired_sum`: coefficient `-0.002792` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002677` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002030` (raises CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.001940` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001922` (raises CT win probability)
- `lag_05__T_shots_fired_sum`: coefficient `-0.001844` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001532` (lowers CT win probability)
- `lag_07__T1__shots_fired`: coefficient `-0.001529` (lowers CT win probability)
- `lag_05__T5__duck_amount`: coefficient `0.001380` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `149292`, seconds `118.00`, LSTM delta `+0.2738`

Top all feature movements:
- `lag_07__T_shots_fired_sum`: contribution `+0.023029`
- `lag_04__T_utility_damage_last_5s`: contribution `+0.013507`
- `lag_15__CT_place_ENTRANCE`: contribution `+0.011326`
- `lag_14__T_utility_damage_last_5s`: contribution `+0.010177`
- `lag_00__T_shots_fired_sum`: contribution `+0.009186`

Top utility-only movements:
- `lag_04__T_utility_damage_last_5s`: contribution `+0.013507`
- `lag_14__T_utility_damage_last_5s`: contribution `+0.010177`
- `lag_04__utility_damage_diff_last_5s`: contribution `+0.005427`
- `lag_14__utility_damage_diff_last_5s`: contribution `+0.003419`

### tick `149452`, seconds `120.50`, LSTM delta `+0.2510`

Top all feature movements:
- `lag_09__T_utility_damage_last_5s`: contribution `+0.015450`
- `lag_05__T_shots_fired_sum`: contribution `+0.011060`
- `lag_04__CT_shots_fired_sum`: contribution `+0.010403`
- `lag_00__T_flash_alpha_mean`: contribution `+0.009698`
- `lag_00__kill_diff_last_3s`: contribution `+0.007833`

Top utility-only movements:
- `lag_09__T_utility_damage_last_5s`: contribution `+0.015450`
- `lag_00__T_flash_alpha_mean`: contribution `+0.009698`
- `lag_09__utility_damage_diff_last_5s`: contribution `+0.006404`

### tick `147372`, seconds `88.00`, LSTM delta `+0.2497`

Top all feature movements:
- `lag_07__T_shots_fired_sum`: contribution `+0.031403`
- `lag_07__T1__shots_fired`: contribution `+0.013707`
- `lag_12__T_place_DUMPSTER`: contribution `+0.011766`
- `lag_00__kill_diff_last_3s`: contribution `+0.007833`
- `lag_00__CT_kills_last_3s`: contribution `+0.007728`

Top utility-only movements:
- `lag_14__CT4__flash_duration`: contribution `+0.003988`

### tick `149164`, seconds `116.00`, LSTM delta `-0.1880`

Top all feature movements:
- `lag_10__T_utility_damage_last_5s`: contribution `-0.015056`
- `lag_00__kill_diff_last_3s`: contribution `-0.007833`
- `lag_03__T5__shots_fired`: contribution `-0.007180`
- `lag_05__T_shots_fired_sum`: contribution `-0.006913`
- `lag_03__T_shots_fired_sum`: contribution `-0.006803`

Top utility-only movements:
- `lag_10__T_utility_damage_last_5s`: contribution `-0.015056`
- `lag_10__utility_damage_diff_last_5s`: contribution `-0.006111`
- `lag_00__T_utility_damage_last_5s`: contribution `-0.005530`

### tick `146988`, seconds `82.00`, LSTM delta `+0.1465`

Top all feature movements:
- `lag_04__CT_place_IVY`: contribution `+0.013677`
- `lag_00__T_place_DUMPSTER`: contribution `+0.009924`
- `lag_00__kill_diff_last_3s`: contribution `+0.007833`
- `lag_00__CT_kills_last_3s`: contribution `+0.007728`
- `lag_13__CT1__duck_amount`: contribution `+0.004410`

Top utility-only movements:
- `lag_02__CT4__flash_duration`: contribution `+0.002081`
