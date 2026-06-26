# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-tyloo-vs-falcons-bo3-MBKGKnSCeuy54EHzS5mmW8/tyloo-vs-falcons-m2-ancient.csv`
- round_num: `4`

## Largest probability jumps

- tick `25679`, seconds `48.50`, LSTM `0.3733`, delta `+0.3031`
- tick `25359`, seconds `43.50`, LSTM `0.1432`, delta `-0.2920`
- tick `25615`, seconds `47.50`, LSTM `0.1242`, delta `-0.2718`
- tick `25135`, seconds `40.00`, LSTM `0.2811`, delta `+0.2418`
- tick `23951`, seconds `21.50`, LSTM `0.2379`, delta `-0.2239`
- tick `25487`, seconds `45.50`, LSTM `0.3051`, delta `+0.1902`
- tick `24527`, seconds `30.50`, LSTM `0.1776`, delta `-0.1821`
- tick `24239`, seconds `26.00`, LSTM `0.4025`, delta `+0.1788`
- tick `25167`, seconds `40.50`, LSTM `0.3702`, delta `+0.0891`
- tick `24431`, seconds `29.00`, LSTM `0.3821`, delta `-0.0670`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005608`, |coef| `0.005608`
- `lag_00__damage_diff_last_5s`: coefficient `0.004612`, |coef| `0.004612`
- `lag_00__T_kills_last_3s`: coefficient `-0.004098`, |coef| `0.004098`
- `lag_00__T_shots_fired_sum`: coefficient `-0.003844`, |coef| `0.003844`
- `lag_08__T_shots_fired_sum`: coefficient `-0.003702`, |coef| `0.003702`
- `lag_00__T_damage_last_5s`: coefficient `-0.003665`, |coef| `0.003665`
- `lag_02__CT_duck_amount_mean`: coefficient `0.003313`, |coef| `0.003313`
- `lag_02__CT3__duck_amount`: coefficient `0.003046`, |coef| `0.003046`
- `lag_00__CT_kills_last_3s`: coefficient `0.002992`, |coef| `0.002992`
- `lag_07__T_place_SIDEENTRANCE`: coefficient `0.002896`, |coef| `0.002896`
- `lag_01__kill_diff_last_3s`: coefficient `0.002679`, |coef| `0.002679`
- `lag_08__T_place_ALLEY`: coefficient `-0.002566`, |coef| `0.002566`
- `lag_15__T5__shots_fired`: coefficient `0.002459`, |coef| `0.002459`
- `lag_01__CT_kills_last_3s`: coefficient `0.002409`, |coef| `0.002409`
- `lag_11__CT3__is_walking`: coefficient `0.002352`, |coef| `0.002352`

## Top 10 utility ridge features

- `lag_09__T_utility_damage_last_5s`: coefficient `-0.002292` (lowers CT win probability)
- `lag_04__T_B_site_active_infernos`: coefficient `-0.002176` (lowers CT win probability)
- `lag_15__T_B_site_active_infernos`: coefficient `-0.002100` (lowers CT win probability)
- `lag_10__CT5__flash_duration`: coefficient `-0.001876` (lowers CT win probability)
- `lag_05__T_B_site_active_infernos`: coefficient `-0.001807` (lowers CT win probability)
- `lag_08__T_B_site_active_infernos`: coefficient `0.001758` (raises CT win probability)
- `lag_04__T_active_infernos`: coefficient `-0.001596` (lowers CT win probability)
- `lag_15__T_active_infernos`: coefficient `-0.001539` (lowers CT win probability)
- `lag_09__T1__flash_duration`: coefficient `0.001497` (raises CT win probability)
- `lag_09__T_B_site_active_infernos`: coefficient `0.001485` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005608` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.004612` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.004098` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.003844` (lowers CT win probability)
- `lag_08__T_shots_fired_sum`: coefficient `-0.003702` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.003665` (lowers CT win probability)
- `lag_02__CT_duck_amount_mean`: coefficient `0.003313` (raises CT win probability)
- `lag_02__CT3__duck_amount`: coefficient `0.003046` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002992` (raises CT win probability)
- `lag_07__T_place_SIDEENTRANCE`: coefficient `0.002896` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `25679`, seconds `48.50`, LSTM delta `+0.3031`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.020176`
- `lag_02__CT_duck_amount_mean`: contribution `+0.019837`
- `lag_00__damage_diff_last_5s`: contribution `+0.015814`
- `lag_02__CT3__duck_amount`: contribution `+0.011332`
- `lag_09__T4__shots_fired`: contribution `+0.009245`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `25359`, seconds `43.50`, LSTM delta `-0.2920`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.017294`
- `lag_07__T_place_SIDEENTRANCE`: contribution `-0.014134`
- `lag_00__kill_diff_last_3s`: contribution `-0.013497`
- `lag_00__T_kills_last_3s`: contribution `-0.012983`
- `lag_09__T_place_SIDEENTRANCE`: contribution `-0.011265`

Top utility-only movements:
- `lag_15__T_B_site_active_infernos`: contribution `-0.005936`

### tick `25615`, seconds `47.50`, LSTM delta `-0.2718`

Top all feature movements:
- `lag_08__T_shots_fired_sum`: contribution `-0.016654`
- `lag_00__kill_diff_last_3s`: contribution `-0.013497`
- `lag_00__T_kills_last_3s`: contribution `-0.012983`
- `lag_02__CT3__duck_amount`: contribution `-0.011332`
- `lag_00__damage_diff_last_5s`: contribution `-0.010092`

Top utility-only movements:
- `lag_09__T_B_site_active_infernos`: contribution `-0.004198`

### tick `25135`, seconds `40.00`, LSTM delta `+0.2418`

Top all feature movements:
- `lag_08__T_shots_fired_sum`: contribution `+0.016654`
- `lag_00__kill_diff_last_3s`: contribution `+0.013497`
- `lag_00__T_place_SIDEENTRANCE`: contribution `+0.011314`
- `lag_08__T_place_ALLEY`: contribution `+0.010870`
- `lag_00__damage_diff_last_5s`: contribution `+0.010404`

Top utility-only movements:
- `lag_09__T_utility_damage_last_5s`: contribution `+0.008834`
- `lag_04__T_B_site_active_infernos`: contribution `+0.006153`
- `lag_08__T_B_site_active_infernos`: contribution `+0.004969`
- `lag_09__utility_damage_diff_last_5s`: contribution `+0.003616`

### tick `23951`, seconds `21.50`, LSTM delta `-0.2239`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.017294`
- `lag_00__kill_diff_last_3s`: contribution `-0.013497`
- `lag_00__T_kills_last_3s`: contribution `-0.012983`
- `lag_10__CT5__flash_duration`: contribution `-0.011483`
- `lag_01__T1__flash_duration`: contribution `-0.009736`

Top utility-only movements:
- `lag_10__CT5__flash_duration`: contribution `-0.011483`
- `lag_01__T1__flash_duration`: contribution `-0.009736`
- `lag_04__CT4__flash_duration`: contribution `-0.008448`
- `lag_07__T3__flash_duration`: contribution `-0.005861`
- `lag_05__T_B_site_active_infernos`: contribution `-0.005109`
