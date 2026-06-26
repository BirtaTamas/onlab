# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-furia-vs-b8-bo3-3h93b_qbGndTgDFTW66Ud1/furia-vs-b8-m1-mirage.csv`
- round_num: `13`

## Largest probability jumps

- tick `120025`, seconds `52.50`, LSTM `0.2074`, delta `-0.2760`
- tick `120249`, seconds `56.00`, LSTM `0.2780`, delta `+0.2022`
- tick `119929`, seconds `51.00`, LSTM `0.5234`, delta `-0.1931`
- tick `119065`, seconds `37.50`, LSTM `0.3882`, delta `+0.1660`
- tick `120537`, seconds `60.50`, LSTM `0.5189`, delta `+0.1415`
- tick `119705`, seconds `47.50`, LSTM `0.5987`, delta `+0.1026`
- tick `120089`, seconds `53.50`, LSTM `0.0689`, delta `-0.0934`
- tick `120345`, seconds `57.50`, LSTM `0.3545`, delta `+0.0851`
- tick `118585`, seconds `30.00`, LSTM `0.4238`, delta `-0.0800`
- tick `119737`, seconds `48.00`, LSTM `0.6602`, delta `+0.0615`

## Top 15 local ridge features

- `lag_00__damage_diff_last_5s`: coefficient `0.003856`, |coef| `0.003856`
- `lag_00__kill_diff_last_3s`: coefficient `0.003288`, |coef| `0.003288`
- `lag_00__T_kills_last_3s`: coefficient `-0.002719`, |coef| `0.002719`
- `lag_15__CT4__flash_duration`: coefficient `0.002709`, |coef| `0.002709`
- `lag_00__T_damage_last_5s`: coefficient `-0.002645`, |coef| `0.002645`
- `lag_00__CT_defusing_count`: coefficient `0.002603`, |coef| `0.002603`
- `lag_09__CT_place_STAIRS`: coefficient `-0.002369`, |coef| `0.002369`
- `lag_01__kill_diff_last_3s`: coefficient `0.002165`, |coef| `0.002165`
- `lag_08__CT_place_JUNGLE`: coefficient `0.002081`, |coef| `0.002081`
- `lag_01__T_kills_last_3s`: coefficient `-0.001960`, |coef| `0.001960`
- `lag_05__CT_place_STAIRS`: coefficient `-0.001919`, |coef| `0.001919`
- `lag_03__T_damage_last_5s`: coefficient `-0.001910`, |coef| `0.001910`
- `lag_05__CT_place_JUNGLE`: coefficient `0.001850`, |coef| `0.001850`
- `lag_10__CT_place_STAIRS`: coefficient `-0.001837`, |coef| `0.001837`
- `lag_09__CT_place_JUNGLE`: coefficient `0.001818`, |coef| `0.001818`

## Top 10 utility ridge features

- `lag_15__CT4__flash_duration`: coefficient `0.002709` (raises CT win probability)
- `lag_12__CT4__flash_duration`: coefficient `0.001729` (raises CT win probability)
- `lag_05__CT4__flash_duration`: coefficient `-0.001322` (lowers CT win probability)
- `lag_15__CT_flash_duration_sum`: coefficient `0.001059` (raises CT win probability)
- `lag_12__CT_flash_duration_sum`: coefficient `0.001039` (raises CT win probability)
- `lag_14__CT4__flash_duration`: coefficient `0.000972` (raises CT win probability)
- `lag_13__CT4__flash_duration`: coefficient `0.000861` (raises CT win probability)
- `lag_04__CT4__flash_duration`: coefficient `-0.000754` (lowers CT win probability)
- `lag_09__CT4__flash_duration`: coefficient `0.000720` (raises CT win probability)
- `lag_09__T_A_site_active_smokes`: coefficient `-0.000706` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__damage_diff_last_5s`: coefficient `0.003856` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003288` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002719` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002645` (lowers CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.002603` (raises CT win probability)
- `lag_09__CT_place_STAIRS`: coefficient `-0.002369` (lowers CT win probability)
- `lag_01__kill_diff_last_3s`: coefficient `0.002165` (raises CT win probability)
- `lag_08__CT_place_JUNGLE`: coefficient `0.002081` (raises CT win probability)
- `lag_01__T_kills_last_3s`: coefficient `-0.001960` (lowers CT win probability)
- `lag_05__CT_place_STAIRS`: coefficient `-0.001919` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `120025`, seconds `52.50`, LSTM delta `-0.2760`

Top all feature movements:
- `lag_15__CT4__flash_duration`: contribution `-0.020147`
- `lag_09__CT_place_STAIRS`: contribution `-0.018434`
- `lag_00__damage_diff_last_5s`: contribution `-0.016527`
- `lag_05__CT_place_STAIRS`: contribution `-0.014939`
- `lag_08__CT_place_JUNGLE`: contribution `-0.013348`

Top utility-only movements:
- `lag_15__CT4__flash_duration`: contribution `-0.020147`
- `lag_15__CT_flash_duration_sum`: contribution `-0.003522`

### tick `120249`, seconds `56.00`, LSTM delta `+0.2022`

Top all feature movements:
- `lag_00__damage_diff_last_5s`: contribution `+0.023051`
- `lag_02__CT_place_STAIRS`: contribution `+0.011050`
- `lag_00__T_damage_last_5s`: contribution `+0.010465`
- `lag_00__kill_diff_last_3s`: contribution `+0.007914`
- `lag_01__T_kills_last_3s`: contribution `+0.006209`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `119929`, seconds `51.00`, LSTM delta `-0.1931`

Top all feature movements:
- `lag_00__damage_diff_last_5s`: contribution `-0.015222`
- `lag_12__CT4__flash_duration`: contribution `-0.012861`
- `lag_05__CT_place_JUNGLE`: contribution `-0.011867`
- `lag_00__T_damage_last_5s`: contribution `-0.011099`
- `lag_02__CT_place_STAIRS`: contribution `-0.011050`

Top utility-only movements:
- `lag_12__CT4__flash_duration`: contribution `-0.012861`
- `lag_12__CT_flash_duration_sum`: contribution `-0.003457`

### tick `119065`, seconds `37.50`, LSTM delta `+0.1660`

Top all feature movements:
- `lag_13__T_place_SCAFFOLDING`: contribution `+0.040412`
- `lag_15__T_place_SCAFFOLDING`: contribution `+0.031945`
- `lag_14__T_place_SCAFFOLDING`: contribution `+0.031711`
- `lag_00__kill_diff_last_3s`: contribution `+0.007914`
- `lag_13__CT_place_LADDER`: contribution `+0.006136`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `120537`, seconds `60.50`, LSTM delta `+0.1415`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `+0.025232`
- `lag_11__CT_place_STAIRS`: contribution `+0.012614`
- `lag_06__CT_velocity_mean`: contribution `+0.006426`
- `lag_09__damage_diff_last_5s`: contribution `+0.006231`
- `lag_09__T_place_CTSPAWN`: contribution `+0.004786`

Top utility-only movements:
- No utility movement among the top local contributors.
