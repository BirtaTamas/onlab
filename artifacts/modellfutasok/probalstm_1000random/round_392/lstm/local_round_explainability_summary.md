# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-m80-vs-rooster-bo3-GFAv4Fg83aXYKbsY0nLkP_/m80-vs-rooster-m2-inferno.csv`
- round_num: `1`

## Largest probability jumps

- tick `4254`, seconds `47.00`, LSTM `0.5631`, delta `-0.2013`
- tick `4414`, seconds `49.50`, LSTM `0.7630`, delta `+0.1780`
- tick `3614`, seconds `37.00`, LSTM `0.7893`, delta `+0.1767`
- tick `3134`, seconds `29.50`, LSTM `0.5792`, delta `+0.1748`
- tick `4062`, seconds `44.00`, LSTM `0.6973`, delta `-0.1407`
- tick `3870`, seconds `41.00`, LSTM `0.9198`, delta `+0.1368`
- tick `3966`, seconds `42.50`, LSTM `0.8240`, delta `-0.1161`
- tick `4606`, seconds `52.50`, LSTM `0.9122`, delta `+0.0947`
- tick `3102`, seconds `29.00`, LSTM `0.4045`, delta `-0.0762`
- tick `4126`, seconds `45.00`, LSTM `0.7593`, delta `+0.0682`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003710`, |coef| `0.003710`
- `lag_00__damage_diff_last_5s`: coefficient `0.003323`, |coef| `0.003323`
- `lag_12__T_place_GRAVEYARD`: coefficient `-0.003127`, |coef| `0.003127`
- `lag_00__CT_kills_last_3s`: coefficient `0.002900`, |coef| `0.002900`
- `lag_05__CT_place_QUAD`: coefficient `-0.002393`, |coef| `0.002393`
- `lag_05__CT_place_TOPOFMID`: coefficient `0.002354`, |coef| `0.002354`
- `lag_00__CT_defusing_count`: coefficient `0.002074`, |coef| `0.002074`
- `lag_03__CT_place_QUAD`: coefficient `0.001976`, |coef| `0.001976`
- `lag_08__CT_kills_last_3s`: coefficient `0.001845`, |coef| `0.001845`
- `lag_08__kill_diff_last_3s`: coefficient `0.001840`, |coef| `0.001840`
- `lag_02__CT_place_BALCONY`: coefficient `-0.001802`, |coef| `0.001802`
- `lag_00__T_damage_last_5s`: coefficient `-0.001801`, |coef| `0.001801`
- `lag_06__T1__is_walking`: coefficient `0.001780`, |coef| `0.001780`
- `lag_10__T_place_ARCH`: coefficient `-0.001712`, |coef| `0.001712`
- `lag_00__T_kills_last_3s`: coefficient `-0.001702`, |coef| `0.001702`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.001700` (lowers CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `-0.001479` (lowers CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `-0.001383` (lowers CT win probability)
- `lag_04__CT_utility_damage_last_5s`: coefficient `-0.001286` (lowers CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `-0.001283` (lowers CT win probability)
- `lag_06__T_flash_alpha_mean`: coefficient `-0.001280` (lowers CT win probability)
- `lag_04__utility_damage_diff_last_5s`: coefficient `-0.001051` (lowers CT win probability)
- `lag_05__CT4__flash_duration`: coefficient `0.001041` (raises CT win probability)
- `lag_09__CT_utility_damage_last_5s`: coefficient `0.001029` (raises CT win probability)
- `lag_05__CT1__flash_duration`: coefficient `0.000972` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003710` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003323` (raises CT win probability)
- `lag_12__T_place_GRAVEYARD`: coefficient `-0.003127` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002900` (raises CT win probability)
- `lag_05__CT_place_QUAD`: coefficient `-0.002393` (lowers CT win probability)
- `lag_05__CT_place_TOPOFMID`: coefficient `0.002354` (raises CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.002074` (raises CT win probability)
- `lag_03__CT_place_QUAD`: coefficient `0.001976` (raises CT win probability)
- `lag_08__CT_kills_last_3s`: coefficient `0.001845` (raises CT win probability)
- `lag_08__kill_diff_last_3s`: coefficient `0.001840` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `4254`, seconds `47.00`, LSTM delta `-0.2013`

Top all feature movements:
- `lag_09__CT_place_QUAD`: contribution `-0.012916`
- `lag_11__CT_place_QUAD`: contribution `-0.011848`
- `lag_02__CT_place_BALCONY`: contribution `-0.011566`
- `lag_00__kill_diff_last_3s`: contribution `-0.008931`
- `lag_05__CT_place_TOPOFMID`: contribution `-0.008543`

Top utility-only movements:
- `lag_00__CT4__flash_duration`: contribution `-0.007533`
- `lag_00__CT1__flash_duration`: contribution `-0.006579`
- `lag_04__CT_utility_damage_last_5s`: contribution `-0.006369`
- `lag_00__CT_flash_duration_sum`: contribution `-0.005663`
- `lag_04__utility_damage_diff_last_5s`: contribution `-0.004269`

### tick `4414`, seconds `49.50`, LSTM delta `+0.1780`

Top all feature movements:
- `lag_14__CT_place_QUAD`: contribution `+0.010391`
- `lag_00__T_flash_alpha_mean`: contribution `+0.010317`
- `lag_00__T_place_PIT`: contribution `+0.009144`
- `lag_00__kill_diff_last_3s`: contribution `+0.008931`
- `lag_07__CT_place_BALCONY`: contribution `+0.008874`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.010317`
- `lag_05__CT4__flash_duration`: contribution `+0.005302`
- `lag_09__CT_utility_damage_last_5s`: contribution `+0.005096`
- `lag_05__CT1__flash_duration`: contribution `+0.004627`
- `lag_05__CT_flash_duration_sum`: contribution `+0.003959`

### tick `3614`, seconds `37.00`, LSTM delta `+0.1767`

Top all feature movements:
- `lag_12__T_place_GRAVEYARD`: contribution `+0.061461`
- `lag_00__kill_diff_last_3s`: contribution `+0.008931`
- `lag_00__CT_kills_last_3s`: contribution `+0.008372`
- `lag_09__T_bomb_zone_count`: contribution `+0.007981`
- `lag_12__T_place_PIT`: contribution `+0.007020`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `3134`, seconds `29.50`, LSTM delta `+0.1748`

Top all feature movements:
- `lag_03__T_place_GRAVEYARD`: contribution `+0.033182`
- `lag_10__T_place_ARCH`: contribution `+0.031863`
- `lag_11__T_place_ARCH`: contribution `+0.010985`
- `lag_12__T_place_ARCH`: contribution `+0.009024`
- `lag_00__kill_diff_last_3s`: contribution `+0.008931`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `4062`, seconds `44.00`, LSTM delta `-0.1407`

Top all feature movements:
- `lag_05__CT_place_QUAD`: contribution `-0.018860`
- `lag_03__CT_place_QUAD`: contribution `-0.015574`
- `lag_00__kill_diff_last_3s`: contribution `-0.008931`
- `lag_00__CT_kills_last_3s`: contribution `-0.008372`
- `lag_00__damage_diff_last_5s`: contribution `-0.006972`

Top utility-only movements:
- No utility movement among the top local contributors.
