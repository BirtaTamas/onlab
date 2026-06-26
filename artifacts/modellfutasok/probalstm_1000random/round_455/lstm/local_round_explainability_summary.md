# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-big-vs-pain-bo3-So89pkF9idYLRaqhIPbo1H/big-vs-pain-m3-inferno-p3.csv`
- round_num: `10`

## Largest probability jumps

- tick `101447`, seconds `77.50`, LSTM `0.7583`, delta `+0.2455`
- tick `98791`, seconds `36.00`, LSTM `0.2999`, delta `-0.1691`
- tick `99367`, seconds `45.00`, LSTM `0.3206`, delta `+0.1370`
- tick `101575`, seconds `79.50`, LSTM `0.9017`, delta `+0.1318`
- tick `100999`, seconds `70.50`, LSTM `0.5329`, delta `-0.1247`
- tick `99591`, seconds `48.50`, LSTM `0.5800`, delta `+0.1185`
- tick `99655`, seconds `49.50`, LSTM `0.4741`, delta `-0.1157`
- tick `100807`, seconds `67.50`, LSTM `0.5918`, delta `+0.1129`
- tick `100775`, seconds `67.00`, LSTM `0.4789`, delta `+0.1067`
- tick `100935`, seconds `69.50`, LSTM `0.6741`, delta `+0.0803`

## Top 15 local ridge features

- `lag_14__CT2__is_scoped`: coefficient `0.003389`, |coef| `0.003389`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.003367`, |coef| `0.003367`
- `lag_00__CT_defusing_count`: coefficient `0.003160`, |coef| `0.003160`
- `lag_00__kill_diff_last_3s`: coefficient `0.002751`, |coef| `0.002751`
- `lag_05__T3__flash_duration`: coefficient `-0.002247`, |coef| `0.002247`
- `lag_12__CT_place_RUINS`: coefficient `-0.002229`, |coef| `0.002229`
- `lag_00__CT_kills_last_3s`: coefficient `0.002110`, |coef| `0.002110`
- `lag_00__damage_diff_last_5s`: coefficient `0.002051`, |coef| `0.002051`
- `lag_04__T_flash_alpha_mean`: coefficient `-0.001972`, |coef| `0.001972`
- `lag_00__CT_damage_last_5s`: coefficient `0.001778`, |coef| `0.001778`
- `lag_15__CT2__duck_amount`: coefficient `-0.001777`, |coef| `0.001777`
- `lag_11__CT_place_RUINS`: coefficient `-0.001732`, |coef| `0.001732`
- `lag_00__CT4__duck_amount`: coefficient `0.001714`, |coef| `0.001714`
- `lag_14__kill_diff_last_3s`: coefficient `-0.001664`, |coef| `0.001664`
- `lag_00__CT_flashed_players`: coefficient `0.001588`, |coef| `0.001588`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.003367` (lowers CT win probability)
- `lag_05__T3__flash_duration`: coefficient `-0.002247` (lowers CT win probability)
- `lag_04__T_flash_alpha_mean`: coefficient `-0.001972` (lowers CT win probability)
- `lag_09__CT1__flash_duration`: coefficient `0.001417` (raises CT win probability)
- `lag_05__T_flash_duration_sum`: coefficient `-0.001274` (lowers CT win probability)
- `lag_13__T5__flash_duration`: coefficient `-0.001265` (lowers CT win probability)
- `lag_12__T3__flash_duration`: coefficient `-0.001206` (lowers CT win probability)
- `lag_06__T3__flash_duration`: coefficient `0.001154` (raises CT win probability)
- `lag_05__T1__flash_duration`: coefficient `-0.001105` (lowers CT win probability)
- `lag_15__CT3__smoke`: coefficient `-0.001063` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_14__CT2__is_scoped`: coefficient `0.003389` (raises CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.003160` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002751` (raises CT win probability)
- `lag_12__CT_place_RUINS`: coefficient `-0.002229` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002110` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002051` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001778` (raises CT win probability)
- `lag_15__CT2__duck_amount`: coefficient `-0.001777` (lowers CT win probability)
- `lag_11__CT_place_RUINS`: coefficient `-0.001732` (lowers CT win probability)
- `lag_00__CT4__duck_amount`: coefficient `0.001714` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `101447`, seconds `77.50`, LSTM delta `+0.2455`

Top all feature movements:
- `lag_14__CT2__is_scoped`: contribution `+0.020741`
- `lag_00__T_flash_alpha_mean`: contribution `+0.020426`
- `lag_14__kill_diff_last_3s`: contribution `+0.008009`
- `lag_14__CT_flashed_players`: contribution `+0.006854`
- `lag_15__CT2__duck_amount`: contribution `+0.006768`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.020426`

### tick `98791`, seconds `36.00`, LSTM delta `-0.1691`

Top all feature movements:
- `lag_09__CT1__flash_duration`: contribution `-0.010676`
- `lag_13__T5__flash_duration`: contribution `-0.008218`
- `lag_05__T_flashed_players`: contribution `-0.007529`
- `lag_05__T1__flash_duration`: contribution `-0.007512`
- `lag_00__kill_diff_last_3s`: contribution `-0.006622`

Top utility-only movements:
- `lag_09__CT1__flash_duration`: contribution `-0.010676`
- `lag_13__T5__flash_duration`: contribution `-0.008218`
- `lag_05__T1__flash_duration`: contribution `-0.007512`
- `lag_05__T_flash_duration_sum`: contribution `-0.006458`
- `lag_01__CT5__flash_duration`: contribution `-0.006326`

### tick `99367`, seconds `45.00`, LSTM delta `+0.1370`

Top all feature movements:
- `lag_05__T3__flash_duration`: contribution `+0.014819`
- `lag_02__T_bomb_zone_count`: contribution `+0.006974`
- `lag_00__kill_diff_last_3s`: contribution `+0.006622`
- `lag_08__T4__flash_duration`: contribution `+0.006383`
- `lag_05__CT_place_LIBRARY`: contribution `+0.006324`

Top utility-only movements:
- `lag_05__T3__flash_duration`: contribution `+0.014819`
- `lag_08__T4__flash_duration`: contribution `+0.006383`
- `lag_07__T_utility_damage_last_5s`: contribution `+0.003751`
- `lag_05__T_flash_duration_sum`: contribution `+0.003501`
- `lag_09__T1__flash_duration`: contribution `+0.003337`

### tick `101575`, seconds `79.50`, LSTM delta `+0.1318`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `+0.030635`
- `lag_04__T_flash_alpha_mean`: contribution `+0.011963`
- `lag_12__CT_place_RUINS`: contribution `+0.007788`
- `lag_13__CT_shots_fired_sum`: contribution `+0.006188`
- `lag_03__CT4__duck_amount`: contribution `+0.005538`

Top utility-only movements:
- `lag_04__T_flash_alpha_mean`: contribution `+0.011963`

### tick `100999`, seconds `70.50`, LSTM delta `-0.1247`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.013244`
- `lag_12__CT_place_RUINS`: contribution `-0.007788`
- `lag_00__CT_flashed_players`: contribution `-0.006957`
- `lag_06__T3__flash_duration`: contribution `-0.006698`
- `lag_02__T_duck_amount_mean`: contribution `-0.006100`

Top utility-only movements:
- `lag_06__T3__flash_duration`: contribution `-0.006698`
