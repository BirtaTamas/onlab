# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-betboom-vs-legacy-anubis-nLMamLTYoRhlv2MuS6sSiC/betboom-vs-legacy-anubis.csv`
- round_num: `7`

## Largest probability jumps

- tick `65164`, seconds `29.00`, LSTM `0.1707`, delta `-0.1731`
- tick `68140`, seconds `75.50`, LSTM `0.1065`, delta `-0.0438`
- tick `65196`, seconds `29.50`, LSTM `0.1281`, delta `-0.0426`
- tick `67564`, seconds `66.50`, LSTM `0.1577`, delta `-0.0403`
- tick `65932`, seconds `41.00`, LSTM `0.1946`, delta `+0.0329`
- tick `65804`, seconds `39.00`, LSTM `0.1241`, delta `+0.0306`
- tick `68268`, seconds `77.50`, LSTM `0.0832`, delta `-0.0255`
- tick `66572`, seconds `51.00`, LSTM `0.1530`, delta `-0.0253`
- tick `66668`, seconds `52.50`, LSTM `0.1284`, delta `-0.0251`
- tick `63756`, seconds `7.00`, LSTM `0.4436`, delta `+0.0239`

## Top 15 local ridge features

- `lag_01__T5__flash_duration`: coefficient `-0.001965`, |coef| `0.001965`
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `0.001525`, |coef| `0.001525`
- `lag_07__T_place_STREET`: coefficient `0.001149`, |coef| `0.001149`
- `lag_00__CT_place_MAIN`: coefficient `-0.001128`, |coef| `0.001128`
- `lag_02__T5__flash_duration`: coefficient `-0.001103`, |coef| `0.001103`
- `lag_09__T5__duck_amount`: coefficient `-0.001096`, |coef| `0.001096`
- `lag_15__CT_place_FOUNTAIN`: coefficient `-0.001075`, |coef| `0.001075`
- `lag_00__CT1__alive`: coefficient `0.001074`, |coef| `0.001074`
- `lag_00__CT1__hp`: coefficient `0.001059`, |coef| `0.001059`
- `lag_14__CT_place_FOUNTAIN`: coefficient `-0.001033`, |coef| `0.001033`
- `lag_07__CT1__duck_amount`: coefficient `-0.001010`, |coef| `0.001010`
- `lag_00__CT1__armor`: coefficient `0.000994`, |coef| `0.000994`
- `lag_00__CT1__smoke`: coefficient `0.000953`, |coef| `0.000953`
- `lag_00__T_kills_last_3s`: coefficient `-0.000937`, |coef| `0.000937`
- `lag_09__T1__duck_amount`: coefficient `0.000909`, |coef| `0.000909`

## Top 10 utility ridge features

- `lag_01__T5__flash_duration`: coefficient `-0.001965` (lowers CT win probability)
- `lag_02__T5__flash_duration`: coefficient `-0.001103` (lowers CT win probability)
- `lag_00__CT1__smoke`: coefficient `0.000953` (raises CT win probability)
- `lag_04__CT3__smoke`: coefficient `0.000872` (raises CT win probability)
- `lag_00__T5__flash_duration`: coefficient `-0.000865` (lowers CT win probability)
- `lag_00__CT_B_site_active_infernos`: coefficient `0.000736` (raises CT win probability)
- `lag_00__CT_active_infernos`: coefficient `0.000733` (raises CT win probability)
- `lag_01__T_flash_duration_sum`: coefficient `-0.000702` (lowers CT win probability)
- `lag_04__CT1__flash`: coefficient `0.000691` (raises CT win probability)
- `lag_06__CT_active_infernos`: coefficient `0.000672` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_CTSIDEUPPER`: coefficient `0.001525` (raises CT win probability)
- `lag_07__T_place_STREET`: coefficient `0.001149` (raises CT win probability)
- `lag_00__CT_place_MAIN`: coefficient `-0.001128` (lowers CT win probability)
- `lag_09__T5__duck_amount`: coefficient `-0.001096` (lowers CT win probability)
- `lag_15__CT_place_FOUNTAIN`: coefficient `-0.001075` (lowers CT win probability)
- `lag_00__CT1__alive`: coefficient `0.001074` (raises CT win probability)
- `lag_00__CT1__hp`: coefficient `0.001059` (raises CT win probability)
- `lag_14__CT_place_FOUNTAIN`: coefficient `-0.001033` (lowers CT win probability)
- `lag_07__CT1__duck_amount`: coefficient `-0.001010` (lowers CT win probability)
- `lag_00__CT1__armor`: coefficient `0.000994` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `65164`, seconds `29.00`, LSTM delta `-0.1731`

Top all feature movements:
- `lag_01__T5__flash_duration`: contribution `-0.012334`
- `lag_07__T_place_STREET`: contribution `-0.006317`
- `lag_09__T5__duck_amount`: contribution `-0.004163`
- `lag_00__T5__is_scoped`: contribution `-0.003942`
- `lag_07__CT1__duck_amount`: contribution `-0.003854`

Top utility-only movements:
- `lag_01__T5__flash_duration`: contribution `-0.012334`
- `lag_00__CT1__smoke`: contribution `-0.002066`
- `lag_01__T_flash_duration_sum`: contribution `-0.002018`

### tick `68140`, seconds `75.50`, LSTM delta `-0.0438`

Top all feature movements:
- `lag_12__CT_place_CANAL`: contribution `-0.004396`
- `lag_14__T_place_MIDDOORS`: contribution `-0.003647`
- `lag_03__T_flashed_players`: contribution `-0.003309`
- `lag_12__T_place_MIDDOORS`: contribution `-0.002787`
- `lag_15__T_place_BRIDGE`: contribution `-0.002637`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `65196`, seconds `29.50`, LSTM delta `-0.0426`

Top all feature movements:
- `lag_02__T5__flash_duration`: contribution `-0.006923`
- `lag_08__CT1__duck_amount`: contribution `-0.002138`
- `lag_10__T5__duck_amount`: contribution `-0.002136`
- `lag_08__T_place_STREET`: contribution `-0.002045`
- `lag_01__CT1__alive`: contribution `-0.001670`

Top utility-only movements:
- `lag_02__T5__flash_duration`: contribution `-0.006923`
- `lag_02__T_flash_duration_sum`: contribution `-0.001249`
- `lag_07__CT_active_infernos`: contribution `-0.001185`
- `lag_05__CT3__smoke`: contribution `-0.001172`

### tick `67564`, seconds `66.50`, LSTM delta `-0.0403`

Top all feature movements:
- `lag_05__T_place_MIDDOORS`: contribution `-0.002679`
- `lag_00__CT_B_site_active_infernos`: contribution `-0.002529`
- `lag_04__CT2__duck_amount`: contribution `-0.002079`
- `lag_03__T2__is_walking`: contribution `-0.001944`
- `lag_11__CT2__duck_amount`: contribution `-0.001871`

Top utility-only movements:
- `lag_00__CT_B_site_active_infernos`: contribution `-0.002529`
- `lag_00__CT_active_infernos`: contribution `-0.001689`
- `lag_11__CT_B_site_active_infernos`: contribution `-0.001121`

### tick `65932`, seconds `41.00`, LSTM delta `+0.0329`

Top all feature movements:
- `lag_14__T_place_MIDDOORS`: contribution `+0.003647`
- `lag_14__T5__flash_duration`: contribution `+0.002907`
- `lag_03__T2__is_walking`: contribution `-0.001944`
- `lag_04__T5__is_scoped`: contribution `+0.001874`
- `lag_06__CT4__is_walking`: contribution `-0.001509`

Top utility-only movements:
- `lag_14__T5__flash_duration`: contribution `+0.002907`
