# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-spirit-vs-the-huns-bo3-TWIJIxJZifB3vPv3OUvjVr/spirit-vs-the-huns-m2-dust2.csv`
- round_num: `17`

## Largest probability jumps

- tick `148247`, seconds `87.00`, LSTM `0.5259`, delta `+0.2858`
- tick `148631`, seconds `93.00`, LSTM `0.8810`, delta `+0.1779`
- tick `148471`, seconds `90.50`, LSTM `0.6682`, delta `+0.1771`
- tick `148887`, seconds `97.00`, LSTM `0.9527`, delta `+0.1281`
- tick `148695`, seconds `94.00`, LSTM `0.8301`, delta `-0.0606`
- tick `148535`, seconds `91.50`, LSTM `0.7514`, delta `+0.0465`
- tick `147959`, seconds `82.50`, LSTM `0.2527`, delta `-0.0409`
- tick `147671`, seconds `78.00`, LSTM `0.3280`, delta `-0.0388`
- tick `146263`, seconds `56.00`, LSTM `0.3423`, delta `+0.0382`
- tick `148503`, seconds `91.00`, LSTM `0.7048`, delta `+0.0366`

## Top 15 local ridge features

- `lag_01__T_place_EXTENDEDA`: coefficient `0.002239`, |coef| `0.002239`
- `lag_15__T_flashes_last_5s`: coefficient `0.002144`, |coef| `0.002144`
- `lag_00__T_place_EXTENDEDA`: coefficient `-0.001842`, |coef| `0.001842`
- `lag_01__T_place_SHORTSTAIRS`: coefficient `-0.001820`, |coef| `0.001820`
- `lag_05__T_flashes_last_5s`: coefficient `-0.001693`, |coef| `0.001693`
- `lag_00__kill_diff_last_3s`: coefficient `0.001519`, |coef| `0.001519`
- `lag_00__CT4__is_scoped`: coefficient `-0.001474`, |coef| `0.001474`
- `lag_14__T_place_EXTENDEDA`: coefficient `0.001461`, |coef| `0.001461`
- `lag_00__CT_kills_last_3s`: coefficient `0.001451`, |coef| `0.001451`
- `lag_02__T4__flash_duration`: coefficient `0.001320`, |coef| `0.001320`
- `lag_08__T_place_SHORTSTAIRS`: coefficient `-0.001270`, |coef| `0.001270`
- `lag_00__damage_diff_last_5s`: coefficient `0.001224`, |coef| `0.001224`
- `lag_08__T_place_EXTENDEDA`: coefficient `0.001187`, |coef| `0.001187`
- `lag_02__T_place_EXTENDEDA`: coefficient `0.001186`, |coef| `0.001186`
- `lag_02__T_flashed_players`: coefficient `0.001150`, |coef| `0.001150`

## Top 10 utility ridge features

- `lag_15__T_flashes_last_5s`: coefficient `0.002144` (raises CT win probability)
- `lag_05__T_flashes_last_5s`: coefficient `-0.001693` (lowers CT win probability)
- `lag_02__T4__flash_duration`: coefficient `0.001320` (raises CT win probability)
- `lag_09__T5__flash_duration`: coefficient `0.001134` (raises CT win probability)
- `lag_02__T5__flash_duration`: coefficient `0.001052` (raises CT win probability)
- `lag_01__T2__flash_duration`: coefficient `-0.001039` (lowers CT win probability)
- `lag_02__CT4__flash_duration`: coefficient `0.000952` (raises CT win probability)
- `lag_03__T5__flash_duration`: coefficient `0.000945` (raises CT win probability)
- `lag_00__T_flashes_last_5s`: coefficient `-0.000857` (lowers CT win probability)
- `lag_15__T2__flash`: coefficient `-0.000823` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__T_place_EXTENDEDA`: coefficient `0.002239` (raises CT win probability)
- `lag_00__T_place_EXTENDEDA`: coefficient `-0.001842` (lowers CT win probability)
- `lag_01__T_place_SHORTSTAIRS`: coefficient `-0.001820` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001519` (raises CT win probability)
- `lag_00__CT4__is_scoped`: coefficient `-0.001474` (lowers CT win probability)
- `lag_14__T_place_EXTENDEDA`: coefficient `0.001461` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001451` (raises CT win probability)
- `lag_08__T_place_SHORTSTAIRS`: coefficient `-0.001270` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001224` (raises CT win probability)
- `lag_08__T_place_EXTENDEDA`: coefficient `0.001187` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `148247`, seconds `87.00`, LSTM delta `+0.2858`

Top all feature movements:
- `lag_01__T_place_EXTENDEDA`: contribution `+0.022201`
- `lag_15__T_flashes_last_5s`: contribution `+0.019426`
- `lag_05__T_flashes_last_5s`: contribution `+0.015338`
- `lag_01__T_place_SHORTSTAIRS`: contribution `+0.015298`
- `lag_00__T_place_EXTENDEDA`: contribution `+0.009130`

Top utility-only movements:
- `lag_15__T_flashes_last_5s`: contribution `+0.019426`
- `lag_05__T_flashes_last_5s`: contribution `+0.015338`
- `lag_02__T4__flash_duration`: contribution `+0.007193`
- `lag_02__CT4__flash_duration`: contribution `+0.004584`
- `lag_02__T5__flash_duration`: contribution `+0.004444`

### tick `148631`, seconds `93.00`, LSTM delta `+0.1779`

Top all feature movements:
- `lag_00__T_place_EXTENDEDA`: contribution `+0.009130`
- `lag_13__T_place_EXTENDEDA`: contribution `+0.008533`
- `lag_14__T_place_EXTENDEDA`: contribution `+0.007241`
- `lag_11__T_place_EXTENDEDA`: contribution `+0.005440`
- `lag_03__T5__flash_duration`: contribution `+0.005181`

Top utility-only movements:
- `lag_03__T5__flash_duration`: contribution `+0.005181`
- `lag_14__T_flash_duration_sum`: contribution `+0.003399`
- `lag_14__T5__flash_duration`: contribution `+0.003137`
- `lag_07__CT1__flash_duration`: contribution `+0.002012`

### tick `148471`, seconds `90.50`, LSTM delta `+0.1771`

Top all feature movements:
- `lag_08__T_place_EXTENDEDA`: contribution `+0.011767`
- `lag_08__T_place_SHORTSTAIRS`: contribution `+0.010674`
- `lag_00__T_place_EXTENDEDA`: contribution `+0.009130`
- `lag_12__T_flashes_last_5s`: contribution `+0.007209`
- `lag_09__T_place_EXTENDEDA`: contribution `+0.005298`

Top utility-only movements:
- `lag_12__T_flashes_last_5s`: contribution `+0.007209`
- `lag_09__T5__flash_duration`: contribution `+0.004789`
- `lag_09__T_flash_duration_sum`: contribution `+0.003336`
- `lag_01__T2__flash_duration`: contribution `+0.002861`
- `lag_09__T4__flash_duration`: contribution `+0.002753`

### tick `148887`, seconds `97.00`, LSTM delta `+0.1281`

Top all feature movements:
- `lag_00__T_place_EXTENDEDA`: contribution `+0.009130`
- `lag_00__kill_diff_last_3s`: contribution `+0.007313`
- `lag_01__T2__flash_duration`: contribution `+0.006067`
- `lag_08__T_place_EXTENDEDA`: contribution `-0.005884`
- `lag_12__CT_place_BDOORS`: contribution `+0.005103`

Top utility-only movements:
- `lag_01__T2__flash_duration`: contribution `+0.006067`
- `lag_11__T5__flash_duration`: contribution `+0.003962`
- `lag_11__T2__flash_duration`: contribution `+0.003915`
- `lag_01__T_flash_duration_sum`: contribution `+0.002886`
- `lag_11__T1__flash_duration`: contribution `+0.002500`

### tick `148695`, seconds `94.00`, LSTM delta `-0.0606`

Top all feature movements:
- `lag_14__T_place_EXTENDEDA`: contribution `-0.007241`
- `lag_02__T_place_EXTENDEDA`: contribution `-0.005878`
- `lag_15__T_place_SHORTSTAIRS`: contribution `-0.005184`
- `lag_15__T_place_EXTENDEDA`: contribution `+0.004894`
- `lag_13__T_place_EXTENDEDA`: contribution `+0.004267`

Top utility-only movements:
- `lag_09__CT1__flash_duration`: contribution `-0.002624`
- `lag_05__T_flash_duration_sum`: contribution `-0.001876`
- `lag_09__T5__flash_duration`: contribution `-0.001810`
- `lag_05__T1__flash_duration`: contribution `-0.001385`
- `lag_13__T5__flash_duration`: contribution `-0.001381`
