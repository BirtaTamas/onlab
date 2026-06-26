# Local Round Explainability

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-faze-vs-bcgame-bo3-daIGc_M_y7Qq42fF9AoQhi/faze-vs-bcgame-m2-anubis.csv`
- round_num: `12`

## Largest probability jumps

- tick `76149`, seconds `16.50`, LSTM `0.7064`, delta `+0.1349`
- tick `76181`, seconds `17.00`, LSTM `0.8301`, delta `+0.1236`
- tick `76469`, seconds `21.50`, LSTM `0.9197`, delta `+0.0840`
- tick `75413`, seconds `5.00`, LSTM `0.5750`, delta `+0.0317`
- tick `78325`, seconds `50.50`, LSTM `0.9718`, delta `+0.0248`
- tick `75349`, seconds `4.00`, LSTM `0.5443`, delta `+0.0223`
- tick `75573`, seconds `7.50`, LSTM `0.5314`, delta `-0.0219`
- tick `75125`, seconds `0.50`, LSTM `0.5290`, delta `-0.0218`
- tick `75925`, seconds `13.00`, LSTM `0.5737`, delta `+0.0212`
- tick `76213`, seconds `17.50`, LSTM `0.8506`, delta `+0.0206`

## Top 15 local ridge features

- `lag_00__CT_place_CTSIDEUPPER`: coefficient `-0.001289`, |coef| `0.001289`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001222`, |coef| `0.001222`
- `lag_00__CT_kills_last_3s`: coefficient `0.001098`, |coef| `0.001098`
- `lag_00__kill_diff_last_3s`: coefficient `0.000915`, |coef| `0.000915`
- `lag_01__CT3__flash_duration`: coefficient `-0.000903`, |coef| `0.000903`
- `lag_13__CT3__flash_duration`: coefficient `0.000873`, |coef| `0.000873`
- `lag_01__CT_shots_fired_sum`: coefficient `0.000806`, |coef| `0.000806`
- `lag_11__T_place_STREET`: coefficient `-0.000763`, |coef| `0.000763`
- `lag_03__CT_place_HEAVEN`: coefficient `0.000761`, |coef| `0.000761`
- `lag_15__T_place_TSTAIRS`: coefficient `0.000757`, |coef| `0.000757`
- `lag_02__CT3__flash_duration`: coefficient `-0.000717`, |coef| `0.000717`
- `lag_10__T_place_TSTAIRS`: coefficient `0.000714`, |coef| `0.000714`
- `lag_04__T_place_TSTAIRS`: coefficient `-0.000712`, |coef| `0.000712`
- `lag_15__T_place_CONNECTOR`: coefficient `0.000689`, |coef| `0.000689`
- `lag_14__CT3__flash_duration`: coefficient `0.000689`, |coef| `0.000689`

## Top 10 utility ridge features

- `lag_01__CT3__flash_duration`: coefficient `-0.000903` (lowers CT win probability)
- `lag_13__CT3__flash_duration`: coefficient `0.000873` (raises CT win probability)
- `lag_02__CT3__flash_duration`: coefficient `-0.000717` (lowers CT win probability)
- `lag_14__CT3__flash_duration`: coefficient `0.000689` (raises CT win probability)
- `lag_13__CT_flash_duration_sum`: coefficient `0.000555` (raises CT win probability)
- `lag_02__CT_utility_damage_last_5s`: coefficient `0.000546` (raises CT win probability)
- `lag_01__CT_utility_damage_last_5s`: coefficient `0.000482` (raises CT win probability)
- `lag_02__utility_damage_diff_last_5s`: coefficient `0.000473` (raises CT win probability)
- `lag_00__T3__smoke`: coefficient `-0.000454` (lowers CT win probability)
- `lag_01__CT_flash_duration_sum`: coefficient `-0.000438` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_CTSIDEUPPER`: coefficient `-0.001289` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001222` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001098` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000915` (raises CT win probability)
- `lag_01__CT_shots_fired_sum`: coefficient `0.000806` (raises CT win probability)
- `lag_11__T_place_STREET`: coefficient `-0.000763` (lowers CT win probability)
- `lag_03__CT_place_HEAVEN`: coefficient `0.000761` (raises CT win probability)
- `lag_15__T_place_TSTAIRS`: coefficient `0.000757` (raises CT win probability)
- `lag_10__T_place_TSTAIRS`: coefficient `0.000714` (raises CT win probability)
- `lag_04__T_place_TSTAIRS`: coefficient `-0.000712` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `76149`, seconds `16.50`, LSTM delta `+0.1349`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.006792`
- `lag_01__CT3__flash_duration`: contribution `+0.006097`
- `lag_13__CT3__flash_duration`: contribution `+0.005900`
- `lag_15__T_place_TSTAIRS`: contribution `+0.004290`
- `lag_11__T_place_STREET`: contribution `+0.004195`

Top utility-only movements:
- `lag_01__CT3__flash_duration`: contribution `+0.006097`
- `lag_13__CT3__flash_duration`: contribution `+0.005900`
- `lag_01__CT_utility_damage_last_5s`: contribution `+0.002654`
- `lag_13__CT_flash_duration_sum`: contribution `+0.002392`
- `lag_01__utility_damage_diff_last_5s`: contribution `+0.001798`

### tick `76181`, seconds `17.00`, LSTM delta `+0.1236`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.009339`
- `lag_02__CT3__flash_duration`: contribution `+0.004847`
- `lag_14__CT3__flash_duration`: contribution `+0.004651`
- `lag_01__CT_shots_fired_sum`: contribution `+0.004478`
- `lag_05__T_place_TSTAIRS`: contribution `+0.003608`

Top utility-only movements:
- `lag_02__CT3__flash_duration`: contribution `+0.004847`
- `lag_14__CT3__flash_duration`: contribution `+0.004651`
- `lag_02__CT_utility_damage_last_5s`: contribution `+0.003007`
- `lag_02__utility_damage_diff_last_5s`: contribution `+0.002136`
- `lag_14__CT_flash_duration_sum`: contribution `+0.001864`

### tick `76469`, seconds `21.50`, LSTM delta `+0.0840`

Top all feature movements:
- `lag_08__CT_shots_fired_sum`: contribution `+0.008663`
- `lag_00__CT_kills_last_3s`: contribution `+0.003169`
- `lag_01__CT_utility_damage_last_5s`: contribution `-0.002601`
- `lag_11__CT3__flash_duration`: contribution `+0.002377`
- `lag_09__CT_shots_fired_sum`: contribution `+0.002337`

Top utility-only movements:
- `lag_01__CT_utility_damage_last_5s`: contribution `-0.002601`
- `lag_11__CT3__flash_duration`: contribution `+0.002377`
- `lag_11__CT_utility_damage_last_5s`: contribution `+0.002269`
- `lag_01__utility_damage_diff_last_5s`: contribution `-0.001762`
- `lag_11__utility_damage_diff_last_5s`: contribution `+0.001550`

### tick `75413`, seconds `5.00`, LSTM delta `+0.0317`

Top all feature movements:
- `lag_10__CT_place_CTSIDEUPPER`: contribution `+0.007830`
- `lag_05__CT_place_CTSIDEUPPER`: contribution `+0.005865`
- `lag_00__CT_place_PALACEINTERIOR`: contribution `+0.003106`
- `lag_00__CT_place_LOWERTUNNEL`: contribution `+0.001237`
- `lag_05__CT_place_LOWERTUNNEL`: contribution `+0.000869`

Top utility-only movements:
- `lag_10__CT5__flash`: contribution `+0.000677`
- `lag_10__CT_flash_inv`: contribution `+0.000316`

### tick `78325`, seconds `50.50`, LSTM delta `+0.0248`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.003396`
- `lag_00__CT_kills_last_3s`: contribution `+0.003169`
- `lag_01__CT_shots_fired_sum`: contribution `+0.002239`
- `lag_00__kill_diff_last_3s`: contribution `+0.002202`
- `lag_03__CT_place_LOWERTUNNEL`: contribution `+0.001999`

Top utility-only movements:
- `lag_00__T5__utility_total`: contribution `+0.000539`
- `lag_05__T1__flash_duration`: contribution `+0.000514`
