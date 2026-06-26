# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-furia-vs-lynn-vision-bo3-KVSQ5iZB0TjTG70slfdqOB/furia-vs-lynn-vision-m2-overpass.csv`
- round_num: `12`

## Largest probability jumps

- tick `102267`, seconds `37.50`, LSTM `0.3420`, delta `-0.2163`
- tick `102939`, seconds `48.00`, LSTM `0.1010`, delta `-0.1856`
- tick `102779`, seconds `45.50`, LSTM `0.3725`, delta `+0.1822`
- tick `102299`, seconds `38.00`, LSTM `0.2382`, delta `-0.1039`
- tick `102619`, seconds `43.00`, LSTM `0.2305`, delta `-0.0694`
- tick `102843`, seconds `46.50`, LSTM `0.3400`, delta `-0.0682`
- tick `102331`, seconds `38.50`, LSTM `0.1833`, delta `-0.0549`
- tick `102459`, seconds `40.50`, LSTM `0.1691`, delta `+0.0464`
- tick `102971`, seconds `48.50`, LSTM `0.0553`, delta `-0.0457`
- tick `102875`, seconds `47.00`, LSTM `0.2954`, delta `-0.0446`

## Top 15 local ridge features

- `lag_03__CT_place_BRIDGE`: coefficient `-0.003356`, |coef| `0.003356`
- `lag_04__CT_place_BRIDGE`: coefficient `-0.002273`, |coef| `0.002273`
- `lag_14__CT_place_BRIDGE`: coefficient `-0.002056`, |coef| `0.002056`
- `lag_15__CT_place_CANAL`: coefficient `-0.001900`, |coef| `0.001900`
- `lag_09__CT_place_CANAL`: coefficient `-0.001899`, |coef| `0.001899`
- `lag_00__T_place_LOWERPARK`: coefficient `-0.001608`, |coef| `0.001608`
- `lag_15__T_place_CONNECTOR`: coefficient `0.001540`, |coef| `0.001540`
- `lag_01__T_place_LOWERPARK`: coefficient `-0.001490`, |coef| `0.001490`
- `lag_10__CT_place_WATER`: coefficient `0.001476`, |coef| `0.001476`
- `lag_15__T_place_FOUNTAIN`: coefficient `-0.001420`, |coef| `0.001420`
- `lag_12__T_place_FOUNTAIN`: coefficient `-0.001384`, |coef| `0.001384`
- `lag_00__kill_diff_last_3s`: coefficient `0.001364`, |coef| `0.001364`
- `lag_14__T5__flash_duration`: coefficient `0.001304`, |coef| `0.001304`
- `lag_09__T4__is_scoped`: coefficient `-0.001301`, |coef| `0.001301`
- `lag_05__CT_place_BRIDGE`: coefficient `-0.001297`, |coef| `0.001297`

## Top 10 utility ridge features

- `lag_14__T5__flash_duration`: coefficient `0.001304` (raises CT win probability)
- `lag_15__T5__flash_duration`: coefficient `0.001199` (raises CT win probability)
- `lag_07__CT3__flash_duration`: coefficient `0.001160` (raises CT win probability)
- `lag_00__CT5__flash`: coefficient `0.001045` (raises CT win probability)
- `lag_12__CT3__flash_duration`: coefficient `-0.001027` (lowers CT win probability)
- `lag_00__CT5__utility_total`: coefficient `0.000922` (raises CT win probability)
- `lag_01__CT5__flash`: coefficient `0.000813` (raises CT win probability)
- `lag_03__T_utility_damage_last_5s`: coefficient `-0.000733` (lowers CT win probability)
- `lag_00__CT5__smoke`: coefficient `0.000723` (raises CT win probability)
- `lag_01__CT5__utility_total`: coefficient `0.000673` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_03__CT_place_BRIDGE`: coefficient `-0.003356` (lowers CT win probability)
- `lag_04__CT_place_BRIDGE`: coefficient `-0.002273` (lowers CT win probability)
- `lag_14__CT_place_BRIDGE`: coefficient `-0.002056` (lowers CT win probability)
- `lag_15__CT_place_CANAL`: coefficient `-0.001900` (lowers CT win probability)
- `lag_09__CT_place_CANAL`: coefficient `-0.001899` (lowers CT win probability)
- `lag_00__T_place_LOWERPARK`: coefficient `-0.001608` (lowers CT win probability)
- `lag_15__T_place_CONNECTOR`: coefficient `0.001540` (raises CT win probability)
- `lag_01__T_place_LOWERPARK`: coefficient `-0.001490` (lowers CT win probability)
- `lag_10__CT_place_WATER`: coefficient `0.001476` (raises CT win probability)
- `lag_15__T_place_FOUNTAIN`: coefficient `-0.001420` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `102267`, seconds `37.50`, LSTM delta `-0.2163`

Top all feature movements:
- `lag_03__CT_place_BRIDGE`: contribution `-0.038468`
- `lag_09__CT_place_CANAL`: contribution `-0.011539`
- `lag_15__T_place_CONNECTOR`: contribution `-0.007459`
- `lag_14__T5__flash_duration`: contribution `-0.007241`
- `lag_12__T_place_FOUNTAIN`: contribution `-0.006542`

Top utility-only movements:
- `lag_14__T5__flash_duration`: contribution `-0.007241`
- `lag_00__CT5__flash`: contribution `-0.003709`

### tick `102939`, seconds `48.00`, LSTM delta `-0.1856`

Top all feature movements:
- `lag_03__CT_place_BRIDGE`: contribution `-0.038468`
- `lag_15__CT_place_CANAL`: contribution `-0.011547`
- `lag_04__CT_place_BACKOFA`: contribution `-0.011302`
- `lag_10__CT_place_WATER`: contribution `-0.008972`
- `lag_12__CT3__flash_duration`: contribution `-0.006642`

Top utility-only movements:
- `lag_12__CT3__flash_duration`: contribution `-0.006642`
- `lag_03__T_utility_damage_last_5s`: contribution `-0.002510`

### tick `102779`, seconds `45.50`, LSTM delta `+0.1822`

Top all feature movements:
- `lag_14__CT_place_BRIDGE`: contribution `+0.023565`
- `lag_15__CT_place_CANAL`: contribution `+0.011547`
- `lag_09__CT_place_CANAL`: contribution `+0.011539`
- `lag_10__CT_place_WATER`: contribution `+0.008972`
- `lag_07__CT3__flash_duration`: contribution `+0.007504`

Top utility-only movements:
- `lag_07__CT3__flash_duration`: contribution `+0.007504`

### tick `102299`, seconds `38.00`, LSTM delta `-0.1039`

Top all feature movements:
- `lag_04__CT_place_BRIDGE`: contribution `-0.026050`
- `lag_15__T5__flash_duration`: contribution `-0.006657`
- `lag_00__T_place_LOWERPARK`: contribution `-0.006483`
- `lag_04__CT_place_WALKWAY`: contribution `-0.006050`
- `lag_13__T_place_FOUNTAIN`: contribution `-0.004638`

Top utility-only movements:
- `lag_15__T5__flash_duration`: contribution `-0.006657`
- `lag_01__CT5__flash`: contribution `-0.002886`

### tick `102619`, seconds `43.00`, LSTM delta `-0.0694`

Top all feature movements:
- `lag_14__CT_place_BRIDGE`: contribution `-0.023565`
- `lag_15__CT_place_CANAL`: contribution `+0.011547`
- `lag_05__CT_place_WATER`: contribution `-0.006686`
- `lag_04__CT_place_CANAL`: contribution `-0.005433`
- `lag_14__CT_place_WALKWAY`: contribution `-0.004651`

Top utility-only movements:
- `lag_02__CT3__flash_duration`: contribution `+0.001519`
