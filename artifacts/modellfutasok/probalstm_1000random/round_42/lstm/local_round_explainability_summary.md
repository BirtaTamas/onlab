# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m2-dust2.csv`
- round_num: `8`

## Largest probability jumps

- tick `62757`, seconds `24.00`, LSTM `0.3046`, delta `-0.1148`
- tick `62885`, seconds `26.00`, LSTM `0.1489`, delta `-0.0831`
- tick `62789`, seconds `24.50`, LSTM `0.2559`, delta `-0.0487`
- tick `62917`, seconds `26.50`, LSTM `0.1119`, delta `-0.0369`
- tick `61829`, seconds `9.50`, LSTM `0.3936`, delta `+0.0317`
- tick `63365`, seconds `33.50`, LSTM `0.0894`, delta `-0.0317`
- tick `62725`, seconds `23.50`, LSTM `0.4194`, delta `+0.0296`
- tick `63429`, seconds `34.50`, LSTM `0.0369`, delta `-0.0274`
- tick `61861`, seconds `10.00`, LSTM `0.3674`, delta `-0.0262`
- tick `63461`, seconds `35.00`, LSTM `0.0116`, delta `-0.0253`

## Top 15 local ridge features

- `lag_00__T_flashed_players`: coefficient `-0.001350`, |coef| `0.001350`
- `lag_08__CT_place_SHORTSTAIRS`: coefficient `0.001311`, |coef| `0.001311`
- `lag_08__CT_place_EXTENDEDA`: coefficient `-0.001091`, |coef| `0.001091`
- `lag_12__CT_place_SHORTSTAIRS`: coefficient `0.001024`, |coef| `0.001024`
- `lag_12__CT_place_LONGDOORS`: coefficient `-0.000996`, |coef| `0.000996`
- `lag_13__T_place_OUTSIDETUNNEL`: coefficient `-0.000869`, |coef| `0.000869`
- `lag_09__CT_place_EXTENDEDA`: coefficient `-0.000843`, |coef| `0.000843`
- `lag_00__CT4__alive`: coefficient `0.000835`, |coef| `0.000835`
- `lag_01__T_flashed_players`: coefficient `-0.000828`, |coef| `0.000828`
- `lag_00__CT4__duck_amount`: coefficient `0.000781`, |coef| `0.000781`
- `lag_13__CT_place_PIT`: coefficient `0.000781`, |coef| `0.000781`
- `lag_12__CT_place_EXTENDEDA`: coefficient `-0.000776`, |coef| `0.000776`
- `lag_15__CT_place_UNDERA`: coefficient `0.000766`, |coef| `0.000766`
- `lag_05__CT2__is_walking`: coefficient `0.000761`, |coef| `0.000761`
- `lag_00__CT3__duck_amount`: coefficient `0.000760`, |coef| `0.000760`

## Top 10 utility ridge features

- `lag_00__CT5__smoke`: coefficient `0.000747` (raises CT win probability)
- `lag_00__CT4__smoke`: coefficient `0.000743` (raises CT win probability)
- `lag_02__T1__smoke`: coefficient `0.000633` (raises CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `-0.000582` (lowers CT win probability)
- `lag_00__CT_smoke_inv`: coefficient `0.000569` (raises CT win probability)
- `lag_00__T1__flash`: coefficient `0.000565` (raises CT win probability)
- `lag_00__CT5__utility_total`: coefficient `0.000538` (raises CT win probability)
- `lag_00__T1__utility_total`: coefficient `0.000514` (raises CT win probability)
- `lag_00__T2__flash`: coefficient `0.000501` (raises CT win probability)
- `lag_00__T_flash_inv`: coefficient `0.000459` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_flashed_players`: coefficient `-0.001350` (lowers CT win probability)
- `lag_08__CT_place_SHORTSTAIRS`: coefficient `0.001311` (raises CT win probability)
- `lag_08__CT_place_EXTENDEDA`: coefficient `-0.001091` (lowers CT win probability)
- `lag_12__CT_place_SHORTSTAIRS`: coefficient `0.001024` (raises CT win probability)
- `lag_12__CT_place_LONGDOORS`: coefficient `-0.000996` (lowers CT win probability)
- `lag_13__T_place_OUTSIDETUNNEL`: coefficient `-0.000869` (lowers CT win probability)
- `lag_09__CT_place_EXTENDEDA`: coefficient `-0.000843` (lowers CT win probability)
- `lag_00__CT4__alive`: coefficient `0.000835` (raises CT win probability)
- `lag_01__T_flashed_players`: coefficient `-0.000828` (lowers CT win probability)
- `lag_00__CT4__duck_amount`: coefficient `0.000781` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `62757`, seconds `24.00`, LSTM delta `-0.1148`

Top all feature movements:
- `lag_08__CT_place_SHORTSTAIRS`: contribution `-0.007308`
- `lag_08__CT_place_EXTENDEDA`: contribution `-0.006122`
- `lag_12__CT_place_LONGDOORS`: contribution `-0.004364`
- `lag_01__CT_place_EXTENDEDA`: contribution `-0.004192`
- `lag_12__T_place_OUTSIDETUNNEL`: contribution `-0.003650`

Top utility-only movements:
- `lag_00__CT5__smoke`: contribution `-0.001638`
- `lag_00__CT4__smoke`: contribution `-0.001621`

### tick `62885`, seconds `26.00`, LSTM delta `-0.0831`

Top all feature movements:
- `lag_00__T_flashed_players`: contribution `-0.010422`
- `lag_12__CT_place_SHORTSTAIRS`: contribution `-0.005708`
- `lag_12__CT_place_EXTENDEDA`: contribution `-0.004355`
- `lag_00__CT2__flash_duration`: contribution `-0.002707`
- `lag_05__CT_place_EXTENDEDA`: contribution `-0.002661`

Top utility-only movements:
- `lag_00__CT2__flash_duration`: contribution `-0.002707`

### tick `62789`, seconds `24.50`, LSTM delta `-0.0487`

Top all feature movements:
- `lag_09__CT_place_EXTENDEDA`: contribution `-0.004731`
- `lag_13__T_place_OUTSIDETUNNEL`: contribution `-0.004341`
- `lag_09__CT_place_SHORTSTAIRS`: contribution `-0.003758`
- `lag_01__CT4__duck_amount`: contribution `-0.002381`
- `lag_02__CT_place_EXTENDEDA`: contribution `-0.002303`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `62917`, seconds `26.50`, LSTM delta `-0.0369`

Top all feature movements:
- `lag_01__T_flashed_players`: contribution `-0.006393`
- `lag_13__CT_place_SHORTSTAIRS`: contribution `-0.002500`
- `lag_05__CT4__duck_amount`: contribution `-0.001687`
- `lag_01__CT2__flash_duration`: contribution `-0.001566`
- `lag_02__CT3__duck_amount`: contribution `-0.001516`

Top utility-only movements:
- `lag_01__CT2__flash_duration`: contribution `-0.001566`

### tick `61829`, seconds `9.50`, LSTM delta `+0.0317`

Top all feature movements:
- `lag_11__T_he_last_5s`: contribution `+0.005938`
- `lag_15__CT_place_UNDERA`: contribution `+0.004681`
- `lag_00__CT_place_HOLE`: contribution `+0.004550`
- `lag_02__T_place_OUTSIDETUNNEL`: contribution `+0.004392`
- `lag_12__T_place_OUTSIDETUNNEL`: contribution `-0.003650`

Top utility-only movements:
- `lag_11__T_he_last_5s`: contribution `+0.005938`
- `lag_03__CT5__flash_duration`: contribution `+0.001276`
- `lag_01__T_he_last_5s`: contribution `+0.000904`
