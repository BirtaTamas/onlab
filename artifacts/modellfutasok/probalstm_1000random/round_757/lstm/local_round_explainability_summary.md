# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-og-vs-tyloo-ancient-6bJQWEKo0L9rHQMGqH72Vs/og-vs-tyloo-ancient.csv`
- round_num: `4`

## Largest probability jumps

- tick `22050`, seconds `19.50`, LSTM `0.3231`, delta `-0.2567`
- tick `24418`, seconds `56.50`, LSTM `0.3918`, delta `+0.2423`
- tick `22210`, seconds `22.00`, LSTM `0.1185`, delta `-0.2272`
- tick `24482`, seconds `57.50`, LSTM `0.1389`, delta `-0.2212`
- tick `24322`, seconds `55.00`, LSTM `0.1943`, delta `-0.0865`
- tick `23970`, seconds `49.50`, LSTM `0.2727`, delta `+0.0810`
- tick `24290`, seconds `54.50`, LSTM `0.2807`, delta `-0.0492`
- tick `23874`, seconds `48.00`, LSTM `0.1557`, delta `+0.0454`
- tick `24002`, seconds `50.00`, LSTM `0.3156`, delta `+0.0429`
- tick `23650`, seconds `44.50`, LSTM `0.0469`, delta `+0.0380`

## Top 15 local ridge features

- `lag_10__T_place_TSIDELOWER`: coefficient `0.003134`, |coef| `0.003134`
- `lag_02__T_shots_fired_sum`: coefficient `0.002461`, |coef| `0.002461`
- `lag_00__CT_flashed_players`: coefficient `0.002086`, |coef| `0.002086`
- `lag_11__CT_place_TSIDEUPPER`: coefficient `0.002076`, |coef| `0.002076`
- `lag_00__CT_flash_duration_sum`: coefficient `0.001939`, |coef| `0.001939`
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.001911`, |coef| `0.001911`
- `lag_13__CT3__is_scoped`: coefficient `-0.001849`, |coef| `0.001849`
- `lag_08__T_place_RAMP`: coefficient `0.001783`, |coef| `0.001783`
- `lag_06__CT_place_TSIDEUPPER`: coefficient `0.001776`, |coef| `0.001776`
- `lag_14__CT3__flash_duration`: coefficient `0.001714`, |coef| `0.001714`
- `lag_00__CT5__flash_duration`: coefficient `0.001688`, |coef| `0.001688`
- `lag_00__kill_diff_last_3s`: coefficient `0.001631`, |coef| `0.001631`
- `lag_10__T_place_RUINS`: coefficient `-0.001603`, |coef| `0.001603`
- `lag_00__T4__flash_duration`: coefficient `0.001593`, |coef| `0.001593`
- `lag_13__CT_place_TSIDEUPPER`: coefficient `-0.001552`, |coef| `0.001552`

## Top 10 utility ridge features

- `lag_00__CT_flash_duration_sum`: coefficient `0.001939` (raises CT win probability)
- `lag_14__CT3__flash_duration`: coefficient `0.001714` (raises CT win probability)
- `lag_00__CT5__flash_duration`: coefficient `0.001688` (raises CT win probability)
- `lag_00__T4__flash_duration`: coefficient `0.001593` (raises CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `0.001515` (raises CT win probability)
- `lag_14__T4__flash_duration`: coefficient `0.001488` (raises CT win probability)
- `lag_01__CT3__flash_duration`: coefficient `0.001221` (raises CT win probability)
- `lag_05__CT5__flash_duration`: coefficient `0.001216` (raises CT win probability)
- `lag_14__CT_flash_duration_sum`: coefficient `0.001211` (raises CT win probability)
- `lag_11__T4__flash_duration`: coefficient `-0.001172` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_10__T_place_TSIDELOWER`: coefficient `0.003134` (raises CT win probability)
- `lag_02__T_shots_fired_sum`: coefficient `0.002461` (raises CT win probability)
- `lag_00__CT_flashed_players`: coefficient `0.002086` (raises CT win probability)
- `lag_11__CT_place_TSIDEUPPER`: coefficient `0.002076` (raises CT win probability)
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.001911` (raises CT win probability)
- `lag_13__CT3__is_scoped`: coefficient `-0.001849` (lowers CT win probability)
- `lag_08__T_place_RAMP`: coefficient `0.001783` (raises CT win probability)
- `lag_06__CT_place_TSIDEUPPER`: coefficient `0.001776` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001631` (raises CT win probability)
- `lag_10__T_place_RUINS`: coefficient `-0.001603` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `22050`, seconds `19.50`, LSTM delta `-0.2567`

Top all feature movements:
- `lag_02__T_shots_fired_sum`: contribution `-0.016609`
- `lag_00__CT_place_TSIDEUPPER`: contribution `-0.014367`
- `lag_00__T_shots_fired_sum`: contribution `-0.011160`
- `lag_14__T_utility_damage_last_5s`: contribution `-0.009257`
- `lag_07__CT_place_TSIDEUPPER`: contribution `-0.007735`

Top utility-only movements:
- `lag_14__T_utility_damage_last_5s`: contribution `-0.009257`
- `lag_04__T_utility_damage_last_5s`: contribution `-0.007273`
- `lag_07__T1__flash_duration`: contribution `-0.005705`
- `lag_00__CT4__flash_duration`: contribution `-0.005596`
- `lag_00__CT_flash_duration_sum`: contribution `-0.005349`

### tick `24418`, seconds `56.50`, LSTM delta `+0.2423`

Top all feature movements:
- `lag_10__T_place_TSIDELOWER`: contribution `+0.023491`
- `lag_11__CT_place_TSIDEUPPER`: contribution `+0.015606`
- `lag_08__T_place_RAMP`: contribution `+0.012609`
- `lag_08__T_place_TSIDELOWER`: contribution `+0.011217`
- `lag_04__CT_place_TSIDEUPPER`: contribution `+0.010666`

Top utility-only movements:
- `lag_14__CT3__flash_duration`: contribution `+0.009427`
- `lag_14__T4__flash_duration`: contribution `+0.008564`
- `lag_14__CT_flash_duration_sum`: contribution `+0.005444`
- `lag_14__CT5__flash_duration`: contribution `+0.004651`
- `lag_03__CT5__flash_duration`: contribution `+0.004411`

### tick `22210`, seconds `22.00`, LSTM delta `-0.2272`

Top all feature movements:
- `lag_02__T_shots_fired_sum`: contribution `-0.033218`
- `lag_02__T5__shots_fired`: contribution `-0.011842`
- `lag_05__CT_place_TSIDEUPPER`: contribution `-0.006343`
- `lag_04__T_shots_fired_sum`: contribution `-0.006254`
- `lag_12__CT_place_TSIDEUPPER`: contribution `-0.006161`

Top utility-only movements:
- `lag_09__T_utility_damage_last_5s`: contribution `-0.005577`
- `lag_12__T5__flash_duration`: contribution `-0.004168`
- `lag_06__CT4__flash_duration`: contribution `-0.003942`
- `lag_12__T1__flash_duration`: contribution `-0.003917`
- `lag_12__T_flash_duration_sum`: contribution `-0.003850`

### tick `24482`, seconds `57.50`, LSTM delta `-0.2212`

Top all feature movements:
- `lag_10__T_place_TSIDELOWER`: contribution `-0.023491`
- `lag_06__CT_place_TSIDEUPPER`: contribution `-0.013348`
- `lag_13__CT_place_TSIDEUPPER`: contribution `-0.011668`
- `lag_10__T_place_RAMP`: contribution `-0.009713`
- `lag_12__T_place_TSIDELOWER`: contribution `-0.009671`

Top utility-only movements:
- `lag_05__CT5__flash_duration`: contribution `-0.005532`
- `lag_05__T4__flash_duration`: contribution `-0.004127`
- `lag_06__CT3__flash_duration`: contribution `-0.003040`

### tick `24322`, seconds `55.00`, LSTM delta `-0.0865`

Top all feature movements:
- `lag_00__CT5__flash_duration`: contribution `-0.007684`
- `lag_11__T4__flash_duration`: contribution `-0.006747`
- `lag_11__CT3__flash_duration`: contribution `-0.006289`
- `lag_01__CT_place_TSIDEUPPER`: contribution `-0.006131`
- `lag_00__T4__flash_duration`: contribution `-0.005731`

Top utility-only movements:
- `lag_00__CT5__flash_duration`: contribution `-0.007684`
- `lag_11__T4__flash_duration`: contribution `-0.006747`
- `lag_11__CT3__flash_duration`: contribution `-0.006289`
- `lag_00__T4__flash_duration`: contribution `-0.005731`
- `lag_00__CT_flash_duration_sum`: contribution `-0.004069`
