# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-mouz-vs-falcons-bo3-xBECUqZMcQ8GCwi-GUyz8e/mouz-vs-falcons-m1-train.csv`
- round_num: `4`

## Largest probability jumps

- tick `40666`, seconds `60.00`, LSTM `0.1238`, delta `-0.1403`
- tick `40634`, seconds `59.50`, LSTM `0.2641`, delta `-0.1074`
- tick `40570`, seconds `58.50`, LSTM `0.4117`, delta `-0.0616`
- tick `40762`, seconds `61.50`, LSTM `0.0267`, delta `-0.0548`
- tick `40602`, seconds `59.00`, LSTM `0.3715`, delta `-0.0402`
- tick `37402`, seconds `9.00`, LSTM `0.4504`, delta `-0.0383`
- tick `40698`, seconds `60.50`, LSTM `0.0871`, delta `-0.0366`
- tick `37690`, seconds `13.50`, LSTM `0.4919`, delta `+0.0293`
- tick `37370`, seconds `8.50`, LSTM `0.4887`, delta `+0.0277`
- tick `37018`, seconds `3.00`, LSTM `0.4490`, delta `-0.0240`

## Top 15 local ridge features

- `lag_12__T1__flash_duration`: coefficient `-0.001248`, |coef| `0.001248`
- `lag_11__T1__flash_duration`: coefficient `-0.001127`, |coef| `0.001127`
- `lag_01__T_shots_fired_sum`: coefficient `-0.001009`, |coef| `0.001009`
- `lag_12__CT_place_ENTRANCE`: coefficient `0.000985`, |coef| `0.000985`
- `lag_13__CT3__flash_duration`: coefficient `-0.000954`, |coef| `0.000954`
- `lag_03__CT_place_CONNECTOR`: coefficient `-0.000903`, |coef| `0.000903`
- `lag_01__T1__shots_fired`: coefficient `-0.000858`, |coef| `0.000858`
- `lag_03__T_place_BACKOFB`: coefficient `0.000852`, |coef| `0.000852`
- `lag_03__T_shots_fired_sum`: coefficient `-0.000819`, |coef| `0.000819`
- `lag_11__CT_place_LONGDOG`: coefficient `0.000815`, |coef| `0.000815`
- `lag_12__CT3__flash_duration`: coefficient `-0.000812`, |coef| `0.000812`
- `lag_00__T_shots_fired_sum`: coefficient `-0.000791`, |coef| `0.000791`
- `lag_11__CT_place_ELECTRICALBOX`: coefficient `0.000745`, |coef| `0.000745`
- `lag_03__T_macro_B`: coefficient `-0.000732`, |coef| `0.000732`
- `lag_03__T_place_BOMBSITEB`: coefficient `-0.000732`, |coef| `0.000732`

## Top 10 utility ridge features

- `lag_12__T1__flash_duration`: coefficient `-0.001248` (lowers CT win probability)
- `lag_11__T1__flash_duration`: coefficient `-0.001127` (lowers CT win probability)
- `lag_13__CT3__flash_duration`: coefficient `-0.000954` (lowers CT win probability)
- `lag_12__CT3__flash_duration`: coefficient `-0.000812` (lowers CT win probability)
- `lag_13__CT4__flash_duration`: coefficient `-0.000716` (lowers CT win probability)
- `lag_11__CT4__flash_duration`: coefficient `-0.000709` (lowers CT win probability)
- `lag_09__T1__flash_duration`: coefficient `-0.000683` (lowers CT win probability)
- `lag_14__CT4__flash_duration`: coefficient `-0.000651` (lowers CT win probability)
- `lag_14__CT3__flash_duration`: coefficient `-0.000611` (lowers CT win probability)
- `lag_10__T1__flash_duration`: coefficient `-0.000603` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__T_shots_fired_sum`: coefficient `-0.001009` (lowers CT win probability)
- `lag_12__CT_place_ENTRANCE`: coefficient `0.000985` (raises CT win probability)
- `lag_03__CT_place_CONNECTOR`: coefficient `-0.000903` (lowers CT win probability)
- `lag_01__T1__shots_fired`: coefficient `-0.000858` (lowers CT win probability)
- `lag_03__T_place_BACKOFB`: coefficient `0.000852` (raises CT win probability)
- `lag_03__T_shots_fired_sum`: coefficient `-0.000819` (lowers CT win probability)
- `lag_11__CT_place_LONGDOG`: coefficient `0.000815` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.000791` (lowers CT win probability)
- `lag_11__CT_place_ELECTRICALBOX`: coefficient `0.000745` (raises CT win probability)
- `lag_03__T_macro_B`: coefficient `-0.000732` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `40666`, seconds `60.00`, LSTM delta `-0.1403`

Top all feature movements:
- `lag_12__T1__flash_duration`: contribution `-0.009766`
- `lag_15__CT_place_ELECTRICALBOX`: contribution `-0.006826`
- `lag_11__CT_place_LONGDOG`: contribution `-0.005319`
- `lag_13__CT3__flash_duration`: contribution `-0.005313`
- `lag_13__T_flashed_players`: contribution `-0.004164`

Top utility-only movements:
- `lag_12__T1__flash_duration`: contribution `-0.009766`
- `lag_13__CT3__flash_duration`: contribution `-0.005313`
- `lag_14__CT4__flash_duration`: contribution `-0.003979`
- `lag_12__T_flash_duration_sum`: contribution `-0.001707`

### tick `40634`, seconds `59.50`, LSTM delta `-0.1074`

Top all feature movements:
- `lag_11__T1__flash_duration`: contribution `-0.008822`
- `lag_13__CT_place_ELECTRICALBOX`: contribution `-0.006442`
- `lag_12__CT3__flash_duration`: contribution `-0.004524`
- `lag_10__CT_place_LONGDOG`: contribution `-0.004486`
- `lag_13__CT4__flash_duration`: contribution `-0.004379`

Top utility-only movements:
- `lag_11__T1__flash_duration`: contribution `-0.008822`
- `lag_12__CT3__flash_duration`: contribution `-0.004524`
- `lag_13__CT4__flash_duration`: contribution `-0.004379`
- `lag_13__CT_flash_duration_sum`: contribution `-0.001855`

### tick `40570`, seconds `58.50`, LSTM delta `-0.0616`

Top all feature movements:
- `lag_11__CT_place_ELECTRICALBOX`: contribution `-0.008658`
- `lag_09__T1__flash_duration`: contribution `-0.005348`
- `lag_11__CT4__flash_duration`: contribution `-0.004332`
- `lag_03__CT_place_CONNECTOR`: contribution `-0.003229`
- `lag_08__CT_place_LONGDOG`: contribution `-0.002829`

Top utility-only movements:
- `lag_09__T1__flash_duration`: contribution `-0.005348`
- `lag_11__CT4__flash_duration`: contribution `-0.004332`
- `lag_11__CT_flash_duration_sum`: contribution `-0.001363`
- `lag_03__CT3__flash_duration`: contribution `-0.001169`

### tick `40762`, seconds `61.50`, LSTM delta `-0.0548`

Top all feature movements:
- `lag_02__T_shots_fired_sum`: contribution `+0.006453`
- `lag_02__T1__shots_fired`: contribution `+0.004807`
- `lag_03__T_shots_fired_sum`: contribution `-0.003071`
- `lag_00__T_shots_fired_sum`: contribution `-0.002964`
- `lag_04__T1__duck_amount`: contribution `+0.002708`

Top utility-only movements:
- `lag_15__T1__flash_duration`: contribution `-0.002384`
- `lag_13__CT3__flash_duration`: contribution `+0.002022`
- `lag_01__T1__flash_duration`: contribution `-0.001260`

### tick `40602`, seconds `59.00`, LSTM delta `-0.0402`

Top all feature movements:
- `lag_13__CT_place_ELECTRICALBOX`: contribution `+0.006442`
- `lag_10__T1__flash_duration`: contribution `-0.004721`
- `lag_00__T_shots_fired_sum`: contribution `-0.002964`
- `lag_12__CT4__flash_duration`: contribution `-0.002690`
- `lag_08__CT_place_CONNECTOR`: contribution `-0.002556`

Top utility-only movements:
- `lag_10__T1__flash_duration`: contribution `-0.004721`
- `lag_12__CT4__flash_duration`: contribution `-0.002690`
- `lag_11__CT3__flash_duration`: contribution `-0.002020`
- `lag_12__CT_flash_duration_sum`: contribution `-0.001533`
