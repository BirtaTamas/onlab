# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-imperial-vs-liquid-bo3-eiIGPV5tjvJFQ73hC8D8JI/imperial-vs-liquid-m3-anubis.csv`
- round_num: `14`

## Largest probability jumps

- tick `117086`, seconds `40.50`, LSTM `0.2337`, delta `-0.2684`
- tick `117118`, seconds `41.00`, LSTM `0.1452`, delta `-0.0885`
- tick `117182`, seconds `42.00`, LSTM `0.0433`, delta `-0.0562`
- tick `117150`, seconds `41.50`, LSTM `0.0994`, delta `-0.0457`
- tick `114974`, seconds `7.50`, LSTM `0.5362`, delta `-0.0252`
- tick `117310`, seconds `44.00`, LSTM `0.0126`, delta `-0.0217`
- tick `114846`, seconds `5.50`, LSTM `0.5529`, delta `+0.0213`
- tick `117022`, seconds `39.50`, LSTM `0.4950`, delta `-0.0172`
- tick `114782`, seconds `4.50`, LSTM `0.5206`, delta `+0.0162`
- tick `115006`, seconds `8.00`, LSTM `0.5249`, delta `-0.0114`

## Top 15 local ridge features

- `lag_05__CT_place_FOUNTAIN`: coefficient `0.003427`, |coef| `0.003427`
- `lag_01__CT_place_WALKWAY`: coefficient `0.002430`, |coef| `0.002430`
- `lag_05__CT_place_WALKWAY`: coefficient `-0.002363`, |coef| `0.002363`
- `lag_03__CT_flashed_players`: coefficient `-0.002063`, |coef| `0.002063`
- `lag_11__T_place_MAIN`: coefficient `-0.001953`, |coef| `0.001953`
- `lag_15__T_place_MAIN`: coefficient `-0.001828`, |coef| `0.001828`
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `0.001648`, |coef| `0.001648`
- `lag_03__CT3__flash_duration`: coefficient `-0.001619`, |coef| `0.001619`
- `lag_00__T_kills_last_3s`: coefficient `-0.001506`, |coef| `0.001506`
- `lag_00__CT2__alive`: coefficient `0.001489`, |coef| `0.001489`
- `lag_00__CT2__hp`: coefficient `0.001472`, |coef| `0.001472`
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.001455`, |coef| `0.001455`
- `lag_08__T_place_MAIN`: coefficient `-0.001431`, |coef| `0.001431`
- `lag_02__T_A_site_active_infernos`: coefficient `-0.001421`, |coef| `0.001421`
- `lag_00__T_damage_last_5s`: coefficient `-0.001418`, |coef| `0.001418`

## Top 10 utility ridge features

- `lag_03__CT3__flash_duration`: coefficient `-0.001619` (lowers CT win probability)
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.001455` (lowers CT win probability)
- `lag_02__T_A_site_active_infernos`: coefficient `-0.001421` (lowers CT win probability)
- `lag_02__T1__flash_duration`: coefficient `-0.001415` (lowers CT win probability)
- `lag_03__T_A_site_active_smokes`: coefficient `-0.001382` (lowers CT win probability)
- `lag_05__T2__molly`: coefficient `0.001265` (raises CT win probability)
- `lag_09__T4__smoke`: coefficient `0.001235` (raises CT win probability)
- `lag_10__T5__smoke`: coefficient `0.001208` (raises CT win probability)
- `lag_02__CT3__flash_duration`: coefficient `-0.001098` (lowers CT win probability)
- `lag_05__CT3__flash`: coefficient `0.001058` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_05__CT_place_FOUNTAIN`: coefficient `0.003427` (raises CT win probability)
- `lag_01__CT_place_WALKWAY`: coefficient `0.002430` (raises CT win probability)
- `lag_05__CT_place_WALKWAY`: coefficient `-0.002363` (lowers CT win probability)
- `lag_03__CT_flashed_players`: coefficient `-0.002063` (lowers CT win probability)
- `lag_11__T_place_MAIN`: coefficient `-0.001953` (lowers CT win probability)
- `lag_15__T_place_MAIN`: coefficient `-0.001828` (lowers CT win probability)
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `0.001648` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001506` (lowers CT win probability)
- `lag_00__CT2__alive`: coefficient `0.001489` (raises CT win probability)
- `lag_00__CT2__hp`: coefficient `0.001472` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `117086`, seconds `40.50`, LSTM delta `-0.2684`

Top all feature movements:
- `lag_05__CT_place_FOUNTAIN`: contribution `-0.036042`
- `lag_11__T_place_MAIN`: contribution `-0.012629`
- `lag_01__CT_place_WALKWAY`: contribution `-0.011926`
- `lag_15__T_place_MAIN`: contribution `-0.011816`
- `lag_05__CT_place_WALKWAY`: contribution `-0.011599`

Top utility-only movements:
- `lag_03__CT3__flash_duration`: contribution `-0.004561`
- `lag_02__T_A_site_active_infernos`: contribution `-0.004228`
- `lag_02__T1__flash_duration`: contribution `-0.004010`
- `lag_03__T_A_site_active_smokes`: contribution `-0.003933`

### tick `117118`, seconds `41.00`, LSTM delta `-0.0885`

Top all feature movements:
- `lag_06__CT_place_FOUNTAIN`: contribution `-0.009820`
- `lag_12__T_place_MAIN`: contribution `-0.005573`
- `lag_02__CT_place_WALKWAY`: contribution `-0.005363`
- `lag_06__CT_place_WALKWAY`: contribution `-0.005072`
- `lag_04__T_place_MAIN`: contribution `-0.004469`

Top utility-only movements:
- `lag_03__CT3__flash_duration`: contribution `-0.002725`
- `lag_00__T_utility_damage_last_5s`: contribution `-0.002286`
- `lag_03__T1__flash_duration`: contribution `-0.001798`
- `lag_03__T_A_site_active_infernos`: contribution `-0.001774`
- `lag_04__T_A_site_active_smokes`: contribution `-0.001716`

### tick `117182`, seconds `42.00`, LSTM delta `-0.0562`

Top all feature movements:
- `lag_11__T_place_MAIN`: contribution `-0.012629`
- `lag_00__T_kills_last_3s`: contribution `-0.004771`
- `lag_03__CT_flashed_players`: contribution `+0.004518`
- `lag_14__T_place_MAIN`: contribution `-0.004331`
- `lag_00__T_utility_damage_last_5s`: contribution `-0.003325`

Top utility-only movements:
- `lag_00__T_utility_damage_last_5s`: contribution `-0.003325`
- `lag_01__T_utility_damage_last_5s`: contribution `-0.002228`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.001330`

### tick `117150`, seconds `41.50`, LSTM delta `-0.0457`

Top all feature movements:
- `lag_00__T_utility_damage_last_5s`: contribution `-0.004779`
- `lag_13__T_place_MAIN`: contribution `-0.004013`
- `lag_00__CT_place_CANAL`: contribution `-0.003451`
- `lag_10__T_place_MAIN`: contribution `-0.003171`
- `lag_08__CT1__is_walking`: contribution `+0.002364`

Top utility-only movements:
- `lag_00__T_utility_damage_last_5s`: contribution `-0.004779`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.001911`
- `lag_02__CT2__flash_duration`: contribution `+0.001074`

### tick `114974`, seconds `7.50`, LSTM delta `-0.0252`

Top all feature movements:
- `lag_11__CT_place_CTSIDEUPPER`: contribution `-0.004580`
- `lag_15__CT_place_CTSIDEUPPER`: contribution `-0.004418`
- `lag_10__CT_place_CTSIDEUPPER`: contribution `-0.003980`
- `lag_00__CT_place_TUNNEL`: contribution `-0.003929`
- `lag_11__CT_place_PALACEINTERIOR`: contribution `-0.003875`

Top utility-only movements:
- No utility movement among the top local contributors.
