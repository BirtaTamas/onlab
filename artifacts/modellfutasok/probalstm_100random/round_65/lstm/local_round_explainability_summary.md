# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-g2-vs-gamerlegion-bo3-gcs9469UuxWlHi6X2zI5Oy/g2-vs-gamerlegion-m2-ancient.csv`
- round_num: `5`

## Largest probability jumps

- tick `27139`, seconds `37.00`, LSTM `0.2501`, delta `-0.2641`
- tick `26307`, seconds `24.00`, LSTM `0.1862`, delta `-0.2615`
- tick `30595`, seconds `91.00`, LSTM `0.0881`, delta `-0.2589`
- tick `30467`, seconds `89.00`, LSTM `0.2886`, delta `+0.1928`
- tick `30723`, seconds `93.00`, LSTM `0.2452`, delta `+0.1888`
- tick `26563`, seconds `28.00`, LSTM `0.3224`, delta `+0.1858`
- tick `31299`, seconds `102.00`, LSTM `0.0321`, delta `-0.0733`
- tick `31011`, seconds `97.50`, LSTM `0.1783`, delta `-0.0704`
- tick `26499`, seconds `27.00`, LSTM `0.1332`, delta `+0.0605`
- tick `27171`, seconds `37.50`, LSTM `0.1904`, delta `-0.0598`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004256`, |coef| `0.004256`
- `lag_00__CT3__flash_duration`: coefficient `0.003431`, |coef| `0.003431`
- `lag_00__damage_diff_last_5s`: coefficient `0.003310`, |coef| `0.003310`
- `lag_00__T_kills_last_3s`: coefficient `-0.002942`, |coef| `0.002942`
- `lag_00__T_damage_last_5s`: coefficient `-0.002816`, |coef| `0.002816`
- `lag_09__CT3__flash_duration`: coefficient `-0.002537`, |coef| `0.002537`
- `lag_09__T2__flash_duration`: coefficient `0.002441`, |coef| `0.002441`
- `lag_00__CT_kills_last_3s`: coefficient `0.002424`, |coef| `0.002424`
- `lag_08__CT5__flash_duration`: coefficient `-0.002336`, |coef| `0.002336`
- `lag_13__CT3__flash_duration`: coefficient `0.002306`, |coef| `0.002306`
- `lag_05__CT3__flash_duration`: coefficient `0.002265`, |coef| `0.002265`
- `lag_14__T1__shots_fired`: coefficient `-0.002194`, |coef| `0.002194`
- `lag_06__T_flashed_players`: coefficient `0.002081`, |coef| `0.002081`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001953`, |coef| `0.001953`
- `lag_07__T_flashed_players`: coefficient `0.001924`, |coef| `0.001924`

## Top 10 utility ridge features

- `lag_00__CT3__flash_duration`: coefficient `0.003431` (raises CT win probability)
- `lag_09__CT3__flash_duration`: coefficient `-0.002537` (lowers CT win probability)
- `lag_09__T2__flash_duration`: coefficient `0.002441` (raises CT win probability)
- `lag_08__CT5__flash_duration`: coefficient `-0.002336` (lowers CT win probability)
- `lag_13__CT3__flash_duration`: coefficient `0.002306` (raises CT win probability)
- `lag_05__CT3__flash_duration`: coefficient `0.002265` (raises CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `-0.001816` (lowers CT win probability)
- `lag_08__T2__flash_duration`: coefficient `0.001806` (raises CT win probability)
- `lag_00__T2__flash_duration`: coefficient `-0.001583` (lowers CT win probability)
- `lag_07__CT5__flash_duration`: coefficient `-0.001489` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004256` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003310` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002942` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002816` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002424` (raises CT win probability)
- `lag_14__T1__shots_fired`: coefficient `-0.002194` (lowers CT win probability)
- `lag_06__T_flashed_players`: coefficient `0.002081` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001953` (lowers CT win probability)
- `lag_07__T_flashed_players`: coefficient `0.001924` (raises CT win probability)
- `lag_10__CT_place_ALLEY`: coefficient `0.001920` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `27139`, seconds `37.00`, LSTM delta `-0.2641`

Top all feature movements:
- `lag_09__T2__flash_duration`: contribution `-0.015968`
- `lag_00__kill_diff_last_3s`: contribution `-0.010244`
- `lag_00__T_kills_last_3s`: contribution `-0.009321`
- `lag_08__T_shots_fired_sum`: contribution `-0.009264`
- `lag_01__T_place_WATER`: contribution `-0.008875`

Top utility-only movements:
- `lag_09__T2__flash_duration`: contribution `-0.015968`
- `lag_13__CT4__flash_duration`: contribution `-0.003901`

### tick `26307`, seconds `24.00`, LSTM delta `-0.2615`

Top all feature movements:
- `lag_08__CT5__flash_duration`: contribution `-0.017189`
- `lag_00__T_flashed_players`: contribution `-0.010921`
- `lag_00__T2__flash_duration`: contribution `-0.010737`
- `lag_00__kill_diff_last_3s`: contribution `-0.010244`
- `lag_00__T_kills_last_3s`: contribution `-0.009321`

Top utility-only movements:
- `lag_08__CT5__flash_duration`: contribution `-0.017189`
- `lag_00__T2__flash_duration`: contribution `-0.010737`
- `lag_00__CT4__flash_duration`: contribution `-0.008968`
- `lag_15__T4__flash_duration`: contribution `-0.007614`
- `lag_15__CT_utility_damage_last_5s`: contribution `-0.006737`

### tick `30595`, seconds `91.00`, LSTM delta `-0.2589`

Top all feature movements:
- `lag_00__CT3__flash_duration`: contribution `-0.022838`
- `lag_09__CT3__flash_duration`: contribution `-0.016887`
- `lag_00__kill_diff_last_3s`: contribution `-0.010244`
- `lag_00__T_kills_last_3s`: contribution `-0.009321`
- `lag_09__T_flashed_players`: contribution `-0.008318`

Top utility-only movements:
- `lag_00__CT3__flash_duration`: contribution `-0.022838`
- `lag_09__CT3__flash_duration`: contribution `-0.016887`
- `lag_09__T2__flash_duration`: contribution `+0.003555`

### tick `30467`, seconds `89.00`, LSTM delta `+0.1928`

Top all feature movements:
- `lag_05__CT3__flash_duration`: contribution `+0.015075`
- `lag_14__T1__shots_fired`: contribution `+0.010490`
- `lag_00__kill_diff_last_3s`: contribution `+0.010244`
- `lag_14__T_shots_fired_sum`: contribution `+0.010075`
- `lag_00__CT_kills_last_3s`: contribution `+0.006997`

Top utility-only movements:
- `lag_05__CT3__flash_duration`: contribution `+0.015075`

### tick `30723`, seconds `93.00`, LSTM delta `+0.1888`

Top all feature movements:
- `lag_13__CT3__flash_duration`: contribution `+0.015353`
- `lag_00__kill_diff_last_3s`: contribution `+0.010244`
- `lag_13__T_flashed_players`: contribution `+0.009711`
- `lag_04__CT3__flash_duration`: contribution `+0.007579`
- `lag_00__CT_kills_last_3s`: contribution `+0.006997`

Top utility-only movements:
- `lag_13__CT3__flash_duration`: contribution `+0.015353`
- `lag_04__CT3__flash_duration`: contribution `+0.007579`
- `lag_13__CT_flash_duration_sum`: contribution `+0.004108`
