# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-b8-vs-wildcard-bo3-EO1cCePneo0X8r6rxB_BMC/b8-vs-wildcard-m2-dust2.csv`
- round_num: `1`

## Largest probability jumps

- tick `5052`, seconds `12.00`, LSTM `0.0957`, delta `-0.2810`
- tick `6524`, seconds `35.00`, LSTM `0.6295`, delta `+0.2715`
- tick `6620`, seconds `36.50`, LSTM `0.6752`, delta `+0.2462`
- tick `5468`, seconds `18.50`, LSTM `0.5873`, delta `+0.2393`
- tick `6588`, seconds `36.00`, LSTM `0.4290`, delta `-0.2195`
- tick `6652`, seconds `37.00`, LSTM `0.5053`, delta `-0.1700`
- tick `5020`, seconds `11.50`, LSTM `0.3767`, delta `-0.1292`
- tick `7484`, seconds `50.00`, LSTM `0.2530`, delta `-0.0907`
- tick `6460`, seconds `34.00`, LSTM `0.3574`, delta `-0.0846`
- tick `5180`, seconds `14.00`, LSTM `0.1054`, delta `+0.0774`

## Top 15 local ridge features

- `lag_00__T_place_ARAMP`: coefficient `-0.003287`, |coef| `0.003287`
- `lag_00__kill_diff_last_3s`: coefficient `0.003140`, |coef| `0.003140`
- `lag_10__CT_place_LOWERTUNNEL`: coefficient `-0.002918`, |coef| `0.002918`
- `lag_02__T_place_ARAMP`: coefficient `0.002844`, |coef| `0.002844`
- `lag_00__damage_diff_last_5s`: coefficient `0.002538`, |coef| `0.002538`
- `lag_00__T_kills_last_3s`: coefficient `-0.002362`, |coef| `0.002362`
- `lag_09__T_flashed_players`: coefficient `0.002356`, |coef| `0.002356`
- `lag_11__CT_place_LOWERTUNNEL`: coefficient `-0.002288`, |coef| `0.002288`
- `lag_01__T_place_ARAMP`: coefficient `0.002040`, |coef| `0.002040`
- `lag_00__T_damage_last_5s`: coefficient `-0.001888`, |coef| `0.001888`
- `lag_12__T_flashes_last_5s`: coefficient `0.001860`, |coef| `0.001860`
- `lag_13__CT_place_ARAMP`: coefficient `-0.001764`, |coef| `0.001764`
- `lag_03__T_place_ARAMP`: coefficient `-0.001666`, |coef| `0.001666`
- `lag_04__CT_duck_amount_mean`: coefficient `0.001622`, |coef| `0.001622`
- `lag_11__CT_place_EXTENDEDA`: coefficient `0.001619`, |coef| `0.001619`

## Top 10 utility ridge features

- `lag_12__T_flashes_last_5s`: coefficient `0.001860` (raises CT win probability)
- `lag_09__CT_flash_duration_sum`: coefficient `0.001470` (raises CT win probability)
- `lag_09__CT3__flash_duration`: coefficient `0.001461` (raises CT win probability)
- `lag_10__T5__flash_duration`: coefficient `0.001322` (raises CT win probability)
- `lag_12__CT_flash_duration_sum`: coefficient `0.001315` (raises CT win probability)
- `lag_10__CT5__flash_duration`: coefficient `0.001230` (raises CT win probability)
- `lag_12__CT4__flash_duration`: coefficient `0.001100` (raises CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `-0.001061` (lowers CT win probability)
- `lag_09__CT4__flash_duration`: coefficient `0.001033` (raises CT win probability)
- `lag_12__CT3__flash_duration`: coefficient `0.001011` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_ARAMP`: coefficient `-0.003287` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003140` (raises CT win probability)
- `lag_10__CT_place_LOWERTUNNEL`: coefficient `-0.002918` (lowers CT win probability)
- `lag_02__T_place_ARAMP`: coefficient `0.002844` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002538` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002362` (lowers CT win probability)
- `lag_09__T_flashed_players`: coefficient `0.002356` (raises CT win probability)
- `lag_11__CT_place_LOWERTUNNEL`: coefficient `-0.002288` (lowers CT win probability)
- `lag_01__T_place_ARAMP`: coefficient `0.002040` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001888` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `5052`, seconds `12.00`, LSTM delta `-0.2810`

Top all feature movements:
- `lag_10__CT_place_LOWERTUNNEL`: contribution `-0.021448`
- `lag_12__T_flashes_last_5s`: contribution `-0.016851`
- `lag_11__CT_place_LOWERTUNNEL`: contribution `-0.016818`
- `lag_00__CT_place_LOWERTUNNEL`: contribution `-0.011680`
- `lag_13__CT_place_ARAMP`: contribution `-0.010987`

Top utility-only movements:
- `lag_12__T_flashes_last_5s`: contribution `-0.016851`

### tick `6524`, seconds `35.00`, LSTM delta `+0.2715`

Top all feature movements:
- `lag_00__T_place_ARAMP`: contribution `+0.029738`
- `lag_02__T_place_ARAMP`: contribution `+0.025736`
- `lag_01__T_place_ARAMP`: contribution `+0.018455`
- `lag_09__T_flashed_players`: contribution `+0.013641`
- `lag_09__CT_flash_duration_sum`: contribution `+0.010865`

Top utility-only movements:
- `lag_09__CT_flash_duration_sum`: contribution `+0.010865`
- `lag_09__CT3__flash_duration`: contribution `+0.008806`
- `lag_09__CT4__flash_duration`: contribution `+0.005749`
- `lag_00__CT2__flash_duration`: contribution `+0.005151`

### tick `6620`, seconds `36.50`, LSTM delta `+0.2462`

Top all feature movements:
- `lag_00__T_place_ARAMP`: contribution `+0.029738`
- `lag_03__T_place_ARAMP`: contribution `+0.015074`
- `lag_09__T_place_ARAMP`: contribution `+0.010937`
- `lag_05__T_place_ARAMP`: contribution `+0.010057`
- `lag_12__CT_flash_duration_sum`: contribution `+0.009721`

Top utility-only movements:
- `lag_12__CT_flash_duration_sum`: contribution `+0.009721`
- `lag_12__CT4__flash_duration`: contribution `+0.006124`
- `lag_12__CT3__flash_duration`: contribution `+0.006093`
- `lag_01__CT3__flash_duration`: contribution `+0.004089`
- `lag_12__CT2__flash_duration`: contribution `+0.003955`

### tick `5468`, seconds `18.50`, LSTM delta `+0.2393`

Top all feature movements:
- `lag_00__CT_place_LOWERTUNNEL`: contribution `+0.011680`
- `lag_03__CT_place_LOWERTUNNEL`: contribution `+0.011639`
- `lag_13__CT_place_LOWERTUNNEL`: contribution `+0.011342`
- `lag_10__T5__flash_duration`: contribution `+0.009918`
- `lag_10__CT5__flash_duration`: contribution `+0.009342`

Top utility-only movements:
- `lag_10__T5__flash_duration`: contribution `+0.009918`
- `lag_10__CT5__flash_duration`: contribution `+0.009342`
- `lag_00__T5__flash_duration`: contribution `+0.004333`

### tick `6588`, seconds `36.00`, LSTM delta `-0.2195`

Top all feature movements:
- `lag_02__T_place_ARAMP`: contribution `-0.025736`
- `lag_03__T_place_ARAMP`: contribution `-0.015074`
- `lag_09__T_flashed_players`: contribution `-0.013641`
- `lag_04__T_place_ARAMP`: contribution `+0.009239`
- `lag_08__T_place_ARAMP`: contribution `-0.008559`

Top utility-only movements:
- `lag_01__CT4__flash_duration`: contribution `-0.004967`
- `lag_11__CT_flash_duration_sum`: contribution `-0.004924`
- `lag_00__CT3__flash_duration`: contribution `-0.004018`
- `lag_02__CT2__flash_duration`: contribution `-0.003452`
