# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-b8-vs-lynn-vision-bo3-Whl3pjYuIoHffY1VOn8vws/b8-vs-lynn-vision-m1-dust2.csv`
- round_num: `7`

## Largest probability jumps

- tick `62454`, seconds `110.00`, LSTM `0.1279`, delta `-0.4366`
- tick `62134`, seconds `105.00`, LSTM `0.7432`, delta `+0.2809`
- tick `61942`, seconds `102.00`, LSTM `0.5140`, delta `+0.2011`
- tick `62326`, seconds `108.00`, LSTM `0.4679`, delta `-0.1486`
- tick `62006`, seconds `103.00`, LSTM `0.4416`, delta `-0.1349`
- tick `62198`, seconds `106.00`, LSTM `0.5799`, delta `-0.1163`
- tick `62390`, seconds `109.00`, LSTM `0.5744`, delta `+0.0854`
- tick `61590`, seconds `96.50`, LSTM `0.2579`, delta `-0.0793`
- tick `62262`, seconds `107.00`, LSTM `0.5611`, delta `-0.0731`
- tick `61974`, seconds `102.50`, LSTM `0.5765`, delta `+0.0625`

## Top 15 local ridge features

- `lag_12__T1__flash_duration`: coefficient `0.002252`, |coef| `0.002252`
- `lag_09__T5__flash_duration`: coefficient `0.002045`, |coef| `0.002045`
- `lag_00__kill_diff_last_3s`: coefficient `0.002029`, |coef| `0.002029`
- `lag_04__T_bomb_zone_count`: coefficient `-0.001996`, |coef| `0.001996`
- `lag_12__T_place_EXTENDEDA`: coefficient `0.001828`, |coef| `0.001828`
- `lag_14__CT_place_BDOORS`: coefficient `0.001814`, |coef| `0.001814`
- `lag_00__damage_diff_last_5s`: coefficient `0.001797`, |coef| `0.001797`
- `lag_07__T_bomb_zone_count`: coefficient `0.001790`, |coef| `0.001790`
- `lag_12__CT_place_HOLE`: coefficient `0.001764`, |coef| `0.001764`
- `lag_06__CT_place_HOLE`: coefficient `0.001761`, |coef| `0.001761`
- `lag_14__T_place_EXTENDEDA`: coefficient `0.001759`, |coef| `0.001759`
- `lag_09__T_bomb_zone_count`: coefficient `-0.001730`, |coef| `0.001730`
- `lag_12__T_flash_duration_sum`: coefficient `0.001622`, |coef| `0.001622`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001616`, |coef| `0.001616`
- `lag_04__CT_duck_amount_mean`: coefficient `0.001564`, |coef| `0.001564`

## Top 10 utility ridge features

- `lag_12__T1__flash_duration`: coefficient `0.002252` (raises CT win probability)
- `lag_09__T5__flash_duration`: coefficient `0.002045` (raises CT win probability)
- `lag_12__T_flash_duration_sum`: coefficient `0.001622` (raises CT win probability)
- `lag_12__T5__flash_duration`: coefficient `0.001413` (raises CT win probability)
- `lag_06__T1__flash_duration`: coefficient `0.001233` (raises CT win probability)
- `lag_02__T1__flash_duration`: coefficient `-0.001184` (lowers CT win probability)
- `lag_10__T_A_site_active_infernos`: coefficient `0.000835` (raises CT win probability)
- `lag_15__T2__flash_duration`: coefficient `0.000822` (raises CT win probability)
- `lag_06__T_flash_duration_sum`: coefficient `0.000799` (raises CT win probability)
- `lag_10__T1__flash_duration`: coefficient `-0.000755` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002029` (raises CT win probability)
- `lag_04__T_bomb_zone_count`: coefficient `-0.001996` (lowers CT win probability)
- `lag_12__T_place_EXTENDEDA`: coefficient `0.001828` (raises CT win probability)
- `lag_14__CT_place_BDOORS`: coefficient `0.001814` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001797` (raises CT win probability)
- `lag_07__T_bomb_zone_count`: coefficient `0.001790` (raises CT win probability)
- `lag_12__CT_place_HOLE`: coefficient `0.001764` (raises CT win probability)
- `lag_06__CT_place_HOLE`: coefficient `0.001761` (raises CT win probability)
- `lag_14__T_place_EXTENDEDA`: coefficient `0.001759` (raises CT win probability)
- `lag_09__T_bomb_zone_count`: coefficient `-0.001730` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `62454`, seconds `110.00`, LSTM delta `-0.4366`

Top all feature movements:
- `lag_09__T5__flash_duration`: contribution `-0.012726`
- `lag_12__T1__flash_duration`: contribution `-0.012475`
- `lag_04__T_bomb_zone_count`: contribution `-0.011622`
- `lag_07__T_bomb_zone_count`: contribution `-0.010420`
- `lag_09__T_bomb_zone_count`: contribution `-0.010071`

Top utility-only movements:
- `lag_09__T5__flash_duration`: contribution `-0.012726`
- `lag_12__T1__flash_duration`: contribution `-0.012475`
- `lag_12__T_flash_duration_sum`: contribution `-0.003739`

### tick `62134`, seconds `105.00`, LSTM delta `+0.2809`

Top all feature movements:
- `lag_12__CT_place_HOLE`: contribution `+0.019693`
- `lag_10__CT_place_HOLE`: contribution `+0.016647`
- `lag_12__T1__flash_duration`: contribution `+0.012475`
- `lag_12__T_flash_duration_sum`: contribution `+0.011780`
- `lag_14__CT_place_BDOORS`: contribution `+0.008729`

Top utility-only movements:
- `lag_12__T1__flash_duration`: contribution `+0.012475`
- `lag_12__T_flash_duration_sum`: contribution `+0.011780`
- `lag_12__T5__flash_duration`: contribution `+0.008194`
- `lag_02__T1__flash_duration`: contribution `+0.006557`

### tick `61942`, seconds `102.00`, LSTM delta `+0.2011`

Top all feature movements:
- `lag_06__CT_place_HOLE`: contribution `+0.019658`
- `lag_15__CT_place_ARAMP`: contribution `+0.014120`
- `lag_11__T_place_SHORTSTAIRS`: contribution `+0.010003`
- `lag_11__T_place_EXTENDEDA`: contribution `+0.009306`
- `lag_04__CT_place_HOLE`: contribution `+0.007679`

Top utility-only movements:
- `lag_06__T1__flash_duration`: contribution `+0.006830`
- `lag_06__T_flash_duration_sum`: contribution `+0.005803`

### tick `62326`, seconds `108.00`, LSTM delta `-0.1486`

Top all feature movements:
- `lag_12__T_place_EXTENDEDA`: contribution `-0.009062`
- `lag_05__T_bomb_zone_count`: contribution `-0.008725`
- `lag_14__T_place_EXTENDEDA`: contribution `-0.008721`
- `lag_00__CT_shots_fired_sum`: contribution `-0.007858`
- `lag_15__CT_place_BDOORS`: contribution `-0.004964`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `62006`, seconds `103.00`, LSTM delta `-0.1349`

Top all feature movements:
- `lag_06__CT_place_HOLE`: contribution `-0.019658`
- `lag_08__CT_place_HOLE`: contribution `-0.008919`
- `lag_11__T_flashed_players`: contribution `-0.006182`
- `lag_02__T_place_EXTENDEDA`: contribution `+0.005206`
- `lag_00__kill_diff_last_3s`: contribution `-0.004884`

Top utility-only movements:
- `lag_08__T_flash_duration_sum`: contribution `-0.003422`
- `lag_12__T5__flash_duration`: contribution `-0.002796`
