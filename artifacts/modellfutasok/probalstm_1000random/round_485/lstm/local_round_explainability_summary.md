# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-lynn-vision-bo3-6KVULP2-Gxo12lI67V9ZfV/chinggis-warriors-vs-lynn-vision-m3-ancient.csv`
- round_num: `13`

## Largest probability jumps

- tick `113985`, seconds `62.00`, LSTM `0.2404`, delta `-0.2613`
- tick `111937`, seconds `30.00`, LSTM `0.8190`, delta `+0.2172`
- tick `112033`, seconds `31.50`, LSTM `0.5289`, delta `-0.1931`
- tick `111969`, seconds `30.50`, LSTM `0.6517`, delta `-0.1672`
- tick `111681`, seconds `26.00`, LSTM `0.6317`, delta `+0.1159`
- tick `111905`, seconds `29.50`, LSTM `0.6018`, delta `-0.0849`
- tick `114017`, seconds `62.50`, LSTM `0.1582`, delta `-0.0822`
- tick `112001`, seconds `31.00`, LSTM `0.7221`, delta `+0.0703`
- tick `112129`, seconds `33.00`, LSTM `0.4705`, delta `-0.0501`
- tick `112225`, seconds `34.50`, LSTM `0.5680`, delta `+0.0444`

## Top 15 local ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.004391`, |coef| `0.004391`
- `lag_00__kill_diff_last_3s`: coefficient `0.004210`, |coef| `0.004210`
- `lag_11__CT_place_SIDEENTRANCE`: coefficient `-0.003531`, |coef| `0.003531`
- `lag_00__CT_place_RAMP`: coefficient `0.003206`, |coef| `0.003206`
- `lag_00__damage_diff_last_5s`: coefficient `0.003115`, |coef| `0.003115`
- `lag_10__CT2__duck_amount`: coefficient `0.003056`, |coef| `0.003056`
- `lag_14__T_place_TSIDELOWER`: coefficient `-0.003020`, |coef| `0.003020`
- `lag_14__CT5__duck_amount`: coefficient `0.002811`, |coef| `0.002811`
- `lag_12__CT_place_SIDEENTRANCE`: coefficient `-0.002770`, |coef| `0.002770`
- `lag_00__CT2__alive`: coefficient `0.002762`, |coef| `0.002762`
- `lag_00__CT2__has_defuser`: coefficient `0.002480`, |coef| `0.002480`
- `lag_00__CT2__smoke`: coefficient `0.002475`, |coef| `0.002475`
- `lag_00__CT2__hp`: coefficient `0.002438`, |coef| `0.002438`
- `lag_00__T_damage_last_5s`: coefficient `-0.002430`, |coef| `0.002430`
- `lag_10__CT_place_RAMP`: coefficient `0.002349`, |coef| `0.002349`

## Top 10 utility ridge features

- `lag_00__CT2__smoke`: coefficient `0.002475` (raises CT win probability)
- `lag_01__CT2__smoke`: coefficient `0.001757` (raises CT win probability)
- `lag_05__CT3__flash_duration`: coefficient `0.001531` (raises CT win probability)
- `lag_08__T2__flash_duration`: coefficient `-0.001451` (lowers CT win probability)
- `lag_05__T2__flash_duration`: coefficient `0.001374` (raises CT win probability)
- `lag_06__T2__flash_duration`: coefficient `-0.001269` (lowers CT win probability)
- `lag_05__CT_flash_duration_sum`: coefficient `0.001256` (raises CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `0.001242` (raises CT win probability)
- `lag_02__CT2__smoke`: coefficient `0.001119` (raises CT win probability)
- `lag_05__CT4__flash_duration`: coefficient `0.001099` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.004391` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.004210` (raises CT win probability)
- `lag_11__CT_place_SIDEENTRANCE`: coefficient `-0.003531` (lowers CT win probability)
- `lag_00__CT_place_RAMP`: coefficient `0.003206` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003115` (raises CT win probability)
- `lag_10__CT2__duck_amount`: coefficient `0.003056` (raises CT win probability)
- `lag_14__T_place_TSIDELOWER`: coefficient `-0.003020` (lowers CT win probability)
- `lag_14__CT5__duck_amount`: coefficient `0.002811` (raises CT win probability)
- `lag_12__CT_place_SIDEENTRANCE`: coefficient `-0.002770` (lowers CT win probability)
- `lag_00__CT2__alive`: coefficient `0.002762` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `113985`, seconds `62.00`, LSTM delta `-0.2613`

Top all feature movements:
- `lag_11__CT_place_SIDEENTRANCE`: contribution `-0.014213`
- `lag_00__T_kills_last_3s`: contribution `-0.013911`
- `lag_10__CT2__duck_amount`: contribution `-0.011641`
- `lag_14__T_place_TSIDELOWER`: contribution `-0.011319`
- `lag_14__CT5__duck_amount`: contribution `-0.010613`

Top utility-only movements:
- `lag_00__CT2__smoke`: contribution `-0.005368`

### tick `111937`, seconds `30.00`, LSTM delta `+0.2172`

Top all feature movements:
- `lag_14__CT_place_SIDEHALL`: contribution `+0.017373`
- `lag_10__CT2__duck_amount`: contribution `+0.011641`
- `lag_05__CT3__flash_duration`: contribution `+0.011389`
- `lag_00__kill_diff_last_3s`: contribution `+0.010133`
- `lag_05__T2__flash_duration`: contribution `+0.008959`

Top utility-only movements:
- `lag_05__CT3__flash_duration`: contribution `+0.011389`
- `lag_05__T2__flash_duration`: contribution `+0.008959`
- `lag_05__CT_flash_duration_sum`: contribution `+0.007895`
- `lag_05__CT4__flash_duration`: contribution `+0.005688`

### tick `112033`, seconds `31.50`, LSTM delta `-0.1931`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.013911`
- `lag_00__kill_diff_last_3s`: contribution `-0.010133`
- `lag_08__T2__flash_duration`: contribution `-0.009461`
- `lag_08__CT_flashed_players`: contribution `-0.007935`
- `lag_08__CT2__duck_amount`: contribution `-0.007654`

Top utility-only movements:
- `lag_08__T2__flash_duration`: contribution `-0.009461`
- `lag_08__CT3__flash_duration`: contribution `-0.007485`
- `lag_08__CT_flash_duration_sum`: contribution `-0.006623`
- `lag_00__CT4__flash_duration`: contribution `-0.006428`
- `lag_08__CT4__flash_duration`: contribution `-0.005457`

### tick `111969`, seconds `30.50`, LSTM delta `-0.1672`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.013911`
- `lag_15__CT_place_SIDEHALL`: contribution `-0.012861`
- `lag_00__kill_diff_last_3s`: contribution `-0.010133`
- `lag_06__T2__flash_duration`: contribution `-0.008277`
- `lag_06__CT3__flash_duration`: contribution `-0.007974`

Top utility-only movements:
- `lag_06__T2__flash_duration`: contribution `-0.008277`
- `lag_06__CT3__flash_duration`: contribution `-0.007974`
- `lag_06__CT_flash_duration_sum`: contribution `-0.005708`
- `lag_06__CT4__flash_duration`: contribution `-0.005250`

### tick `111681`, seconds `26.00`, LSTM delta `+0.1159`

Top all feature movements:
- `lag_06__CT_place_SIDEHALL`: contribution `+0.012983`
- `lag_00__kill_diff_last_3s`: contribution `+0.010133`
- `lag_08__CT2__duck_amount`: contribution `+0.007654`
- `lag_12__T5__duck_amount`: contribution `+0.005103`
- `lag_00__damage_diff_last_5s`: contribution `+0.004989`

Top utility-only movements:
- No utility movement among the top local contributors.
