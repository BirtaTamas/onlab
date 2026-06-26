# Local Round Explainability

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-virtuspro-vs-spirit-bo3-KJqZR5yNeHXaNsc7MGaDWB/virtus-pro-vs-spirit-m1-train.csv`
- round_num: `7`

## Largest probability jumps

- tick `53533`, seconds `65.00`, LSTM `0.9353`, delta `+0.1461`
- tick `53437`, seconds `63.50`, LSTM `0.9174`, delta `+0.1285`
- tick `53501`, seconds `64.50`, LSTM `0.7892`, delta `-0.1273`
- tick `53341`, seconds `62.00`, LSTM `0.8215`, delta `+0.1179`
- tick `49437`, seconds `1.00`, LSTM `0.7518`, delta `-0.0561`
- tick `49565`, seconds `3.00`, LSTM `0.6944`, delta `-0.0397`
- tick `53373`, seconds `62.50`, LSTM `0.7822`, delta `-0.0392`
- tick `53277`, seconds `61.00`, LSTM `0.6913`, delta `-0.0344`
- tick `49405`, seconds `0.50`, LSTM `0.8079`, delta `-0.0331`
- tick `49885`, seconds `8.00`, LSTM `0.7103`, delta `+0.0318`

## Top 15 local ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.001856`, |coef| `0.001856`
- `lag_00__kill_diff_last_3s`: coefficient `0.001068`, |coef| `0.001068`
- `lag_08__CT_flashed_players`: coefficient `0.001045`, |coef| `0.001045`
- `lag_03__T5__flash_duration`: coefficient `0.001000`, |coef| `0.001000`
- `lag_00__T5__flash_duration`: coefficient `0.000993`, |coef| `0.000993`
- `lag_00__CT_kills_last_3s`: coefficient `0.000985`, |coef| `0.000985`
- `lag_08__CT4__flash_duration`: coefficient `0.000939`, |coef| `0.000939`
- `lag_08__T_flashed_players`: coefficient `0.000936`, |coef| `0.000936`
- `lag_01__CT_shots_fired_sum`: coefficient `-0.000933`, |coef| `0.000933`
- `lag_15__CT1__is_walking`: coefficient `0.000911`, |coef| `0.000911`
- `lag_08__T4__flash_duration`: coefficient `0.000909`, |coef| `0.000909`
- `lag_00__damage_diff_last_5s`: coefficient `0.000881`, |coef| `0.000881`
- `lag_00__CT5__is_walking`: coefficient `-0.000878`, |coef| `0.000878`
- `lag_00__CT_walking_count`: coefficient `-0.000871`, |coef| `0.000871`
- `lag_11__CT4__flash_duration`: coefficient `0.000853`, |coef| `0.000853`

## Top 10 utility ridge features

- `lag_03__T5__flash_duration`: coefficient `0.001000` (raises CT win probability)
- `lag_00__T5__flash_duration`: coefficient `0.000993` (raises CT win probability)
- `lag_08__CT4__flash_duration`: coefficient `0.000939` (raises CT win probability)
- `lag_08__T4__flash_duration`: coefficient `0.000909` (raises CT win probability)
- `lag_11__CT4__flash_duration`: coefficient `0.000853` (raises CT win probability)
- `lag_08__CT_flash_duration_sum`: coefficient `0.000801` (raises CT win probability)
- `lag_01__T_mollies_last_5s`: coefficient `-0.000771` (lowers CT win probability)
- `lag_06__T5__flash_duration`: coefficient `0.000767` (raises CT win probability)
- `lag_11__CT1__flash_duration`: coefficient `0.000761` (raises CT win probability)
- `lag_01__T_flashes_last_5s`: coefficient `-0.000729` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.001856` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001068` (raises CT win probability)
- `lag_08__CT_flashed_players`: coefficient `0.001045` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000985` (raises CT win probability)
- `lag_08__T_flashed_players`: coefficient `0.000936` (raises CT win probability)
- `lag_01__CT_shots_fired_sum`: coefficient `-0.000933` (lowers CT win probability)
- `lag_15__CT1__is_walking`: coefficient `0.000911` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000881` (raises CT win probability)
- `lag_00__CT5__is_walking`: coefficient `-0.000878` (lowers CT win probability)
- `lag_00__CT_walking_count`: coefficient `-0.000871` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `53533`, seconds `65.00`, LSTM delta `+0.1461`

Top all feature movements:
- `lag_01__CT_shots_fired_sum`: contribution `+0.008427`
- `lag_00__CT_shots_fired_sum`: contribution `+0.006449`
- `lag_06__T5__flash_duration`: contribution `+0.004879`
- `lag_14__CT4__flash_duration`: contribution `+0.003720`
- `lag_14__CT_flashed_players`: contribution `+0.003619`

Top utility-only movements:
- `lag_06__T5__flash_duration`: contribution `+0.004879`
- `lag_14__CT4__flash_duration`: contribution `+0.003720`
- `lag_14__CT1__flash_duration`: contribution `+0.003532`
- `lag_14__CT_flash_duration_sum`: contribution `+0.003107`
- `lag_06__CT1__flash_duration`: contribution `+0.002188`

### tick `53437`, seconds `63.50`, LSTM delta `+0.1285`

Top all feature movements:
- `lag_03__T5__flash_duration`: contribution `+0.006363`
- `lag_11__CT_flashed_players`: contribution `+0.004892`
- `lag_11__CT1__flash_duration`: contribution `+0.004885`
- `lag_11__CT4__flash_duration`: contribution `+0.004706`
- `lag_11__CT_flash_duration_sum`: contribution `+0.004182`

Top utility-only movements:
- `lag_03__T5__flash_duration`: contribution `+0.006363`
- `lag_11__CT1__flash_duration`: contribution `+0.004885`
- `lag_11__CT4__flash_duration`: contribution `+0.004706`
- `lag_11__CT_flash_duration_sum`: contribution `+0.004182`
- `lag_06__T4__flash_duration`: contribution `+0.002341`

### tick `53501`, seconds `64.50`, LSTM delta `-0.1273`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.016766`
- `lag_01__CT_shots_fired_sum`: contribution `-0.005834`
- `lag_04__CT_shots_fired_sum`: contribution `-0.004546`
- `lag_05__T5__flash_duration`: contribution `-0.004106`
- `lag_13__CT1__flash_duration`: contribution `-0.003587`

Top utility-only movements:
- `lag_05__T5__flash_duration`: contribution `-0.004106`
- `lag_13__CT1__flash_duration`: contribution `-0.003587`
- `lag_08__T4__flash_duration`: contribution `-0.003020`
- `lag_13__CT_flash_duration_sum`: contribution `-0.002662`
- `lag_13__CT4__flash_duration`: contribution `-0.002438`

### tick `53341`, seconds `62.00`, LSTM delta `+0.1179`

Top all feature movements:
- `lag_08__CT_flashed_players`: contribution `+0.006864`
- `lag_00__CT_shots_fired_sum`: contribution `+0.006449`
- `lag_00__T5__flash_duration`: contribution `+0.006321`
- `lag_08__CT4__flash_duration`: contribution `+0.005179`
- `lag_08__CT_flash_duration_sum`: contribution `+0.004683`

Top utility-only movements:
- `lag_00__T5__flash_duration`: contribution `+0.006321`
- `lag_08__CT4__flash_duration`: contribution `+0.005179`
- `lag_08__CT_flash_duration_sum`: contribution `+0.004683`
- `lag_08__CT1__flash_duration`: contribution `+0.004396`
- `lag_08__T4__flash_duration`: contribution `+0.003020`

### tick `49437`, seconds `1.00`, LSTM delta `-0.0561`

Top all feature movements:
- `lag_01__T_mollies_last_5s`: contribution `-0.015854`
- `lag_01__T_flashes_last_5s`: contribution `-0.013220`
- `lag_02__CT_closest_enemy_dist`: contribution `-0.001242`
- `lag_02__T_closest_enemy_dist`: contribution `-0.001148`
- `lag_02__CT_place_CTSPAWN`: contribution `-0.001145`

Top utility-only movements:
- `lag_01__T_mollies_last_5s`: contribution `-0.015854`
- `lag_01__T_flashes_last_5s`: contribution `-0.013220`
- `lag_02__CT_molly_inv`: contribution `-0.000627`
- `lag_01__T1__flash`: contribution `-0.000578`
- `lag_01__T4__molly`: contribution `-0.000536`
