# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-b8-vs-lynn-vision-bo3-Whl3pjYuIoHffY1VOn8vws/b8-vs-lynn-vision-m1-dust2.csv`
- round_num: `3`

## Largest probability jumps

- tick `26346`, seconds `89.00`, LSTM `0.4053`, delta `-0.2671`
- tick `25130`, seconds `70.00`, LSTM `0.2394`, delta `+0.1906`
- tick `25098`, seconds `69.50`, LSTM `0.0487`, delta `-0.1709`
- tick `25066`, seconds `69.00`, LSTM `0.2196`, delta `+0.1600`
- tick `26186`, seconds `86.50`, LSTM `0.5578`, delta `+0.1501`
- tick `25770`, seconds `80.00`, LSTM `0.4142`, delta `+0.1232`
- tick `26218`, seconds `87.00`, LSTM `0.6492`, delta `+0.0914`
- tick `26858`, seconds `97.00`, LSTM `0.3120`, delta `-0.0785`
- tick `20682`, seconds `0.50`, LSTM `0.1118`, delta `-0.0781`
- tick `25514`, seconds `76.00`, LSTM `0.3463`, delta `+0.0756`

## Top 15 local ridge features

- `lag_00__T_place_ARAMP`: coefficient `-0.004365`, |coef| `0.004365`
- `lag_00__kill_diff_last_3s`: coefficient `0.002785`, |coef| `0.002785`
- `lag_05__T_place_ARAMP`: coefficient `0.002670`, |coef| `0.002670`
- `lag_01__CT_place_EXTENDEDA`: coefficient `0.002405`, |coef| `0.002405`
- `lag_14__T_place_ARAMP`: coefficient `-0.002376`, |coef| `0.002376`
- `lag_04__T5__flash_duration`: coefficient `0.002166`, |coef| `0.002166`
- `lag_00__CT_duck_amount_mean`: coefficient `0.002070`, |coef| `0.002070`
- `lag_00__T_kills_last_3s`: coefficient `-0.001984`, |coef| `0.001984`
- `lag_06__T_place_ARAMP`: coefficient `0.001864`, |coef| `0.001864`
- `lag_00__CT_place_EXTENDEDA`: coefficient `0.001854`, |coef| `0.001854`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001721`, |coef| `0.001721`
- `lag_00__damage_diff_last_5s`: coefficient `0.001719`, |coef| `0.001719`
- `lag_00__CT_velocity_mean`: coefficient `-0.001672`, |coef| `0.001672`
- `lag_05__T5__flash_duration`: coefficient `0.001639`, |coef| `0.001639`
- `lag_06__T5__flash_duration`: coefficient `0.001564`, |coef| `0.001564`

## Top 10 utility ridge features

- `lag_04__T5__flash_duration`: coefficient `0.002166` (raises CT win probability)
- `lag_05__T5__flash_duration`: coefficient `0.001639` (raises CT win probability)
- `lag_06__T5__flash_duration`: coefficient `0.001564` (raises CT win probability)
- `lag_11__T5__flash_duration`: coefficient `-0.001377` (lowers CT win probability)
- `lag_07__T5__flash_duration`: coefficient `0.001209` (raises CT win probability)
- `lag_05__T1__flash`: coefficient `0.001065` (raises CT win probability)
- `lag_03__T5__flash_duration`: coefficient `0.001011` (raises CT win probability)
- `lag_04__T_flash_duration_sum`: coefficient `0.000815` (raises CT win probability)
- `lag_00__T1__flash`: coefficient `-0.000750` (lowers CT win probability)
- `lag_11__T_flash_duration_sum`: coefficient `-0.000749` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_ARAMP`: coefficient `-0.004365` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002785` (raises CT win probability)
- `lag_05__T_place_ARAMP`: coefficient `0.002670` (raises CT win probability)
- `lag_01__CT_place_EXTENDEDA`: coefficient `0.002405` (raises CT win probability)
- `lag_14__T_place_ARAMP`: coefficient `-0.002376` (lowers CT win probability)
- `lag_00__CT_duck_amount_mean`: coefficient `0.002070` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001984` (lowers CT win probability)
- `lag_06__T_place_ARAMP`: coefficient `0.001864` (raises CT win probability)
- `lag_00__CT_place_EXTENDEDA`: coefficient `0.001854` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001721` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `26346`, seconds `89.00`, LSTM delta `-0.2671`

Top all feature movements:
- `lag_05__T_place_ARAMP`: contribution `-0.024156`
- `lag_14__T_place_ARAMP`: contribution `-0.021502`
- `lag_01__CT_place_EXTENDEDA`: contribution `-0.013502`
- `lag_04__T5__flash_duration`: contribution `-0.009534`
- `lag_00__kill_diff_last_3s`: contribution `-0.006703`

Top utility-only movements:
- `lag_04__T5__flash_duration`: contribution `-0.009534`
- `lag_11__T5__flash_duration`: contribution `-0.006062`
- `lag_05__T1__flash`: contribution `-0.002963`

### tick `25130`, seconds `70.00`, LSTM delta `+0.1906`

Top all feature movements:
- `lag_00__T_place_ARAMP`: contribution `+0.039497`
- `lag_06__T_place_ARAMP`: contribution `+0.016864`
- `lag_00__T_shots_fired_sum`: contribution `+0.011457`
- `lag_01__CT_shots_fired_sum`: contribution `+0.008608`
- `lag_00__kill_diff_last_3s`: contribution `+0.006703`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `25098`, seconds `69.50`, LSTM delta `-0.1709`

Top all feature movements:
- `lag_00__T_place_ARAMP`: contribution `-0.039497`
- `lag_05__T_place_ARAMP`: contribution `+0.024156`
- `lag_00__CT_shots_fired_sum`: contribution `-0.013150`
- `lag_15__T_place_SIDE`: contribution `-0.012652`
- `lag_09__T_place_ARAMP`: contribution `-0.008276`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `25066`, seconds `69.00`, LSTM delta `+0.1600`

Top all feature movements:
- `lag_00__T_place_ARAMP`: contribution `+0.039497`
- `lag_14__T_place_SIDE`: contribution `+0.020712`
- `lag_00__CT_shots_fired_sum`: contribution `+0.008368`
- `lag_04__T_place_ARAMP`: contribution `+0.007638`
- `lag_01__CT_shots_fired_sum`: contribution `+0.007043`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `26186`, seconds `86.50`, LSTM delta `+0.1501`

Top all feature movements:
- `lag_00__T_place_ARAMP`: contribution `+0.039497`
- `lag_05__T_place_ARAMP`: contribution `+0.024156`
- `lag_09__T_place_ARAMP`: contribution `-0.008276`
- `lag_09__CT_place_EXTENDEDA`: contribution `+0.007143`
- `lag_06__T5__flash_duration`: contribution `+0.006885`

Top utility-only movements:
- `lag_06__T5__flash_duration`: contribution `+0.006885`
- `lag_00__T1__flash`: contribution `+0.002086`
