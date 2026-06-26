# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-mouz-vs-pain-bo3-Ao8EIC0rxvFkpkJ5bGImFu/mouz-vs-pain-m3-dust2.csv`
- round_num: `3`

## Largest probability jumps

- tick `16370`, seconds `14.00`, LSTM `0.0698`, delta `-0.1614`
- tick `16306`, seconds `13.00`, LSTM `0.3226`, delta `-0.1214`
- tick `16242`, seconds `12.00`, LSTM `0.3590`, delta `+0.0975`
- tick `16338`, seconds `13.50`, LSTM `0.2312`, delta `-0.0913`
- tick `16274`, seconds `12.50`, LSTM `0.4440`, delta `+0.0850`
- tick `15506`, seconds `0.50`, LSTM `0.2142`, delta `-0.0634`
- tick `16722`, seconds `19.50`, LSTM `0.0350`, delta `-0.0252`
- tick `15954`, seconds `7.50`, LSTM `0.2093`, delta `+0.0228`
- tick `15538`, seconds `1.00`, LSTM `0.1923`, delta `-0.0219`
- tick `16754`, seconds `20.00`, LSTM `0.0138`, delta `-0.0212`

## Top 15 local ridge features

- `lag_11__CT_place_BDOORS`: coefficient `0.002070`, |coef| `0.002070`
- `lag_01__CT_place_LONGDOORS`: coefficient `0.001164`, |coef| `0.001164`
- `lag_01__CT_shots_fired_sum`: coefficient `0.001039`, |coef| `0.001039`
- `lag_14__CT_place_BDOORS`: coefficient `-0.001016`, |coef| `0.001016`
- `lag_02__CT_place_LONGDOORS`: coefficient `0.001002`, |coef| `0.001002`
- `lag_03__T_flashed_players`: coefficient `0.000946`, |coef| `0.000946`
- `lag_00__T_velocity_mean`: coefficient `-0.000900`, |coef| `0.000900`
- `lag_00__kill_diff_last_3s`: coefficient `0.000864`, |coef| `0.000864`
- `lag_01__CT2__duck_amount`: coefficient `0.000857`, |coef| `0.000857`
- `lag_00__T_kills_last_3s`: coefficient `-0.000830`, |coef| `0.000830`
- `lag_14__T_place_LONGDOORS`: coefficient `-0.000805`, |coef| `0.000805`
- `lag_15__T_place_LONGDOORS`: coefficient `-0.000796`, |coef| `0.000796`
- `lag_07__CT1__flash_duration`: coefficient `0.000794`, |coef| `0.000794`
- `lag_10__CT_place_BDOORS`: coefficient `0.000793`, |coef| `0.000793`
- `lag_06__T_flashed_players`: coefficient `-0.000772`, |coef| `0.000772`

## Top 10 utility ridge features

- `lag_07__CT1__flash_duration`: coefficient `0.000794` (raises CT win probability)
- `lag_11__CT2__flash_duration`: coefficient `-0.000756` (lowers CT win probability)
- `lag_07__CT2__flash_duration`: coefficient `0.000741` (raises CT win probability)
- `lag_01__CT2__flash_duration`: coefficient `0.000740` (raises CT win probability)
- `lag_06__CT1__flash_duration`: coefficient `0.000740` (raises CT win probability)
- `lag_02__CT2__flash_duration`: coefficient `0.000725` (raises CT win probability)
- `lag_07__CT_flash_duration_sum`: coefficient `0.000697` (raises CT win probability)
- `lag_11__CT1__flash_duration`: coefficient `-0.000673` (lowers CT win probability)
- `lag_05__CT1__flash_duration`: coefficient `0.000657` (raises CT win probability)
- `lag_09__CT2__flash_duration`: coefficient `-0.000653` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_11__CT_place_BDOORS`: coefficient `0.002070` (raises CT win probability)
- `lag_01__CT_place_LONGDOORS`: coefficient `0.001164` (raises CT win probability)
- `lag_01__CT_shots_fired_sum`: coefficient `0.001039` (raises CT win probability)
- `lag_14__CT_place_BDOORS`: coefficient `-0.001016` (lowers CT win probability)
- `lag_02__CT_place_LONGDOORS`: coefficient `0.001002` (raises CT win probability)
- `lag_03__T_flashed_players`: coefficient `0.000946` (raises CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000900` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000864` (raises CT win probability)
- `lag_01__CT2__duck_amount`: coefficient `0.000857` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.000830` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `16370`, seconds `14.00`, LSTM delta `-0.1614`

Top all feature movements:
- `lag_11__CT_place_BDOORS`: contribution `-0.009957`
- `lag_00__T_shots_fired_sum`: contribution `-0.008089`
- `lag_14__CT_place_BDOORS`: contribution `-0.004888`
- `lag_06__T_flashed_players`: contribution `-0.004468`
- `lag_02__CT_place_LONGDOORS`: contribution `-0.004390`

Top utility-only movements:
- `lag_11__CT2__flash_duration`: contribution `-0.004368`
- `lag_02__CT2__flash_duration`: contribution `-0.004183`
- `lag_11__CT1__flash_duration`: contribution `-0.003263`
- `lag_11__CT_flash_duration_sum`: contribution `-0.003078`
- `lag_06__CT1__flash_duration`: contribution `-0.003024`

### tick `16306`, seconds `13.00`, LSTM delta `-0.1214`

Top all feature movements:
- `lag_11__CT_place_BDOORS`: contribution `-0.009957`
- `lag_01__CT_shots_fired_sum`: contribution `-0.005051`
- `lag_09__CT2__flash_duration`: contribution `-0.003769`
- `lag_00__CT2__flash_duration`: contribution `-0.003756`
- `lag_03__T_flashed_players`: contribution `-0.003649`

Top utility-only movements:
- `lag_09__CT2__flash_duration`: contribution `-0.003769`
- `lag_00__CT2__flash_duration`: contribution `-0.003756`
- `lag_09__CT1__flash_duration`: contribution `-0.002642`
- `lag_09__CT_flash_duration_sum`: contribution `-0.002576`
- `lag_04__CT1__flash_duration`: contribution `-0.002419`

### tick `16242`, seconds `12.00`, LSTM delta `+0.0975`

Top all feature movements:
- `lag_11__CT_place_BDOORS`: contribution `+0.009957`
- `lag_01__CT_place_LONGDOORS`: contribution `+0.005096`
- `lag_07__CT2__flash_duration`: contribution `+0.004278`
- `lag_07__CT1__flash_duration`: contribution `+0.003850`
- `lag_10__CT_place_BDOORS`: contribution `+0.003815`

Top utility-only movements:
- `lag_07__CT2__flash_duration`: contribution `+0.004278`
- `lag_07__CT1__flash_duration`: contribution `+0.003850`
- `lag_07__CT_flash_duration_sum`: contribution `+0.003347`
- `lag_02__CT1__flash_duration`: contribution `+0.002050`

### tick `16338`, seconds `13.50`, LSTM delta `-0.0913`

Top all feature movements:
- `lag_01__CT_place_LONGDOORS`: contribution `-0.005096`
- `lag_14__CT_place_BDOORS`: contribution `-0.004888`
- `lag_01__CT2__flash_duration`: contribution `-0.004274`
- `lag_10__CT_place_BDOORS`: contribution `-0.003815`
- `lag_10__CT2__flash_duration`: contribution `-0.003712`

Top utility-only movements:
- `lag_01__CT2__flash_duration`: contribution `-0.004274`
- `lag_10__CT2__flash_duration`: contribution `-0.003712`
- `lag_10__CT1__flash_duration`: contribution `-0.002763`
- `lag_05__CT1__flash_duration`: contribution `-0.002686`
- `lag_10__CT_flash_duration_sum`: contribution `-0.002600`

### tick `16274`, seconds `12.50`, LSTM delta `+0.0850`

Top all feature movements:
- `lag_11__CT_place_BDOORS`: contribution `+0.009957`
- `lag_03__T_flashed_players`: contribution `+0.005474`
- `lag_01__CT_shots_fired_sum`: contribution `+0.005051`
- `lag_02__CT_place_LONGDOORS`: contribution `+0.004390`
- `lag_10__CT_place_BDOORS`: contribution `-0.003815`

Top utility-only movements:
- `lag_08__CT2__flash_duration`: contribution `+0.002720`
- `lag_08__CT1__flash_duration`: contribution `+0.002453`
- `lag_08__CT_flash_duration_sum`: contribution `+0.002139`
