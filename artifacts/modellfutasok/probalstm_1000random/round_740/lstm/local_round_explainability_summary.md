# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-housebets-bo3-NgyLHfqCvYO4WZnaqhUlfi/heroic-vs-housebets-m1-dust2.csv`
- round_num: `6`

## Largest probability jumps

- tick `37811`, seconds `107.50`, LSTM `0.6366`, delta `-0.2299`
- tick `38291`, seconds `115.00`, LSTM `0.8367`, delta `+0.1662`
- tick `38227`, seconds `114.00`, LSTM `0.6585`, delta `+0.0871`
- tick `37875`, seconds `108.50`, LSTM `0.5846`, delta `-0.0797`
- tick `38355`, seconds `116.00`, LSTM `0.9217`, delta `+0.0739`
- tick `37555`, seconds `103.50`, LSTM `0.8852`, delta `-0.0638`
- tick `32851`, seconds `30.00`, LSTM `0.8474`, delta `+0.0612`
- tick `32531`, seconds `25.00`, LSTM `0.7556`, delta `-0.0419`
- tick `38195`, seconds `113.50`, LSTM `0.5714`, delta `-0.0403`
- tick `37395`, seconds `101.00`, LSTM `0.9281`, delta `+0.0390`

## Top 15 local ridge features

- `lag_08__T2__is_scoped`: coefficient `0.002213`, |coef| `0.002213`
- `lag_00__kill_diff_last_3s`: coefficient `0.001739`, |coef| `0.001739`
- `lag_04__T_bomb_zone_count`: coefficient `-0.001301`, |coef| `0.001301`
- `lag_00__CT_kills_last_3s`: coefficient `0.001289`, |coef| `0.001289`
- `lag_14__CT_place_HOLE`: coefficient `0.001282`, |coef| `0.001282`
- `lag_08__CT_place_ARAMP`: coefficient `0.001257`, |coef| `0.001257`
- `lag_04__CT_place_BDOORS`: coefficient `0.001231`, |coef| `0.001231`
- `lag_13__T_bomb_zone_count`: coefficient `0.001231`, |coef| `0.001231`
- `lag_15__CT_flashed_players`: coefficient `-0.001166`, |coef| `0.001166`
- `lag_06__T2__is_scoped`: coefficient `0.001150`, |coef| `0.001150`
- `lag_09__T_kills_last_3s`: coefficient `-0.001126`, |coef| `0.001126`
- `lag_00__CT_place_ARAMP`: coefficient `0.001096`, |coef| `0.001096`
- `lag_08__T_kills_last_3s`: coefficient `-0.001089`, |coef| `0.001089`
- `lag_15__CT_place_MIDDOORS`: coefficient `-0.001077`, |coef| `0.001077`
- `lag_00__damage_diff_last_5s`: coefficient `0.001068`, |coef| `0.001068`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.000704` (lowers CT win probability)
- `lag_15__CT_flash_duration_sum`: coefficient `-0.000635` (lowers CT win probability)
- `lag_15__CT1__flash_duration`: coefficient `-0.000588` (lowers CT win probability)
- `lag_10__CT1__flash_duration`: coefficient `0.000535` (raises CT win probability)
- `lag_15__T5__flash_duration`: coefficient `-0.000513` (lowers CT win probability)
- `lag_10__T5__flash_duration`: coefficient `0.000474` (raises CT win probability)
- `lag_02__T_smokes_last_5s`: coefficient `-0.000454` (lowers CT win probability)
- `lag_12__CT2__flash_duration`: coefficient `0.000436` (raises CT win probability)
- `lag_15__CT2__flash_duration`: coefficient `-0.000435` (lowers CT win probability)
- `lag_01__T_smokes_last_5s`: coefficient `-0.000425` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_08__T2__is_scoped`: coefficient `0.002213` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001739` (raises CT win probability)
- `lag_04__T_bomb_zone_count`: coefficient `-0.001301` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001289` (raises CT win probability)
- `lag_14__CT_place_HOLE`: coefficient `0.001282` (raises CT win probability)
- `lag_08__CT_place_ARAMP`: coefficient `0.001257` (raises CT win probability)
- `lag_04__CT_place_BDOORS`: coefficient `0.001231` (raises CT win probability)
- `lag_13__T_bomb_zone_count`: coefficient `0.001231` (raises CT win probability)
- `lag_15__CT_flashed_players`: coefficient `-0.001166` (lowers CT win probability)
- `lag_06__T2__is_scoped`: coefficient `0.001150` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `37811`, seconds `107.50`, LSTM delta `-0.2299`

Top all feature movements:
- `lag_14__CT_place_HOLE`: contribution `-0.014316`
- `lag_08__CT_place_ARAMP`: contribution `-0.007830`
- `lag_04__CT_place_BDOORS`: contribution `-0.005923`
- `lag_15__CT_flashed_players`: contribution `-0.005107`
- `lag_12__T_place_EXTENDEDA`: contribution `-0.004300`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `38291`, seconds `115.00`, LSTM delta `+0.1662`

Top all feature movements:
- `lag_08__T2__is_scoped`: contribution `+0.019509`
- `lag_04__T_bomb_zone_count`: contribution `+0.007571`
- `lag_13__T_bomb_zone_count`: contribution `+0.007165`
- `lag_02__CT_place_EXTENDEDA`: contribution `+0.005840`
- `lag_00__kill_diff_last_3s`: contribution `+0.004185`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `38227`, seconds `114.00`, LSTM delta `+0.0871`

Top all feature movements:
- `lag_06__T2__is_scoped`: contribution `+0.010134`
- `lag_02__T_bomb_zone_count`: contribution `+0.005219`
- `lag_12__T2__duck_amount`: contribution `+0.003070`
- `lag_11__T_bomb_zone_count`: contribution `+0.002757`
- `lag_13__CT_place_MIDDOORS`: contribution `+0.002681`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `37875`, seconds `108.50`, LSTM delta `-0.0797`

Top all feature movements:
- `lag_00__T_bomb_zone_count`: contribution `-0.004537`
- `lag_12__CT2__is_scoped`: contribution `-0.004341`
- `lag_12__CT_place_BDOORS`: contribution `-0.004268`
- `lag_15__CT2__is_scoped`: contribution `+0.003205`
- `lag_14__CT1__duck_amount`: contribution `-0.002727`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `38355`, seconds `116.00`, LSTM delta `+0.0739`

Top all feature movements:
- `lag_10__T2__is_scoped`: contribution `+0.009027`
- `lag_00__T_flash_alpha_mean`: contribution `+0.004271`
- `lag_00__kill_diff_last_3s`: contribution `+0.004185`
- `lag_00__CT_kills_last_3s`: contribution `+0.003721`
- `lag_00__T_place_LONGA`: contribution `+0.003311`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.004271`
