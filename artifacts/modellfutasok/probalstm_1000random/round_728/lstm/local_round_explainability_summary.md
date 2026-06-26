# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-lynn-vision-vs-housebets-bo3-GrWDn9AJOxYQcZMXkSI-Tw/lynn-vision-vs-housebets-m2-dust2.csv`
- round_num: `6`

## Largest probability jumps

- tick `53865`, seconds `69.00`, LSTM `0.0483`, delta `-0.1523`
- tick `53545`, seconds `64.00`, LSTM `0.1164`, delta `+0.1047`
- tick `53609`, seconds `65.00`, LSTM `0.2336`, delta `+0.0591`
- tick `53577`, seconds `64.50`, LSTM `0.1745`, delta `+0.0581`
- tick `53737`, seconds `67.00`, LSTM `0.1989`, delta `-0.0574`
- tick `51881`, seconds `38.00`, LSTM `0.0327`, delta `-0.0471`
- tick `50825`, seconds `21.50`, LSTM `0.1507`, delta `+0.0454`
- tick `49481`, seconds `0.50`, LSTM `0.1116`, delta `-0.0365`
- tick `49801`, seconds `5.50`, LSTM `0.0789`, delta `-0.0347`
- tick `53993`, seconds `71.00`, LSTM `0.0067`, delta `-0.0278`

## Top 15 local ridge features

- `lag_00__bomb_events_last_5s`: coefficient `0.001203`, |coef| `0.001203`
- `lag_10__T4__flash_duration`: coefficient `-0.001039`, |coef| `0.001039`
- `lag_03__T_place_SHORTSTAIRS`: coefficient `-0.001031`, |coef| `0.001031`
- `lag_10__T2__flash_duration`: coefficient `-0.001031`, |coef| `0.001031`
- `lag_12__bomb_events_last_5s`: coefficient `0.000997`, |coef| `0.000997`
- `lag_03__T_place_EXTENDEDA`: coefficient `0.000971`, |coef| `0.000971`
- `lag_00__T4__flash_duration`: coefficient `0.000953`, |coef| `0.000953`
- `lag_10__T3__flash_duration`: coefficient `-0.000911`, |coef| `0.000911`
- `lag_10__T_flash_duration_sum`: coefficient `-0.000888`, |coef| `0.000888`
- `lag_00__kill_diff_last_3s`: coefficient `0.000878`, |coef| `0.000878`
- `lag_00__T_shots_fired_sum`: coefficient `-0.000811`, |coef| `0.000811`
- `lag_15__CT_flashed_players`: coefficient `-0.000809`, |coef| `0.000809`
- `lag_15__CT3__flash_duration`: coefficient `-0.000802`, |coef| `0.000802`
- `lag_01__T_place_SHORTSTAIRS`: coefficient `-0.000761`, |coef| `0.000761`
- `lag_11__T_place_EXTENDEDA`: coefficient `-0.000746`, |coef| `0.000746`

## Top 10 utility ridge features

- `lag_10__T4__flash_duration`: coefficient `-0.001039` (lowers CT win probability)
- `lag_10__T2__flash_duration`: coefficient `-0.001031` (lowers CT win probability)
- `lag_00__T4__flash_duration`: coefficient `0.000953` (raises CT win probability)
- `lag_10__T3__flash_duration`: coefficient `-0.000911` (lowers CT win probability)
- `lag_10__T_flash_duration_sum`: coefficient `-0.000888` (lowers CT win probability)
- `lag_15__CT3__flash_duration`: coefficient `-0.000802` (lowers CT win probability)
- `lag_00__T3__flash_duration`: coefficient `0.000735` (raises CT win probability)
- `lag_05__CT3__flash_duration`: coefficient `0.000734` (raises CT win probability)
- `lag_04__CT3__flash_duration`: coefficient `0.000728` (raises CT win probability)
- `lag_00__T_flash_duration_sum`: coefficient `0.000684` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__bomb_events_last_5s`: coefficient `0.001203` (raises CT win probability)
- `lag_03__T_place_SHORTSTAIRS`: coefficient `-0.001031` (lowers CT win probability)
- `lag_12__bomb_events_last_5s`: coefficient `0.000997` (raises CT win probability)
- `lag_03__T_place_EXTENDEDA`: coefficient `0.000971` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000878` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.000811` (lowers CT win probability)
- `lag_15__CT_flashed_players`: coefficient `-0.000809` (lowers CT win probability)
- `lag_01__T_place_SHORTSTAIRS`: coefficient `-0.000761` (lowers CT win probability)
- `lag_11__T_place_EXTENDEDA`: coefficient `-0.000746` (lowers CT win probability)
- `lag_04__T_place_EXTENDEDA`: coefficient `0.000739` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `53865`, seconds `69.00`, LSTM delta `-0.1523`

Top all feature movements:
- `lag_10__T4__flash_duration`: contribution `-0.008093`
- `lag_10__T2__flash_duration`: contribution `-0.007364`
- `lag_10__T3__flash_duration`: contribution `-0.006511`
- `lag_10__T_flash_duration_sum`: contribution `-0.005922`
- `lag_15__CT_flashed_players`: contribution `-0.005318`

Top utility-only movements:
- `lag_10__T4__flash_duration`: contribution `-0.008093`
- `lag_10__T2__flash_duration`: contribution `-0.007364`
- `lag_10__T3__flash_duration`: contribution `-0.006511`
- `lag_10__T_flash_duration_sum`: contribution `-0.005922`
- `lag_15__CT3__flash_duration`: contribution `-0.004811`

### tick `53545`, seconds `64.00`, LSTM delta `+0.1047`

Top all feature movements:
- `lag_00__T4__flash_duration`: contribution `+0.007422`
- `lag_00__T3__flash_duration`: contribution `+0.005248`
- `lag_00__bomb_events_last_5s`: contribution `+0.005026`
- `lag_03__T_place_EXTENDEDA`: contribution `+0.004815`
- `lag_00__T_flash_duration_sum`: contribution `+0.004565`

Top utility-only movements:
- `lag_00__T4__flash_duration`: contribution `+0.007422`
- `lag_00__T3__flash_duration`: contribution `+0.005248`
- `lag_00__T_flash_duration_sum`: contribution `+0.004565`
- `lag_05__CT3__flash_duration`: contribution `+0.004404`
- `lag_00__T2__flash_duration`: contribution `+0.004044`

### tick `53609`, seconds `65.00`, LSTM delta `+0.0591`

Top all feature movements:
- `lag_03__T_place_EXTENDEDA`: contribution `+0.004815`
- `lag_03__T_place_SHORTSTAIRS`: contribution `+0.004334`
- `lag_04__T_place_EXTENDEDA`: contribution `+0.003665`
- `lag_02__T4__flash_duration`: contribution `+0.003421`
- `lag_04__T_place_SHORTSTAIRS`: contribution `+0.003064`

Top utility-only movements:
- `lag_02__T4__flash_duration`: contribution `+0.003421`
- `lag_02__T3__flash_duration`: contribution `+0.002435`
- `lag_07__CT3__flash_duration`: contribution `+0.002304`
- `lag_02__T2__flash_duration`: contribution `+0.002181`
- `lag_02__T_flash_duration_sum`: contribution `+0.002128`

### tick `53577`, seconds `64.50`, LSTM delta `+0.0581`

Top all feature movements:
- `lag_03__T_place_EXTENDEDA`: contribution `+0.004815`
- `lag_03__T_place_SHORTSTAIRS`: contribution `+0.004334`
- `lag_04__T_place_EXTENDEDA`: contribution `+0.003665`
- `lag_04__T_place_SHORTSTAIRS`: contribution `+0.003064`
- `lag_01__T4__flash_duration`: contribution `+0.002904`

Top utility-only movements:
- `lag_01__T4__flash_duration`: contribution `+0.002904`
- `lag_01__T3__flash_duration`: contribution `+0.002276`
- `lag_06__CT3__flash_duration`: contribution `+0.002193`
- `lag_01__T_flash_duration_sum`: contribution `+0.002022`
- `lag_06__CT_flash_duration_sum`: contribution `+0.001270`

### tick `53737`, seconds `67.00`, LSTM delta `-0.0574`

Top all feature movements:
- `lag_06__T3__flash_duration`: contribution `-0.003204`
- `lag_06__T4__flash_duration`: contribution `-0.002967`
- `lag_11__CT3__flash_duration`: contribution `-0.002459`
- `lag_09__CT2__duck_amount`: contribution `-0.002324`
- `lag_01__T3__duck_amount`: contribution `-0.002235`

Top utility-only movements:
- `lag_06__T3__flash_duration`: contribution `-0.003204`
- `lag_06__T4__flash_duration`: contribution `-0.002967`
- `lag_11__CT3__flash_duration`: contribution `-0.002459`
- `lag_06__T_flash_duration_sum`: contribution `-0.001876`
- `lag_00__CT3__flash_duration`: contribution `-0.001756`
