# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-furia-vs-saw-bo3-hxORpk_jCtMpGRLo1Voi3p/furia-vs-saw-m2-dust2.csv`
- round_num: `13`

## Largest probability jumps

- tick `113377`, seconds `19.50`, LSTM `0.5508`, delta `+0.1551`
- tick `113857`, seconds `27.00`, LSTM `0.1533`, delta `-0.0994`
- tick `113729`, seconds `25.00`, LSTM `0.3495`, delta `-0.0921`
- tick `113313`, seconds `18.50`, LSTM `0.3986`, delta `-0.0897`
- tick `113409`, seconds `20.00`, LSTM `0.4923`, delta `-0.0585`
- tick `114145`, seconds `31.50`, LSTM `0.0297`, delta `-0.0579`
- tick `113697`, seconds `24.50`, LSTM `0.4416`, delta `-0.0567`
- tick `113761`, seconds `25.50`, LSTM `0.3021`, delta `-0.0474`
- tick `113921`, seconds `28.00`, LSTM `0.0801`, delta `-0.0450`
- tick `113825`, seconds `26.50`, LSTM `0.2527`, delta `-0.0401`

## Top 15 local ridge features

- `lag_07__CT_place_HOLE`: coefficient `-0.001727`, |coef| `0.001727`
- `lag_02__T_flashed_players`: coefficient `0.001217`, |coef| `0.001217`
- `lag_02__T3__flash_duration`: coefficient `0.001204`, |coef| `0.001204`
- `lag_02__CT_flashed_players`: coefficient `0.001076`, |coef| `0.001076`
- `lag_11__CT_flashed_players`: coefficient `0.001032`, |coef| `0.001032`
- `lag_06__T_place_LONGA`: coefficient `-0.001016`, |coef| `0.001016`
- `lag_00__CT_flashed_players`: coefficient `-0.001014`, |coef| `0.001014`
- `lag_05__CT2__duck_amount`: coefficient `-0.000971`, |coef| `0.000971`
- `lag_09__CT_place_LOWERTUNNEL`: coefficient `-0.000948`, |coef| `0.000948`
- `lag_02__CT_place_EXTENDEDA`: coefficient `-0.000943`, |coef| `0.000943`
- `lag_02__T_flash_duration_sum`: coefficient `0.000916`, |coef| `0.000916`
- `lag_02__T5__flash_duration`: coefficient `0.000909`, |coef| `0.000909`
- `lag_00__damage_diff_last_5s`: coefficient `0.000901`, |coef| `0.000901`
- `lag_00__kill_diff_last_3s`: coefficient `0.000886`, |coef| `0.000886`
- `lag_05__CT_place_SHORTSTAIRS`: coefficient `-0.000867`, |coef| `0.000867`

## Top 10 utility ridge features

- `lag_02__T3__flash_duration`: coefficient `0.001204` (raises CT win probability)
- `lag_02__T_flash_duration_sum`: coefficient `0.000916` (raises CT win probability)
- `lag_02__T5__flash_duration`: coefficient `0.000909` (raises CT win probability)
- `lag_10__CT1__flash_duration`: coefficient `-0.000772` (lowers CT win probability)
- `lag_01__CT1__flash_duration`: coefficient `-0.000662` (lowers CT win probability)
- `lag_04__T1__flash_duration`: coefficient `-0.000654` (lowers CT win probability)
- `lag_12__T2__flash`: coefficient `0.000654` (raises CT win probability)
- `lag_13__T4__flash_duration`: coefficient `-0.000651` (lowers CT win probability)
- `lag_04__T3__flash_duration`: coefficient `0.000646` (raises CT win probability)
- `lag_13__T5__flash_duration`: coefficient `-0.000570` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_07__CT_place_HOLE`: coefficient `-0.001727` (lowers CT win probability)
- `lag_02__T_flashed_players`: coefficient `0.001217` (raises CT win probability)
- `lag_02__CT_flashed_players`: coefficient `0.001076` (raises CT win probability)
- `lag_11__CT_flashed_players`: coefficient `0.001032` (raises CT win probability)
- `lag_06__T_place_LONGA`: coefficient `-0.001016` (lowers CT win probability)
- `lag_00__CT_flashed_players`: coefficient `-0.001014` (lowers CT win probability)
- `lag_05__CT2__duck_amount`: coefficient `-0.000971` (lowers CT win probability)
- `lag_09__CT_place_LOWERTUNNEL`: coefficient `-0.000948` (lowers CT win probability)
- `lag_02__CT_place_EXTENDEDA`: coefficient `-0.000943` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000901` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `113377`, seconds `19.50`, LSTM delta `+0.1551`

Top all feature movements:
- `lag_07__CT_place_HOLE`: contribution `+0.019278`
- `lag_02__T_flashed_players`: contribution `+0.009393`
- `lag_02__T3__flash_duration`: contribution `+0.007818`
- `lag_02__CT_flashed_players`: contribution `+0.007066`
- `lag_09__CT_place_LOWERTUNNEL`: contribution `+0.006965`

Top utility-only movements:
- `lag_02__T3__flash_duration`: contribution `+0.007818`
- `lag_02__T5__flash_duration`: contribution `+0.006323`
- `lag_02__T_flash_duration_sum`: contribution `+0.005993`
- `lag_10__CT1__flash_duration`: contribution `+0.003391`

### tick `113857`, seconds `27.00`, LSTM delta `-0.0994`

Top all feature movements:
- `lag_05__CT_place_SHORTSTAIRS`: contribution `-0.004835`
- `lag_11__CT_flashed_players`: contribution `-0.004521`
- `lag_01__T_place_ARAMP`: contribution `-0.003442`
- `lag_08__T1__flash_duration`: contribution `-0.003116`
- `lag_00__T_kills_last_3s`: contribution `-0.002485`

Top utility-only movements:
- `lag_08__T1__flash_duration`: contribution `-0.003116`
- `lag_13__T4__flash_duration`: contribution `-0.001607`
- `lag_13__CT1__flash_duration`: contribution `-0.001455`
- `lag_04__T3__flash_duration`: contribution `-0.001444`

### tick `113729`, seconds `25.00`, LSTM delta `-0.0921`

Top all feature movements:
- `lag_11__CT_flashed_players`: contribution `-0.004521`
- `lag_13__T5__flash_duration`: contribution `-0.003962`
- `lag_04__T1__flash_duration`: contribution `-0.003623`
- `lag_13__T_flashed_players`: contribution `-0.002978`
- `lag_13__T3__flash_duration`: contribution `-0.002502`

Top utility-only movements:
- `lag_13__T5__flash_duration`: contribution `-0.003962`
- `lag_04__T1__flash_duration`: contribution `-0.003623`
- `lag_13__T3__flash_duration`: contribution `-0.002502`
- `lag_13__T_flash_duration_sum`: contribution `-0.002114`

### tick `113313`, seconds `18.50`, LSTM delta `-0.0897`

Top all feature movements:
- `lag_07__CT_place_HOLE`: contribution `-0.019278`
- `lag_05__CT_place_HOLE`: contribution `-0.007991`
- `lag_00__CT_flashed_players`: contribution `-0.006664`
- `lag_06__T_place_LONGA`: contribution `-0.004329`
- `lag_05__CT_place_BDOORS`: contribution `-0.004172`

Top utility-only movements:
- `lag_00__T3__flash_duration`: contribution `-0.002004`

### tick `113409`, seconds `20.00`, LSTM delta `-0.0585`

Top all feature movements:
- `lag_10__CT_place_HOLE`: contribution `-0.005746`
- `lag_05__T_place_LONGA`: contribution `-0.003483`
- `lag_08__CT_place_HOLE`: contribution `-0.002706`
- `lag_00__T_kills_last_3s`: contribution `-0.002485`
- `lag_00__T5__duck_amount`: contribution `-0.002457`

Top utility-only movements:
- `lag_03__T5__flash_duration`: contribution `-0.001891`
- `lag_03__T_flash_duration_sum`: contribution `-0.001705`
