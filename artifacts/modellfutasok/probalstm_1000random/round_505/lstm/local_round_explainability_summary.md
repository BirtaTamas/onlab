# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-3dmax-vs-betboom-anubis-9yOMu3EhAmKzkIxUzvijXH/3dmax-vs-betboom-anubis.csv`
- round_num: `7`

## Largest probability jumps

- tick `48920`, seconds `73.50`, LSTM `0.2708`, delta `-0.1412`
- tick `49048`, seconds `75.50`, LSTM `0.0495`, delta `-0.1171`
- tick `48952`, seconds `74.00`, LSTM `0.2104`, delta `-0.0604`
- tick `48984`, seconds `74.50`, LSTM `0.1764`, delta `-0.0339`
- tick `45976`, seconds `27.50`, LSTM `0.4978`, delta `-0.0329`
- tick `46072`, seconds `29.00`, LSTM `0.4693`, delta `-0.0310`
- tick `47256`, seconds `47.50`, LSTM `0.4761`, delta `+0.0291`
- tick `48856`, seconds `72.50`, LSTM `0.4121`, delta `-0.0232`
- tick `45912`, seconds `26.50`, LSTM `0.5472`, delta `+0.0232`
- tick `48760`, seconds `71.00`, LSTM `0.4317`, delta `-0.0219`

## Top 15 local ridge features

- `lag_00__CT_place_CTSIDEUPPER`: coefficient `0.001295`, |coef| `0.001295`
- `lag_04__CT_place_FOUNTAIN`: coefficient `0.001185`, |coef| `0.001185`
- `lag_00__CT2__flash_duration`: coefficient `-0.001176`, |coef| `0.001176`
- `lag_03__CT2__flash_duration`: coefficient `-0.001164`, |coef| `0.001164`
- `lag_01__T_place_WALKWAY`: coefficient `-0.001148`, |coef| `0.001148`
- `lag_14__CT_place_FOUNTAIN`: coefficient `-0.001024`, |coef| `0.001024`
- `lag_07__T_place_WALKWAY`: coefficient `-0.000993`, |coef| `0.000993`
- `lag_12__T_place_BRIDGE`: coefficient `0.000944`, |coef| `0.000944`
- `lag_04__CT_place_MAIN`: coefficient `0.000862`, |coef| `0.000862`
- `lag_01__CT2__flash_duration`: coefficient `-0.000854`, |coef| `0.000854`
- `lag_00__CT2__duck_amount`: coefficient `-0.000813`, |coef| `0.000813`
- `lag_00__T_kills_last_3s`: coefficient `-0.000772`, |coef| `0.000772`
- `lag_03__T_place_WALKWAY`: coefficient `-0.000771`, |coef| `0.000771`
- `lag_04__CT2__flash_duration`: coefficient `-0.000761`, |coef| `0.000761`
- `lag_02__CT2__flash_duration`: coefficient `-0.000755`, |coef| `0.000755`

## Top 10 utility ridge features

- `lag_00__CT2__flash_duration`: coefficient `-0.001176` (lowers CT win probability)
- `lag_03__CT2__flash_duration`: coefficient `-0.001164` (lowers CT win probability)
- `lag_01__CT2__flash_duration`: coefficient `-0.000854` (lowers CT win probability)
- `lag_04__CT2__flash_duration`: coefficient `-0.000761` (lowers CT win probability)
- `lag_02__CT2__flash_duration`: coefficient `-0.000755` (lowers CT win probability)
- `lag_05__CT2__flash_duration`: coefficient `-0.000616` (lowers CT win probability)
- `lag_10__CT2__flash_duration`: coefficient `-0.000612` (lowers CT win probability)
- `lag_09__CT2__flash_duration`: coefficient `-0.000607` (lowers CT win probability)
- `lag_06__CT2__flash_duration`: coefficient `-0.000578` (lowers CT win probability)
- `lag_12__CT2__flash_duration`: coefficient `-0.000566` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_CTSIDEUPPER`: coefficient `0.001295` (raises CT win probability)
- `lag_04__CT_place_FOUNTAIN`: coefficient `0.001185` (raises CT win probability)
- `lag_01__T_place_WALKWAY`: coefficient `-0.001148` (lowers CT win probability)
- `lag_14__CT_place_FOUNTAIN`: coefficient `-0.001024` (lowers CT win probability)
- `lag_07__T_place_WALKWAY`: coefficient `-0.000993` (lowers CT win probability)
- `lag_12__T_place_BRIDGE`: coefficient `0.000944` (raises CT win probability)
- `lag_04__CT_place_MAIN`: coefficient `0.000862` (raises CT win probability)
- `lag_00__CT2__duck_amount`: coefficient `-0.000813` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.000772` (lowers CT win probability)
- `lag_03__T_place_WALKWAY`: coefficient `-0.000771` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `48920`, seconds `73.50`, LSTM delta `-0.1412`

Top all feature movements:
- `lag_01__T_place_WALKWAY`: contribution `-0.015609`
- `lag_07__T_place_WALKWAY`: contribution `-0.013503`
- `lag_04__CT_place_FOUNTAIN`: contribution `-0.012467`
- `lag_14__CT_place_FOUNTAIN`: contribution `-0.010771`
- `lag_14__CT_place_MAIN`: contribution `-0.004806`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `49048`, seconds `75.50`, LSTM delta `-0.1171`

Top all feature movements:
- `lag_03__T_place_WALKWAY`: contribution `-0.010488`
- `lag_11__T_place_WALKWAY`: contribution `-0.009023`
- `lag_05__T_place_WALKWAY`: contribution `-0.008762`
- `lag_03__CT2__flash_duration`: contribution `-0.007238`
- `lag_08__CT_place_FOUNTAIN`: contribution `-0.007064`

Top utility-only movements:
- `lag_03__CT2__flash_duration`: contribution `-0.007238`

### tick `48952`, seconds `74.00`, LSTM delta `-0.0604`

Top all feature movements:
- `lag_02__T_place_WALKWAY`: contribution `-0.009727`
- `lag_00__CT2__flash_duration`: contribution `-0.007313`
- `lag_15__CT_place_FOUNTAIN`: contribution `-0.006793`
- `lag_00__T_place_WALKWAY`: contribution `-0.004774`
- `lag_05__CT_place_FOUNTAIN`: contribution `-0.004683`

Top utility-only movements:
- `lag_00__CT2__flash_duration`: contribution `-0.007313`
- `lag_00__T2__flash_duration`: contribution `-0.001160`

### tick `48984`, seconds `74.50`, LSTM delta `-0.0339`

Top all feature movements:
- `lag_01__T_place_WALKWAY`: contribution `-0.015609`
- `lag_03__T_place_WALKWAY`: contribution `-0.010488`
- `lag_04__CT_place_MAIN`: contribution `+0.005806`
- `lag_01__CT2__flash_duration`: contribution `-0.005309`
- `lag_09__T_place_WALKWAY`: contribution `-0.004955`

Top utility-only movements:
- `lag_01__CT2__flash_duration`: contribution `-0.005309`
- `lag_01__T2__flash_duration`: contribution `-0.000697`

### tick `45976`, seconds `27.50`, LSTM delta `-0.0329`

Top all feature movements:
- `lag_12__T_place_BRIDGE`: contribution `-0.004089`
- `lag_15__T2__flash_duration`: contribution `-0.002152`
- `lag_01__T1__duck_amount`: contribution `+0.002122`
- `lag_00__CT_shots_fired_sum`: contribution `-0.002103`
- `lag_10__CT4__flash_duration`: contribution `-0.002042`

Top utility-only movements:
- `lag_15__T2__flash_duration`: contribution `-0.002152`
- `lag_10__CT4__flash_duration`: contribution `-0.002042`
- `lag_12__CT_utility_damage_last_5s`: contribution `-0.001463`
- `lag_12__utility_damage_diff_last_5s`: contribution `-0.001039`
