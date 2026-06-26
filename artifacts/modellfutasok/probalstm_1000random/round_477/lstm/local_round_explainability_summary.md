# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-vitality-vs-gamerlegion-bo3-HfAhqHTEhpe_HlObeToa76/vitality-vs-gamerlegion-m1-overpass.csv`
- round_num: `1`

## Largest probability jumps

- tick `3777`, seconds `39.50`, LSTM `0.5547`, delta `+0.2639`
- tick `2561`, seconds `20.50`, LSTM `0.1349`, delta `-0.2573`
- tick `4001`, seconds `43.00`, LSTM `0.8393`, delta `+0.2312`
- tick `3233`, seconds `31.00`, LSTM `0.3778`, delta `+0.2263`
- tick `3041`, seconds `28.00`, LSTM `0.2226`, delta `+0.1450`
- tick `3489`, seconds `35.00`, LSTM `0.3502`, delta `+0.1387`
- tick `3297`, seconds `32.00`, LSTM `0.3271`, delta `-0.1256`
- tick `3553`, seconds `36.00`, LSTM `0.2831`, delta `-0.1083`
- tick `3265`, seconds `31.50`, LSTM `0.4527`, delta `+0.0749`
- tick `2593`, seconds `21.00`, LSTM `0.0778`, delta `-0.0571`

## Top 15 local ridge features

- `lag_00__T_place_LOWERPARK`: coefficient `-0.003332`, |coef| `0.003332`
- `lag_03__T_place_RESTROOM`: coefficient `0.003280`, |coef| `0.003280`
- `lag_09__CT_place_STORAGEROOM`: coefficient `-0.002745`, |coef| `0.002745`
- `lag_07__T_place_LOWERPARK`: coefficient `-0.002100`, |coef| `0.002100`
- `lag_00__damage_diff_last_5s`: coefficient `0.002094`, |coef| `0.002094`
- `lag_14__CT_place_LOBBY`: coefficient `0.002039`, |coef| `0.002039`
- `lag_07__CT_place_RESTROOM`: coefficient `-0.002019`, |coef| `0.002019`
- `lag_05__T_place_LOWERPARK`: coefficient `-0.001993`, |coef| `0.001993`
- `lag_09__CT_place_CONSTRUCTION`: coefficient `-0.001957`, |coef| `0.001957`
- `lag_15__CT_place_STORAGEROOM`: coefficient `-0.001946`, |coef| `0.001946`
- `lag_00__kill_diff_last_3s`: coefficient `0.001936`, |coef| `0.001936`
- `lag_07__CT_place_STORAGEROOM`: coefficient `-0.001877`, |coef| `0.001877`
- `lag_11__CT_place_STORAGEROOM`: coefficient `-0.001842`, |coef| `0.001842`
- `lag_05__CT_place_STORAGEROOM`: coefficient `0.001702`, |coef| `0.001702`
- `lag_00__CT_kills_last_3s`: coefficient `0.001700`, |coef| `0.001700`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.001405` (lowers CT win probability)
- `lag_00__T3__flash_duration`: coefficient `-0.001385` (lowers CT win probability)
- `lag_11__CT2__flash_duration`: coefficient `0.001008` (raises CT win probability)
- `lag_07__T3__flash_duration`: coefficient `0.000954` (raises CT win probability)
- `lag_01__CT2__flash_duration`: coefficient `-0.000954` (lowers CT win probability)
- `lag_08__T4__flash_duration`: coefficient `0.000938` (raises CT win probability)
- `lag_00__T_flash_duration_sum`: coefficient `-0.000877` (lowers CT win probability)
- `lag_01__T4__flash_duration`: coefficient `0.000820` (raises CT win probability)
- `lag_12__CT2__flash_duration`: coefficient `0.000798` (raises CT win probability)
- `lag_08__CT_utility_damage_last_5s`: coefficient `-0.000790` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_LOWERPARK`: coefficient `-0.003332` (lowers CT win probability)
- `lag_03__T_place_RESTROOM`: coefficient `0.003280` (raises CT win probability)
- `lag_09__CT_place_STORAGEROOM`: coefficient `-0.002745` (lowers CT win probability)
- `lag_07__T_place_LOWERPARK`: coefficient `-0.002100` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002094` (raises CT win probability)
- `lag_14__CT_place_LOBBY`: coefficient `0.002039` (raises CT win probability)
- `lag_07__CT_place_RESTROOM`: coefficient `-0.002019` (lowers CT win probability)
- `lag_05__T_place_LOWERPARK`: coefficient `-0.001993` (lowers CT win probability)
- `lag_09__CT_place_CONSTRUCTION`: coefficient `-0.001957` (lowers CT win probability)
- `lag_15__CT_place_STORAGEROOM`: coefficient `-0.001946` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `3777`, seconds `39.50`, LSTM delta `+0.2639`

Top all feature movements:
- `lag_09__CT_place_STORAGEROOM`: contribution `+0.058716`
- `lag_15__CT_place_STORAGEROOM`: contribution `+0.041620`
- `lag_14__CT_place_STORAGEROOM`: contribution `+0.023865`
- `lag_14__CT_place_BACKOFA`: contribution `+0.016011`
- `lag_09__CT_place_LOBBY`: contribution `+0.013787`

Top utility-only movements:
- `lag_01__CT_utility_damage_last_5s`: contribution `+0.002661`
- `lag_11__CT_utility_damage_last_5s`: contribution `+0.002637`

### tick `2561`, seconds `20.50`, LSTM delta `-0.2573`

Top all feature movements:
- `lag_07__CT_place_RESTROOM`: contribution `-0.028786`
- `lag_09__CT_place_CONSTRUCTION`: contribution `-0.024613`
- `lag_00__CT_place_RESTROOM`: contribution `-0.021669`
- `lag_05__T_place_LOWERPARK`: contribution `-0.016067`
- `lag_03__CT_place_CONSTRUCTION`: contribution `-0.015380`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `4001`, seconds `43.00`, LSTM delta `+0.2312`

Top all feature movements:
- `lag_03__T_place_RESTROOM`: contribution `+0.063266`
- `lag_01__T_place_RESTROOM`: contribution `+0.024562`
- `lag_00__T_place_LOWERPARK`: contribution `+0.013432`
- `lag_07__T_place_LOWERPARK`: contribution `+0.008465`
- `lag_03__T_place_UPPERPARK`: contribution `+0.006766`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.004261`
- `lag_08__CT_utility_damage_last_5s`: contribution `+0.002781`

### tick `3233`, seconds `31.00`, LSTM delta `+0.2263`

Top all feature movements:
- `lag_05__CT_place_STORAGEROOM`: contribution `+0.036412`
- `lag_00__T_place_LOWERPARK`: contribution `+0.013432`
- `lag_00__T3__flash_duration`: contribution `+0.010516`
- `lag_07__CT_place_STAIRS`: contribution `+0.008470`
- `lag_07__T_place_LOWERPARK`: contribution `+0.008465`

Top utility-only movements:
- `lag_00__T3__flash_duration`: contribution `+0.010516`
- `lag_07__T3__flash_duration`: contribution `+0.007244`
- `lag_11__CT2__flash_duration`: contribution `+0.005738`
- `lag_01__CT2__flash_duration`: contribution `+0.005429`
- `lag_07__T_flash_duration_sum`: contribution `+0.003826`

### tick `3041`, seconds `28.00`, LSTM delta `+0.1450`

Top all feature movements:
- `lag_15__CT_place_RESTROOM`: contribution `+0.023650`
- `lag_12__CT_place_BRIDGE`: contribution `+0.014345`
- `lag_14__CT_place_BRIDGE`: contribution `+0.011851`
- `lag_06__CT_place_BACKOFA`: contribution `-0.009078`
- `lag_06__CT_place_STAIRS`: contribution `+0.007194`

Top utility-only movements:
- `lag_01__T4__flash_duration`: contribution `+0.004626`
- `lag_00__T4__flash_duration`: contribution `+0.004242`
