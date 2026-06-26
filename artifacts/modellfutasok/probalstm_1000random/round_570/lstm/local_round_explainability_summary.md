# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-falcons-vs-3dmax-bo3-XHM3Ovc8L9TfLFTYQFrGdT/falcons-vs-3dmax-m3-dust2.csv`
- round_num: `1`

## Largest probability jumps

- tick `2906`, seconds `25.50`, LSTM `0.4165`, delta `+0.2652`
- tick `2970`, seconds `26.50`, LSTM `0.1608`, delta `-0.2520`
- tick `2842`, seconds `24.50`, LSTM `0.2475`, delta `-0.2353`
- tick `2874`, seconds `25.00`, LSTM `0.1513`, delta `-0.0963`
- tick `3002`, seconds `27.00`, LSTM `0.1055`, delta `-0.0553`
- tick `3130`, seconds `29.00`, LSTM `0.0449`, delta `-0.0252`
- tick `2778`, seconds `23.50`, LSTM `0.4749`, delta `+0.0211`
- tick `3098`, seconds `28.50`, LSTM `0.0701`, delta `-0.0189`
- tick `2618`, seconds `21.00`, LSTM `0.4769`, delta `-0.0188`
- tick `2266`, seconds `15.50`, LSTM `0.5044`, delta `+0.0181`

## Top 15 local ridge features

- `lag_09__CT_place_ARAMP`: coefficient `0.002913`, |coef| `0.002913`
- `lag_07__CT_place_ARAMP`: coefficient `-0.002208`, |coef| `0.002208`
- `lag_09__CT_place_LONGA`: coefficient `-0.002127`, |coef| `0.002127`
- `lag_00__T1__flash_duration`: coefficient `-0.002094`, |coef| `0.002094`
- `lag_02__T1__flash_duration`: coefficient `0.002081`, |coef| `0.002081`
- `lag_06__CT_place_ARAMP`: coefficient `0.002010`, |coef| `0.002010`
- `lag_02__T_flashed_players`: coefficient `0.002010`, |coef| `0.002010`
- `lag_00__T_flashed_players`: coefficient `-0.001920`, |coef| `0.001920`
- `lag_07__CT1__flash_duration`: coefficient `-0.001688`, |coef| `0.001688`
- `lag_00__T_kills_last_3s`: coefficient `-0.001627`, |coef| `0.001627`
- `lag_00__kill_diff_last_3s`: coefficient `0.001617`, |coef| `0.001617`
- `lag_08__CT_place_ARAMP`: coefficient `-0.001615`, |coef| `0.001615`
- `lag_11__CT_place_ARAMP`: coefficient `-0.001607`, |coef| `0.001607`
- `lag_04__T5__flash_duration`: coefficient `-0.001584`, |coef| `0.001584`
- `lag_03__CT_place_ARAMP`: coefficient `0.001570`, |coef| `0.001570`

## Top 10 utility ridge features

- `lag_00__T1__flash_duration`: coefficient `-0.002094` (lowers CT win probability)
- `lag_02__T1__flash_duration`: coefficient `0.002081` (raises CT win probability)
- `lag_07__CT1__flash_duration`: coefficient `-0.001688` (lowers CT win probability)
- `lag_04__T5__flash_duration`: coefficient `-0.001584` (lowers CT win probability)
- `lag_02__CT1__flash_duration`: coefficient `-0.001504` (lowers CT win probability)
- `lag_11__CT1__flash_duration`: coefficient `-0.001439` (lowers CT win probability)
- `lag_03__T5__flash_duration`: coefficient `-0.001437` (lowers CT win probability)
- `lag_00__T_flash_duration_sum`: coefficient `-0.001270` (lowers CT win probability)
- `lag_04__CT1__flash_duration`: coefficient `0.001253` (raises CT win probability)
- `lag_01__T1__flash_duration`: coefficient `0.001241` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_09__CT_place_ARAMP`: coefficient `0.002913` (raises CT win probability)
- `lag_07__CT_place_ARAMP`: coefficient `-0.002208` (lowers CT win probability)
- `lag_09__CT_place_LONGA`: coefficient `-0.002127` (lowers CT win probability)
- `lag_06__CT_place_ARAMP`: coefficient `0.002010` (raises CT win probability)
- `lag_02__T_flashed_players`: coefficient `0.002010` (raises CT win probability)
- `lag_00__T_flashed_players`: coefficient `-0.001920` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001627` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001617` (raises CT win probability)
- `lag_08__CT_place_ARAMP`: coefficient `-0.001615` (lowers CT win probability)
- `lag_11__CT_place_ARAMP`: coefficient `-0.001607` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `2906`, seconds `25.50`, LSTM delta `+0.2652`

Top all feature movements:
- `lag_09__CT_place_ARAMP`: contribution `+0.018147`
- `lag_07__CT_place_ARAMP`: contribution `+0.013756`
- `lag_06__CT_place_ARAMP`: contribution `+0.012519`
- `lag_00__T1__flash_duration`: contribution `+0.010511`
- `lag_02__CT1__flash_duration`: contribution `+0.008475`

Top utility-only movements:
- `lag_00__T1__flash_duration`: contribution `+0.010511`
- `lag_02__CT1__flash_duration`: contribution `+0.008475`
- `lag_09__CT1__flash_duration`: contribution `+0.006710`
- `lag_01__T5__flash_duration`: contribution `+0.006578`
- `lag_01__T1__flash_duration`: contribution `+0.006232`

### tick `2970`, seconds `26.50`, LSTM delta `-0.2520`

Top all feature movements:
- `lag_09__CT_place_ARAMP`: contribution `-0.018147`
- `lag_07__CT_place_ARAMP`: contribution `+0.013756`
- `lag_02__T1__flash_duration`: contribution `-0.010449`
- `lag_08__CT_place_ARAMP`: contribution `-0.010062`
- `lag_11__CT_place_ARAMP`: contribution `-0.010012`

Top utility-only movements:
- `lag_02__T1__flash_duration`: contribution `-0.010449`
- `lag_11__CT1__flash_duration`: contribution `-0.008105`
- `lag_03__T5__flash_duration`: contribution `-0.007658`
- `lag_04__CT1__flash_duration`: contribution `-0.007058`
- `lag_03__T_flash_duration_sum`: contribution `-0.004002`

### tick `2842`, seconds `24.50`, LSTM delta `-0.2353`

Top all feature movements:
- `lag_07__CT_place_ARAMP`: contribution `-0.013756`
- `lag_03__CT_place_ARAMP`: contribution `-0.009780`
- `lag_07__CT1__flash_duration`: contribution `-0.009510`
- `lag_02__T_flashed_players`: contribution `-0.007755`
- `lag_01__CT_place_BDOORS`: contribution `-0.007307`

Top utility-only movements:
- `lag_07__CT1__flash_duration`: contribution `-0.009510`
- `lag_00__CT1__flash_duration`: contribution `-0.006720`
- `lag_02__T1__flash_duration`: contribution `-0.005803`

### tick `2874`, seconds `25.00`, LSTM delta `-0.0963`

Top all feature movements:
- `lag_06__CT_place_ARAMP`: contribution `-0.012519`
- `lag_00__T_flashed_players`: contribution `-0.011115`
- `lag_00__T1__flash_duration`: contribution `-0.010511`
- `lag_08__CT_place_ARAMP`: contribution `-0.010062`
- `lag_04__CT_place_ARAMP`: contribution `+0.007044`

Top utility-only movements:
- `lag_00__T1__flash_duration`: contribution `-0.010511`
- `lag_00__T_flash_duration_sum`: contribution `-0.005791`
- `lag_01__CT1__flash_duration`: contribution `-0.004841`
- `lag_08__CT1__flash_duration`: contribution `-0.004302`
- `lag_03__T5__flash_duration`: contribution `+0.004267`

### tick `3002`, seconds `27.00`, LSTM delta `-0.0553`

Top all feature movements:
- `lag_09__CT_place_ARAMP`: contribution `+0.018147`
- `lag_08__CT_place_ARAMP`: contribution `+0.010062`
- `lag_04__T_flashed_players`: contribution `-0.008602`
- `lag_04__T5__flash_duration`: contribution `-0.008438`
- `lag_09__CT_place_LONGA`: contribution `+0.005682`

Top utility-only movements:
- `lag_04__T5__flash_duration`: contribution `-0.008438`
- `lag_04__T1__flash_duration`: contribution `-0.004853`
- `lag_04__T_flash_duration_sum`: contribution `-0.004794`
- `lag_12__CT1__flash_duration`: contribution `-0.004109`
- `lag_03__T1__flash_duration`: contribution `+0.003060`
