# Local Round Explainability

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-spirit-vs-astralis-bo3-GZVTrKsE-zdG9dH6juITei/spirit-vs-astralis-m1-nuke.csv`
- round_num: `5`

## Largest probability jumps

- tick `38130`, seconds `70.00`, LSTM `0.1203`, delta `-0.2201`
- tick `38098`, seconds `69.50`, LSTM `0.3404`, delta `-0.1737`
- tick `37746`, seconds `64.00`, LSTM `0.6895`, delta `+0.1446`
- tick `37842`, seconds `65.50`, LSTM `0.5217`, delta `-0.0988`
- tick `37810`, seconds `65.00`, LSTM `0.6205`, delta `-0.0709`
- tick `38162`, seconds `70.50`, LSTM `0.0696`, delta `-0.0506`
- tick `38290`, seconds `72.50`, LSTM `0.0176`, delta `-0.0467`
- tick `38194`, seconds `71.00`, LSTM `0.0507`, delta `-0.0189`
- tick `36530`, seconds `45.00`, LSTM `0.5693`, delta `+0.0187`
- tick `38258`, seconds `72.00`, LSTM `0.0643`, delta `+0.0186`

## Top 15 local ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.002195`, |coef| `0.002195`
- `lag_00__kill_diff_last_3s`: coefficient `0.002140`, |coef| `0.002140`
- `lag_07__CT_place_ADMIN`: coefficient `-0.002102`, |coef| `0.002102`
- `lag_06__CT_place_ADMIN`: coefficient `-0.001767`, |coef| `0.001767`
- `lag_00__CT_place_ADMIN`: coefficient `0.001634`, |coef| `0.001634`
- `lag_01__T_kills_last_3s`: coefficient `-0.001349`, |coef| `0.001349`
- `lag_15__T_flashed_players`: coefficient `0.001287`, |coef| `0.001287`
- `lag_08__CT_place_ADMIN`: coefficient `-0.001277`, |coef| `0.001277`
- `lag_12__T3__is_scoped`: coefficient `0.001221`, |coef| `0.001221`
- `lag_14__CT5__duck_amount`: coefficient `0.001202`, |coef| `0.001202`
- `lag_01__kill_diff_last_3s`: coefficient `0.001194`, |coef| `0.001194`
- `lag_02__CT_place_HEAVEN`: coefficient `0.001167`, |coef| `0.001167`
- `lag_01__T_shots_fired_sum`: coefficient `-0.001160`, |coef| `0.001160`
- `lag_14__T_place_GARAGE`: coefficient `0.001154`, |coef| `0.001154`
- `lag_06__kill_diff_last_3s`: coefficient `0.001087`, |coef| `0.001087`

## Top 10 utility ridge features

- `lag_15__T5__flash_duration`: coefficient `0.001085` (raises CT win probability)
- `lag_12__CT3__flash_duration`: coefficient `-0.001034` (lowers CT win probability)
- `lag_11__T3__flash`: coefficient `0.000998` (raises CT win probability)
- `lag_12__T3__flash`: coefficient `0.000985` (raises CT win probability)
- `lag_00__CT5__molly`: coefficient `0.000958` (raises CT win probability)
- `lag_00__CT3__smoke`: coefficient `0.000879` (raises CT win probability)
- `lag_10__T5__flash_duration`: coefficient `0.000879` (raises CT win probability)
- `lag_14__CT3__flash_duration`: coefficient `0.000844` (raises CT win probability)
- `lag_09__CT1__molly`: coefficient `0.000784` (raises CT win probability)
- `lag_15__CT3__flash_duration`: coefficient `0.000783` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.002195` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002140` (raises CT win probability)
- `lag_07__CT_place_ADMIN`: coefficient `-0.002102` (lowers CT win probability)
- `lag_06__CT_place_ADMIN`: coefficient `-0.001767` (lowers CT win probability)
- `lag_00__CT_place_ADMIN`: coefficient `0.001634` (raises CT win probability)
- `lag_01__T_kills_last_3s`: coefficient `-0.001349` (lowers CT win probability)
- `lag_15__T_flashed_players`: coefficient `0.001287` (raises CT win probability)
- `lag_08__CT_place_ADMIN`: coefficient `-0.001277` (lowers CT win probability)
- `lag_12__T3__is_scoped`: coefficient `0.001221` (raises CT win probability)
- `lag_14__CT5__duck_amount`: coefficient `0.001202` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `38130`, seconds `70.00`, LSTM delta `-0.2201`

Top all feature movements:
- `lag_07__CT_place_ADMIN`: contribution `-0.014605`
- `lag_00__CT_place_ADMIN`: contribution `-0.011351`
- `lag_00__T_kills_last_3s`: contribution `-0.006954`
- `lag_01__T_shots_fired_sum`: contribution `-0.006088`
- `lag_03__CT_place_HEAVEN`: contribution `-0.005746`

Top utility-only movements:
- `lag_12__T3__flash`: contribution `-0.002904`

### tick `38098`, seconds `69.50`, LSTM delta `-0.1737`

Top all feature movements:
- `lag_06__CT_place_ADMIN`: contribution `-0.012279`
- `lag_00__T_kills_last_3s`: contribution `-0.006954`
- `lag_02__CT_place_HEAVEN`: contribution `-0.006303`
- `lag_15__T3__is_scoped`: contribution `-0.005309`
- `lag_00__kill_diff_last_3s`: contribution `-0.005150`

Top utility-only movements:
- `lag_11__T3__flash`: contribution `-0.002942`
- `lag_00__CT5__molly`: contribution `-0.002376`

### tick `37746`, seconds `64.00`, LSTM delta `+0.1446`

Top all feature movements:
- `lag_14__T_place_GARAGE`: contribution `+0.013870`
- `lag_12__CT3__flash_duration`: contribution `+0.008601`
- `lag_12__T3__is_scoped`: contribution `+0.007831`
- `lag_15__T_flashed_players`: contribution `+0.007448`
- `lag_15__T_place_SILO`: contribution `+0.006864`

Top utility-only movements:
- `lag_12__CT3__flash_duration`: contribution `+0.008601`
- `lag_15__T5__flash_duration`: contribution `+0.005174`
- `lag_07__T5__flash_duration`: contribution `+0.003289`
- `lag_12__CT_flash_duration_sum`: contribution `+0.001853`
- `lag_15__T_flash_duration_sum`: contribution `+0.001721`

### tick `37842`, seconds `65.50`, LSTM delta `-0.0988`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.006954`
- `lag_15__CT3__flash_duration`: contribution `-0.006515`
- `lag_15__T3__is_scoped`: contribution `-0.005309`
- `lag_00__kill_diff_last_3s`: contribution `-0.005150`
- `lag_13__T3__is_scoped`: contribution `-0.004501`

Top utility-only movements:
- `lag_15__CT3__flash_duration`: contribution `-0.006515`
- `lag_10__T5__flash_duration`: contribution `-0.004193`
- `lag_00__CT1__molly`: contribution `-0.001356`
- `lag_15__CT_flash_duration_sum`: contribution `-0.001352`

### tick `37810`, seconds `65.00`, LSTM delta `-0.0709`

Top all feature movements:
- `lag_12__T3__is_scoped`: contribution `-0.007831`
- `lag_14__CT3__flash_duration`: contribution `-0.007026`
- `lag_05__T3__is_scoped`: contribution `-0.004527`
- `lag_09__T5__flash_duration`: contribution `-0.003630`
- `lag_14__CT4__duck_amount`: contribution `-0.003404`

Top utility-only movements:
- `lag_14__CT3__flash_duration`: contribution `-0.007026`
- `lag_09__T5__flash_duration`: contribution `-0.003630`
- `lag_14__CT_flash_duration_sum`: contribution `-0.001314`
