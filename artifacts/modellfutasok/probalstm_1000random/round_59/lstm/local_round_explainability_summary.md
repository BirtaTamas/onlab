# Local Round Explainability

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-spirit-vs-furia-bo3-_zQK5XUu10iN1JLmPA8zQ4/spirit-vs-furia-m2-nuke.csv`
- round_num: `6`

## Largest probability jumps

- tick `50553`, seconds `87.00`, LSTM `0.8866`, delta `+0.2374`
- tick `48025`, seconds `47.50`, LSTM `0.6340`, delta `-0.1582`
- tick `47033`, seconds `32.00`, LSTM `0.7559`, delta `+0.1529`
- tick `47193`, seconds `34.50`, LSTM `0.6901`, delta `-0.1461`
- tick `48153`, seconds `49.50`, LSTM `0.7525`, delta `+0.1128`
- tick `47961`, seconds `46.50`, LSTM `0.7867`, delta `+0.1089`
- tick `46425`, seconds `22.50`, LSTM `0.7522`, delta `+0.0897`
- tick `50969`, seconds `93.50`, LSTM `0.8463`, delta `-0.0801`
- tick `47929`, seconds `46.00`, LSTM `0.6777`, delta `+0.0702`
- tick `47257`, seconds `35.50`, LSTM `0.5918`, delta `-0.0631`

## Top 15 local ridge features

- `lag_02__T_place_GARAGE`: coefficient `-0.002844`, |coef| `0.002844`
- `lag_00__T_place_HEAVEN`: coefficient `-0.002672`, |coef| `0.002672`
- `lag_07__CT_place_DECON`: coefficient `0.001947`, |coef| `0.001947`
- `lag_00__kill_diff_last_3s`: coefficient `0.001906`, |coef| `0.001906`
- `lag_00__CT_place_CRANE`: coefficient `0.001735`, |coef| `0.001735`
- `lag_15__T_place_HEAVEN`: coefficient `0.001725`, |coef| `0.001725`
- `lag_00__CT_kills_last_3s`: coefficient `0.001573`, |coef| `0.001573`
- `lag_10__CT_place_DECON`: coefficient `-0.001551`, |coef| `0.001551`
- `lag_08__CT_place_LOCKERROOM`: coefficient `0.001499`, |coef| `0.001499`
- `lag_09__T_place_HEAVEN`: coefficient `0.001497`, |coef| `0.001497`
- `lag_08__CT1__duck_amount`: coefficient `0.001478`, |coef| `0.001478`
- `lag_00__T2__is_scoped`: coefficient `0.001384`, |coef| `0.001384`
- `lag_00__damage_diff_last_5s`: coefficient `0.001322`, |coef| `0.001322`
- `lag_14__CT_place_DECON`: coefficient `0.001316`, |coef| `0.001316`
- `lag_09__T3__is_scoped`: coefficient `0.001216`, |coef| `0.001216`

## Top 10 utility ridge features

- `lag_00__T4__molly`: coefficient `-0.000762` (lowers CT win probability)
- `lag_00__T_smokes_last_5s`: coefficient `-0.000707` (lowers CT win probability)
- `lag_08__T_smokes_last_5s`: coefficient `0.000597` (raises CT win probability)
- `lag_00__T4__utility_total`: coefficient `-0.000577` (lowers CT win probability)
- `lag_00__T4__flash`: coefficient `-0.000518` (lowers CT win probability)
- `lag_01__T3__flash_duration`: coefficient `0.000509` (raises CT win probability)
- `lag_11__T_active_smokes`: coefficient `-0.000481` (lowers CT win probability)
- `lag_12__T_smokes_last_5s`: coefficient `-0.000452` (lowers CT win probability)
- `lag_01__T4__molly`: coefficient `-0.000405` (lowers CT win probability)
- `lag_10__T_active_smokes`: coefficient `-0.000399` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_02__T_place_GARAGE`: coefficient `-0.002844` (lowers CT win probability)
- `lag_00__T_place_HEAVEN`: coefficient `-0.002672` (lowers CT win probability)
- `lag_07__CT_place_DECON`: coefficient `0.001947` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001906` (raises CT win probability)
- `lag_00__CT_place_CRANE`: coefficient `0.001735` (raises CT win probability)
- `lag_15__T_place_HEAVEN`: coefficient `0.001725` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001573` (raises CT win probability)
- `lag_10__CT_place_DECON`: coefficient `-0.001551` (lowers CT win probability)
- `lag_08__CT_place_LOCKERROOM`: coefficient `0.001499` (raises CT win probability)
- `lag_09__T_place_HEAVEN`: coefficient `0.001497` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `50553`, seconds `87.00`, LSTM delta `+0.2374`

Top all feature movements:
- `lag_02__T_place_GARAGE`: contribution `+0.034201`
- `lag_00__T_place_HEAVEN`: contribution `+0.032791`
- `lag_15__T_place_HEAVEN`: contribution `+0.021170`
- `lag_09__T_place_HEAVEN`: contribution `+0.018374`
- `lag_10__T_place_HEAVEN`: contribution `+0.010841`

Top utility-only movements:
- `lag_00__T4__molly`: contribution `+0.001661`

### tick `48025`, seconds `47.50`, LSTM delta `-0.1582`

Top all feature movements:
- `lag_02__T_place_GARAGE`: contribution `-0.034201`
- `lag_10__CT_place_DECON`: contribution `-0.024663`
- `lag_03__T_place_GARAGE`: contribution `-0.009595`
- `lag_15__T3__is_scoped`: contribution `-0.006934`
- `lag_10__T3__is_scoped`: contribution `-0.006284`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `47033`, seconds `32.00`, LSTM delta `+0.1529`

Top all feature movements:
- `lag_08__CT_place_LOCKERROOM`: contribution `+0.018664`
- `lag_01__CT_place_LOCKERROOM`: contribution `+0.011685`
- `lag_15__CT_place_VENTS`: contribution `+0.008784`
- `lag_08__CT_place_HUTROOF`: contribution `+0.005792`
- `lag_08__CT1__duck_amount`: contribution `+0.005640`

Top utility-only movements:
- `lag_01__T3__flash_duration`: contribution `+0.002162`

### tick `47193`, seconds `34.50`, LSTM delta `-0.1461`

Top all feature movements:
- `lag_13__CT_place_LOCKERROOM`: contribution `-0.011043`
- `lag_06__CT_place_LOCKERROOM`: contribution `-0.010457`
- `lag_08__CT_place_CONTROL`: contribution `-0.010203`
- `lag_09__T3__is_scoped`: contribution `-0.007798`
- `lag_12__T3__is_scoped`: contribution `-0.004902`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `48153`, seconds `49.50`, LSTM delta `+0.1128`

Top all feature movements:
- `lag_14__CT_place_DECON`: contribution `+0.020928`
- `lag_07__T_place_GARAGE`: contribution `+0.011517`
- `lag_14__T_place_GARAGE`: contribution `+0.007438`
- `lag_03__T3__is_scoped`: contribution `+0.005794`
- `lag_00__T_place_GARAGE`: contribution `+0.004946`

Top utility-only movements:
- No utility movement among the top local contributors.
