# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-faze-vs-inner-circle-bo3-runM3q2zOKSAHTeRui0Q2h/faze-vs-inner-circle-m2-nuke.csv`
- round_num: `4`

## Largest probability jumps

- tick `30037`, seconds `95.00`, LSTM `0.8805`, delta `+0.0932`
- tick `30005`, seconds `94.50`, LSTM `0.7873`, delta `+0.0794`
- tick `24789`, seconds `13.00`, LSTM `0.4914`, delta `+0.0773`
- tick `24885`, seconds `14.50`, LSTM `0.5927`, delta `+0.0678`
- tick `27765`, seconds `59.50`, LSTM `0.6177`, delta `+0.0663`
- tick `30229`, seconds `98.00`, LSTM `0.9533`, delta `+0.0656`
- tick `28725`, seconds `74.50`, LSTM `0.5634`, delta `+0.0571`
- tick `28245`, seconds `67.00`, LSTM `0.5798`, delta `-0.0542`
- tick `29045`, seconds `79.50`, LSTM `0.6378`, delta `+0.0488`
- tick `26037`, seconds `32.50`, LSTM `0.5944`, delta `+0.0487`

## Top 15 local ridge features

- `lag_03__T_place_HELL`: coefficient `0.002022`, |coef| `0.002022`
- `lag_02__T_place_HELL`: coefficient `0.001981`, |coef| `0.001981`
- `lag_05__T_place_HELL`: coefficient `0.001834`, |coef| `0.001834`
- `lag_04__T_place_HELL`: coefficient `0.001814`, |coef| `0.001814`
- `lag_09__T_place_HELL`: coefficient `0.001598`, |coef| `0.001598`
- `lag_07__T_place_HELL`: coefficient `0.001440`, |coef| `0.001440`
- `lag_00__CT_place_HUT`: coefficient `-0.001389`, |coef| `0.001389`
- `lag_06__T_place_HELL`: coefficient `0.001334`, |coef| `0.001334`
- `lag_08__T_place_HELL`: coefficient `0.001332`, |coef| `0.001332`
- `lag_00__CT_place_LOCKERROOM`: coefficient `-0.001227`, |coef| `0.001227`
- `lag_10__CT4__duck_amount`: coefficient `-0.001187`, |coef| `0.001187`
- `lag_10__T_place_HELL`: coefficient `0.001129`, |coef| `0.001129`
- `lag_00__CT_place_VENDING`: coefficient `-0.001115`, |coef| `0.001115`
- `lag_10__CT_place_TROPHY`: coefficient `0.001104`, |coef| `0.001104`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001080`, |coef| `0.001080`

## Top 10 utility ridge features

- `lag_01__T5__flash_duration`: coefficient `0.000571` (raises CT win probability)
- `lag_00__T5__flash_duration`: coefficient `0.000559` (raises CT win probability)
- `lag_00__T5__flash`: coefficient `-0.000462` (lowers CT win probability)
- `lag_10__T5__flash_duration`: coefficient `-0.000449` (lowers CT win probability)
- `lag_02__T5__flash_duration`: coefficient `0.000424` (raises CT win probability)
- `lag_06__T1__flash_duration`: coefficient `0.000420` (raises CT win probability)
- `lag_00__T_flash_duration_sum`: coefficient `0.000412` (raises CT win probability)
- `lag_01__T_flash_duration_sum`: coefficient `0.000401` (raises CT win probability)
- `lag_06__CT4__flash_duration`: coefficient `0.000390` (raises CT win probability)
- `lag_01__T5__flash`: coefficient `-0.000381` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_03__T_place_HELL`: coefficient `0.002022` (raises CT win probability)
- `lag_02__T_place_HELL`: coefficient `0.001981` (raises CT win probability)
- `lag_05__T_place_HELL`: coefficient `0.001834` (raises CT win probability)
- `lag_04__T_place_HELL`: coefficient `0.001814` (raises CT win probability)
- `lag_09__T_place_HELL`: coefficient `0.001598` (raises CT win probability)
- `lag_07__T_place_HELL`: coefficient `0.001440` (raises CT win probability)
- `lag_00__CT_place_HUT`: coefficient `-0.001389` (lowers CT win probability)
- `lag_06__T_place_HELL`: coefficient `0.001334` (raises CT win probability)
- `lag_08__T_place_HELL`: coefficient `0.001332` (raises CT win probability)
- `lag_00__CT_place_LOCKERROOM`: coefficient `-0.001227` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `30037`, seconds `95.00`, LSTM delta `+0.0932`

Top all feature movements:
- `lag_03__T_place_HELL`: contribution `+0.043126`
- `lag_15__T_flashed_players`: contribution `+0.005825`
- `lag_10__CT4__duck_amount`: contribution `+0.004358`
- `lag_13__T4__duck_amount`: contribution `+0.003217`
- `lag_00__CT_kills_last_3s`: contribution `+0.002601`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `30005`, seconds `94.50`, LSTM delta `+0.0794`

Top all feature movements:
- `lag_02__T_place_HELL`: contribution `+0.042238`
- `lag_14__T_flashed_players`: contribution `+0.003738`
- `lag_13__T5__duck_amount`: contribution `+0.003038`
- `lag_09__CT4__duck_amount`: contribution `+0.002661`
- `lag_00__CT_shots_fired_sum`: contribution `+0.002250`

Top utility-only movements:
- `lag_10__T5__flash_duration`: contribution `+0.001249`

### tick `24789`, seconds `13.00`, LSTM delta `+0.0773`

Top all feature movements:
- `lag_00__CT_place_HUT`: contribution `+0.013544`
- `lag_10__CT_place_ADMIN`: contribution `+0.011845`
- `lag_00__CT_place_CONTROL`: contribution `-0.010999`
- `lag_00__CT_place_TROPHY`: contribution `+0.009856`
- `lag_04__CT_place_CONTROL`: contribution `+0.009200`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `24885`, seconds `14.50`, LSTM delta `+0.0678`

Top all feature movements:
- `lag_08__CT_place_CONTROL`: contribution `+0.010118`
- `lag_03__CT_place_TROPHY`: contribution `+0.008012`
- `lag_00__CT_shots_fired_sum`: contribution `+0.006001`
- `lag_07__CT_place_CONTROL`: contribution `+0.005069`
- `lag_01__CT_place_TROPHY`: contribution `-0.004657`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `27765`, seconds `59.50`, LSTM delta `+0.0663`

Top all feature movements:
- `lag_00__CT_place_LOCKERROOM`: contribution `+0.015271`
- `lag_00__CT_place_CONTROL`: contribution `+0.010999`
- `lag_04__CT_place_LOCKERROOM`: contribution `+0.008731`
- `lag_02__CT_place_HUT`: contribution `+0.008239`
- `lag_02__CT_place_LOBBY`: contribution `+0.005553`

Top utility-only movements:
- No utility movement among the top local contributors.
