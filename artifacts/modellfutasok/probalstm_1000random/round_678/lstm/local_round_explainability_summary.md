# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-vitality-vs-hotu-bo3-g2oB3RySVGugmKq6yJcHo9/vitality-vs-hotu-m2-dust2.csv`
- round_num: `16`

## Largest probability jumps

- tick `108181`, seconds `37.00`, LSTM `0.4158`, delta `-0.3344`
- tick `108245`, seconds `38.00`, LSTM `0.5529`, delta `+0.2415`
- tick `107637`, seconds `28.50`, LSTM `0.7926`, delta `-0.1284`
- tick `108949`, seconds `49.00`, LSTM `0.9086`, delta `+0.1242`
- tick `108213`, seconds `37.50`, LSTM `0.3114`, delta `-0.1045`
- tick `108533`, seconds `42.50`, LSTM `0.6861`, delta `+0.0791`
- tick `108405`, seconds `40.50`, LSTM `0.5627`, delta `+0.0691`
- tick `107669`, seconds `29.00`, LSTM `0.7401`, delta `-0.0524`
- tick `109461`, seconds `57.00`, LSTM `0.8366`, delta `-0.0501`
- tick `108117`, seconds `36.00`, LSTM `0.7534`, delta `-0.0492`

## Top 15 local ridge features

- `lag_02__CT_place_OUTSIDETUNNEL`: coefficient `-0.004340`, |coef| `0.004340`
- `lag_06__T_place_PIT`: coefficient `-0.002141`, |coef| `0.002141`
- `lag_00__kill_diff_last_3s`: coefficient `0.002038`, |coef| `0.002038`
- `lag_10__CT_place_EXTENDEDA`: coefficient `0.001958`, |coef| `0.001958`
- `lag_15__CT_place_UPPERTUNNEL`: coefficient `-0.001906`, |coef| `0.001906`
- `lag_10__T5__is_scoped`: coefficient `-0.001889`, |coef| `0.001889`
- `lag_05__T_place_PIT`: coefficient `-0.001878`, |coef| `0.001878`
- `lag_02__CT_place_UPPERTUNNEL`: coefficient `0.001560`, |coef| `0.001560`
- `lag_00__CT_kills_last_3s`: coefficient `0.001536`, |coef| `0.001536`
- `lag_06__T_place_LONGDOORS`: coefficient `0.001500`, |coef| `0.001500`
- `lag_03__CT_place_OUTSIDETUNNEL`: coefficient `-0.001443`, |coef| `0.001443`
- `lag_15__T_place_PIT`: coefficient `-0.001440`, |coef| `0.001440`
- `lag_04__T4__duck_amount`: coefficient `-0.001413`, |coef| `0.001413`
- `lag_00__T_place_PIT`: coefficient `-0.001397`, |coef| `0.001397`
- `lag_04__CT_place_OUTSIDETUNNEL`: coefficient `0.001370`, |coef| `0.001370`

## Top 10 utility ridge features

- `lag_14__T5__smoke`: coefficient `-0.000838` (lowers CT win probability)
- `lag_00__CT1__utility_total`: coefficient `0.000646` (raises CT win probability)
- `lag_13__CT_B_site_active_smokes`: coefficient `-0.000628` (lowers CT win probability)
- `lag_00__T_smokes_last_5s`: coefficient `-0.000617` (lowers CT win probability)
- `lag_02__T_smokes_last_5s`: coefficient `-0.000600` (lowers CT win probability)
- `lag_00__T_flashes_last_5s`: coefficient `-0.000577` (lowers CT win probability)
- `lag_00__CT1__molly`: coefficient `0.000571` (raises CT win probability)
- `lag_07__T_smokes_last_5s`: coefficient `0.000559` (raises CT win probability)
- `lag_12__CT2__flash_duration`: coefficient `-0.000554` (lowers CT win probability)
- `lag_01__CT1__flash_duration`: coefficient `0.000550` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_02__CT_place_OUTSIDETUNNEL`: coefficient `-0.004340` (lowers CT win probability)
- `lag_06__T_place_PIT`: coefficient `-0.002141` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002038` (raises CT win probability)
- `lag_10__CT_place_EXTENDEDA`: coefficient `0.001958` (raises CT win probability)
- `lag_15__CT_place_UPPERTUNNEL`: coefficient `-0.001906` (lowers CT win probability)
- `lag_10__T5__is_scoped`: coefficient `-0.001889` (lowers CT win probability)
- `lag_05__T_place_PIT`: coefficient `-0.001878` (lowers CT win probability)
- `lag_02__CT_place_UPPERTUNNEL`: coefficient `0.001560` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001536` (raises CT win probability)
- `lag_06__T_place_LONGDOORS`: coefficient `0.001500` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `108181`, seconds `37.00`, LSTM delta `-0.3344`

Top all feature movements:
- `lag_02__CT_place_OUTSIDETUNNEL`: contribution `-0.093361`
- `lag_15__CT_place_UPPERTUNNEL`: contribution `-0.014620`
- `lag_06__T_place_PIT`: contribution `-0.013513`
- `lag_06__T_place_LONGDOORS`: contribution `-0.012063`
- `lag_02__CT_place_UPPERTUNNEL`: contribution `-0.011964`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `108245`, seconds `38.00`, LSTM delta `+0.2415`

Top all feature movements:
- `lag_04__CT_place_OUTSIDETUNNEL`: contribution `+0.029480`
- `lag_10__T5__is_scoped`: contribution `+0.009008`
- `lag_00__T_place_PIT`: contribution `+0.008813`
- `lag_08__T_place_LONGDOORS`: contribution `+0.008490`
- `lag_04__CT_place_UPPERTUNNEL`: contribution `+0.007847`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `107637`, seconds `28.50`, LSTM delta `-0.1284`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.004906`
- `lag_09__T1__flash_duration`: contribution `-0.003500`
- `lag_00__T_kills_last_3s`: contribution `-0.003162`
- `lag_01__CT1__flash_duration`: contribution `-0.003137`
- `lag_08__CT_shots_fired_sum`: contribution `-0.003116`

Top utility-only movements:
- `lag_09__T1__flash_duration`: contribution `-0.003500`
- `lag_01__CT1__flash_duration`: contribution `-0.003137`
- `lag_11__CT1__flash_duration`: contribution `-0.002646`
- `lag_04__CT2__flash_duration`: contribution `-0.002212`
- `lag_11__CT2__flash_duration`: contribution `-0.002164`

### tick `108949`, seconds `49.00`, LSTM delta `+0.1242`

Top all feature movements:
- `lag_10__CT_place_EXTENDEDA`: contribution `+0.010994`
- `lag_00__kill_diff_last_3s`: contribution `+0.004906`
- `lag_10__CT4__duck_amount`: contribution `+0.004796`
- `lag_13__CT_velocity_mean`: contribution `+0.004444`
- `lag_00__CT_kills_last_3s`: contribution `+0.004433`

Top utility-only movements:
- `lag_14__T5__smoke`: contribution `+0.001816`

### tick `108213`, seconds `37.50`, LSTM delta `-0.1045`

Top all feature movements:
- `lag_03__CT_place_OUTSIDETUNNEL`: contribution `-0.031042`
- `lag_06__T_place_PIT`: contribution `-0.013513`
- `lag_07__T_place_LONGA`: contribution `-0.010975`
- `lag_06__CT_place_ARAMP`: contribution `-0.006899`
- `lag_03__CT_place_UPPERTUNNEL`: contribution `-0.005800`

Top utility-only movements:
- No utility movement among the top local contributors.
