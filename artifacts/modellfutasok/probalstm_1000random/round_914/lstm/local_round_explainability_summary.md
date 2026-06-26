# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-falcons-bo3-yayytstbo8IxTFlUpfbUPR/mouz-vs-falcons-m1-train.csv`
- round_num: `4`

## Largest probability jumps

- tick `23671`, seconds `91.50`, LSTM `0.4598`, delta `+0.2567`
- tick `24119`, seconds `98.50`, LSTM `0.7872`, delta `+0.2208`
- tick `23607`, seconds `90.50`, LSTM `0.2807`, delta `-0.1954`
- tick `23703`, seconds `92.00`, LSTM `0.6404`, delta `+0.1806`
- tick `24247`, seconds `100.50`, LSTM `0.9311`, delta `+0.1539`
- tick `23639`, seconds `91.00`, LSTM `0.2030`, delta `-0.0777`
- tick `24055`, seconds `97.50`, LSTM `0.5751`, delta `-0.0764`
- tick `23799`, seconds `93.50`, LSTM `0.6957`, delta `+0.0595`
- tick `18871`, seconds `16.50`, LSTM `0.4854`, delta `+0.0580`
- tick `23575`, seconds `90.00`, LSTM `0.4761`, delta `-0.0545`

## Top 15 local ridge features

- `lag_02__CT_flashed_players`: coefficient `-0.002801`, |coef| `0.002801`
- `lag_00__kill_diff_last_3s`: coefficient `0.002736`, |coef| `0.002736`
- `lag_12__CT_shots_fired_sum`: coefficient `-0.002571`, |coef| `0.002571`
- `lag_06__CT5__duck_amount`: coefficient `-0.002423`, |coef| `0.002423`
- `lag_04__T_place_DUMPSTER`: coefficient `-0.002217`, |coef| `0.002217`
- `lag_00__CT_flashed_players`: coefficient `0.002083`, |coef| `0.002083`
- `lag_00__CT_kills_last_3s`: coefficient `0.002080`, |coef| `0.002080`
- `lag_03__CT_flashed_players`: coefficient `-0.002038`, |coef| `0.002038`
- `lag_05__CT5__flash_duration`: coefficient `-0.001964`, |coef| `0.001964`
- `lag_13__CT2__is_walking`: coefficient `-0.001949`, |coef| `0.001949`
- `lag_14__T1__duck_amount`: coefficient `-0.001928`, |coef| `0.001928`
- `lag_00__CT5__flash_duration`: coefficient `0.001881`, |coef| `0.001881`
- `lag_12__CT1__shots_fired`: coefficient `-0.001814`, |coef| `0.001814`
- `lag_00__CT_damage_last_5s`: coefficient `0.001733`, |coef| `0.001733`
- `lag_13__T1__duck_amount`: coefficient `-0.001725`, |coef| `0.001725`

## Top 10 utility ridge features

- `lag_05__CT5__flash_duration`: coefficient `-0.001964` (lowers CT win probability)
- `lag_00__CT5__flash_duration`: coefficient `0.001881` (raises CT win probability)
- `lag_07__CT5__flash_duration`: coefficient `0.001598` (raises CT win probability)
- `lag_02__CT_flash_duration_sum`: coefficient `-0.001559` (lowers CT win probability)
- `lag_02__CT5__flash_duration`: coefficient `-0.001410` (lowers CT win probability)
- `lag_03__CT_flash_duration_sum`: coefficient `-0.001299` (lowers CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `0.001262` (raises CT win probability)
- `lag_02__CT1__flash_duration`: coefficient `-0.001247` (lowers CT win probability)
- `lag_03__CT5__flash_duration`: coefficient `-0.001202` (lowers CT win probability)
- `lag_13__T_flash_duration_sum`: coefficient `0.001166` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_02__CT_flashed_players`: coefficient `-0.002801` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002736` (raises CT win probability)
- `lag_12__CT_shots_fired_sum`: coefficient `-0.002571` (lowers CT win probability)
- `lag_06__CT5__duck_amount`: coefficient `-0.002423` (lowers CT win probability)
- `lag_04__T_place_DUMPSTER`: coefficient `-0.002217` (lowers CT win probability)
- `lag_00__CT_flashed_players`: coefficient `0.002083` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002080` (raises CT win probability)
- `lag_03__CT_flashed_players`: coefficient `-0.002038` (lowers CT win probability)
- `lag_13__CT2__is_walking`: coefficient `-0.001949` (lowers CT win probability)
- `lag_14__T1__duck_amount`: coefficient `-0.001928` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `23671`, seconds `91.50`, LSTM delta `+0.2567`

Top all feature movements:
- `lag_04__T_place_DUMPSTER`: contribution `+0.020156`
- `lag_02__CT_flashed_players`: contribution `+0.012269`
- `lag_06__CT5__duck_amount`: contribution `+0.009146`
- `lag_07__CT5__flash_duration`: contribution `+0.007964`
- `lag_13__T1__duck_amount`: contribution `+0.006755`

Top utility-only movements:
- `lag_07__CT5__flash_duration`: contribution `+0.007964`
- `lag_02__CT5__flash_duration`: contribution `+0.006622`
- `lag_02__CT_flash_duration_sum`: contribution `+0.004152`

### tick `24119`, seconds `98.50`, LSTM delta `+0.2208`

Top all feature movements:
- `lag_12__CT_shots_fired_sum`: contribution `+0.033943`
- `lag_12__CT1__shots_fired`: contribution `+0.018212`
- `lag_13__T_flash_duration_sum`: contribution `+0.008706`
- `lag_13__T3__flash_duration`: contribution `+0.007031`
- `lag_00__kill_diff_last_3s`: contribution `+0.006586`

Top utility-only movements:
- `lag_13__T_flash_duration_sum`: contribution `+0.008706`
- `lag_13__T3__flash_duration`: contribution `+0.007031`
- `lag_13__T4__flash_duration`: contribution `+0.005223`
- `lag_13__T2__flash_duration`: contribution `+0.004925`
- `lag_09__CT1__flash_duration`: contribution `+0.004196`

### tick `23607`, seconds `90.50`, LSTM delta `-0.1954`

Top all feature movements:
- `lag_02__CT_flashed_players`: contribution `-0.012269`
- `lag_05__CT5__flash_duration`: contribution `-0.009790`
- `lag_02__T_place_DUMPSTER`: contribution `-0.009439`
- `lag_00__CT_flashed_players`: contribution `-0.009124`
- `lag_00__CT5__flash_duration`: contribution `-0.008830`

Top utility-only movements:
- `lag_05__CT5__flash_duration`: contribution `-0.009790`
- `lag_00__CT5__flash_duration`: contribution `-0.008830`
- `lag_02__CT1__flash_duration`: contribution `-0.004487`
- `lag_02__CT_flash_duration_sum`: contribution `-0.004101`
- `lag_00__CT_flash_duration_sum`: contribution `-0.003360`

### tick `23703`, seconds `92.00`, LSTM delta `+0.1806`

Top all feature movements:
- `lag_03__CT_flashed_players`: contribution `+0.008927`
- `lag_05__T_place_DUMPSTER`: contribution `+0.008904`
- `lag_14__T1__duck_amount`: contribution `+0.007550`
- `lag_00__T_flash_duration_sum`: contribution `+0.006704`
- `lag_00__kill_diff_last_3s`: contribution `+0.006586`

Top utility-only movements:
- `lag_00__T_flash_duration_sum`: contribution `+0.006704`
- `lag_00__T4__flash_duration`: contribution `+0.006159`
- `lag_03__CT5__flash_duration`: contribution `+0.005642`
- `lag_00__T2__flash_duration`: contribution `+0.004813`
- `lag_03__CT_flash_duration_sum`: contribution `+0.003459`

### tick `24247`, seconds `100.50`, LSTM delta `+0.1539`

Top all feature movements:
- `lag_00__CT_place_ELECTRICALBOX`: contribution `+0.013599`
- `lag_00__kill_diff_last_3s`: contribution `+0.006586`
- `lag_00__CT_kills_last_3s`: contribution `+0.006006`
- `lag_02__T4__is_scoped`: contribution `+0.005819`
- `lag_03__T_place_TMAIN`: contribution `+0.005574`

Top utility-only movements:
- `lag_04__T3__flash_duration`: contribution `+0.004988`
- `lag_13__CT1__flash_duration`: contribution `+0.004488`
- `lag_07__CT2__flash_duration`: contribution `+0.003528`
- `lag_14__CT2__flash_duration`: contribution `+0.002786`
- `lag_06__T4__flash_duration`: contribution `+0.002445`
