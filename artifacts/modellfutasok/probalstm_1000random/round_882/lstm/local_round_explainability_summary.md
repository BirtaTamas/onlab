# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-b8-vs-hotu-bo3-tmCfOETKzYqjV6vSvNp3-F/b8-vs-hotu-m3-ancient.csv`
- round_num: `14`

## Largest probability jumps

- tick `103399`, seconds `69.50`, LSTM `0.4783`, delta `+0.2621`
- tick `101095`, seconds `33.50`, LSTM `0.3552`, delta `+0.1194`
- tick `105543`, seconds `103.00`, LSTM `0.0683`, delta `-0.1139`
- tick `101127`, seconds `34.00`, LSTM `0.2458`, delta `-0.1094`
- tick `104391`, seconds `85.00`, LSTM `0.3600`, delta `-0.0843`
- tick `99559`, seconds `9.50`, LSTM `0.2625`, delta `-0.0810`
- tick `104583`, seconds `88.00`, LSTM `0.2985`, delta `-0.0688`
- tick `100359`, seconds `22.00`, LSTM `0.1836`, delta `-0.0680`
- tick `105671`, seconds `105.00`, LSTM `0.0588`, delta `-0.0654`
- tick `102919`, seconds `62.00`, LSTM `0.3539`, delta `-0.0606`

## Top 15 local ridge features

- `lag_00__damage_diff_last_5s`: coefficient `0.003603`, |coef| `0.003603`
- `lag_00__kill_diff_last_3s`: coefficient `0.003294`, |coef| `0.003294`
- `lag_00__T_place_MAINHALL`: coefficient `0.003184`, |coef| `0.003184`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002861`, |coef| `0.002861`
- `lag_00__CT_place_MAINHALL`: coefficient `-0.002815`, |coef| `0.002815`
- `lag_09__CT_place_HOUSE`: coefficient `0.002548`, |coef| `0.002548`
- `lag_00__T_macro_A`: coefficient `-0.002399`, |coef| `0.002399`
- `lag_00__T_place_BOMBSITEA`: coefficient `-0.002399`, |coef| `0.002399`
- `lag_15__T_place_MAINHALL`: coefficient `-0.002388`, |coef| `0.002388`
- `lag_14__T_place_MAINHALL`: coefficient `-0.002172`, |coef| `0.002172`
- `lag_08__T3__duck_amount`: coefficient `-0.002171`, |coef| `0.002171`
- `lag_00__T_damage_last_5s`: coefficient `-0.002150`, |coef| `0.002150`
- `lag_00__T_kills_last_3s`: coefficient `-0.002108`, |coef| `0.002108`
- `lag_00__T_shots_fired_sum`: coefficient `-0.002107`, |coef| `0.002107`
- `lag_00__CT_kills_last_3s`: coefficient `0.002031`, |coef| `0.002031`

## Top 10 utility ridge features

- `lag_04__T2__smoke`: coefficient `-0.001780` (lowers CT win probability)
- `lag_12__CT4__smoke`: coefficient `-0.001659` (lowers CT win probability)
- `lag_09__CT_A_site_active_smokes`: coefficient `0.001443` (raises CT win probability)
- `lag_12__CT_B_site_active_smokes`: coefficient `-0.001191` (lowers CT win probability)
- `lag_00__T5__flash`: coefficient `-0.001058` (lowers CT win probability)
- `lag_14__CT5__flash_duration`: coefficient `0.001000` (raises CT win probability)
- `lag_04__T2__utility_total`: coefficient `-0.000923` (lowers CT win probability)
- `lag_14__CT_flash_duration_sum`: coefficient `0.000787` (raises CT win probability)
- `lag_14__CT4__flash_duration`: coefficient `0.000769` (raises CT win probability)
- `lag_08__CT_A_site_active_smokes`: coefficient `0.000739` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__damage_diff_last_5s`: coefficient `0.003603` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003294` (raises CT win probability)
- `lag_00__T_place_MAINHALL`: coefficient `0.003184` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002861` (raises CT win probability)
- `lag_00__CT_place_MAINHALL`: coefficient `-0.002815` (lowers CT win probability)
- `lag_09__CT_place_HOUSE`: coefficient `0.002548` (raises CT win probability)
- `lag_00__T_macro_A`: coefficient `-0.002399` (lowers CT win probability)
- `lag_00__T_place_BOMBSITEA`: coefficient `-0.002399` (lowers CT win probability)
- `lag_15__T_place_MAINHALL`: coefficient `-0.002388` (lowers CT win probability)
- `lag_14__T_place_MAINHALL`: coefficient `-0.002172` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `103399`, seconds `69.50`, LSTM delta `+0.2621`

Top all feature movements:
- `lag_15__T_place_MAINHALL`: contribution `+0.008619`
- `lag_03__CT_place_SIDEHALL`: contribution `+0.008290`
- `lag_08__T3__duck_amount`: contribution `+0.008186`
- `lag_00__damage_diff_last_5s`: contribution `+0.008129`
- `lag_00__kill_diff_last_3s`: contribution `+0.007929`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `101095`, seconds `33.50`, LSTM delta `+0.1194`

Top all feature movements:
- `lag_00__CT_place_MAINHALL`: contribution `+0.023302`
- `lag_00__CT_shots_fired_sum`: contribution `+0.015901`
- `lag_00__CT4__shots_fired`: contribution `+0.008050`
- `lag_12__T_place_TUNNEL`: contribution `+0.005523`
- `lag_02__T_place_TSIDELOWER`: contribution `+0.005013`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `105543`, seconds `103.00`, LSTM delta `-0.1139`

Top all feature movements:
- `lag_00__damage_diff_last_5s`: contribution `-0.008129`
- `lag_00__kill_diff_last_3s`: contribution `-0.007929`
- `lag_00__T_kills_last_3s`: contribution `-0.006677`
- `lag_00__CT_place_SIDEHALL`: contribution `-0.005192`
- `lag_00__T_damage_last_5s`: contribution `-0.005155`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `101127`, seconds `34.00`, LSTM delta `-0.1094`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.023851`
- `lag_00__CT4__shots_fired`: contribution `-0.012075`
- `lag_08__T3__duck_amount`: contribution `-0.008186`
- `lag_05__CT_shots_fired_sum`: contribution `-0.004930`
- `lag_07__T3__duck_amount`: contribution `-0.004622`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `104391`, seconds `85.00`, LSTM delta `-0.0843`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.012636`
- `lag_00__CT_place_TSIDEUPPER`: contribution `-0.010632`
- `lag_04__T_place_MAINHALL`: contribution `-0.005576`
- `lag_02__CT_shots_fired_sum`: contribution `-0.005021`
- `lag_01__T_shots_fired_sum`: contribution `-0.004976`

Top utility-only movements:
- No utility movement among the top local contributors.
