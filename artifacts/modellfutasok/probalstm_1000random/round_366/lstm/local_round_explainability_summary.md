# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-falcons-bo3-yayytstbo8IxTFlUpfbUPR/mouz-vs-falcons-m1-train.csv`
- round_num: `13`

## Largest probability jumps

- tick `119434`, seconds `88.50`, LSTM `0.1397`, delta `-0.3190`
- tick `115178`, seconds `22.00`, LSTM `0.1544`, delta `-0.2373`
- tick `115434`, seconds `26.00`, LSTM `0.5620`, delta `+0.2285`
- tick `115530`, seconds `27.50`, LSTM `0.4652`, delta `-0.1884`
- tick `115242`, seconds `23.00`, LSTM `0.2532`, delta `+0.1420`
- tick `115114`, seconds `21.00`, LSTM `0.4043`, delta `-0.1118`
- tick `115498`, seconds `27.00`, LSTM `0.6536`, delta `+0.1072`
- tick `115306`, seconds `24.00`, LSTM `0.2631`, delta `+0.0668`
- tick `115402`, seconds `25.50`, LSTM `0.3335`, delta `+0.0658`
- tick `116618`, seconds `44.50`, LSTM `0.5051`, delta `+0.0623`

## Top 15 local ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.008672`, |coef| `0.008672`
- `lag_01__T_place_CONNECTOR`: coefficient `0.007889`, |coef| `0.007889`
- `lag_00__damage_diff_last_5s`: coefficient `0.007189`, |coef| `0.007189`
- `lag_00__kill_diff_last_3s`: coefficient `0.007098`, |coef| `0.007098`
- `lag_00__T_damage_last_5s`: coefficient `-0.006797`, |coef| `0.006797`
- `lag_00__CT4__alive`: coefficient `0.006178`, |coef| `0.006178`
- `lag_00__CT4__hp`: coefficient `0.005389`, |coef| `0.005389`
- `lag_00__CT4__armor`: coefficient `0.005378`, |coef| `0.005378`
- `lag_00__CT4__duck_amount`: coefficient `0.004344`, |coef| `0.004344`
- `lag_00__CT_spread_xy`: coefficient `0.004096`, |coef| `0.004096`
- `lag_01__T_place_BOMBSITEA`: coefficient `-0.003983`, |coef| `0.003983`
- `lag_01__T_macro_A`: coefficient `-0.003983`, |coef| `0.003983`
- `lag_00__T_place_CONNECTOR`: coefficient `0.003811`, |coef| `0.003811`
- `lag_00__spread_diff`: coefficient `0.003776`, |coef| `0.003776`
- `lag_02__T5__is_walking`: coefficient `0.003377`, |coef| `0.003377`

## Top 10 utility ridge features

- `lag_06__T_B_site_active_smokes`: coefficient `0.001197` (raises CT win probability)
- `lag_06__T_A_site_active_smokes`: coefficient `0.001125` (raises CT win probability)
- `lag_06__CT_B_site_active_smokes`: coefficient `0.001012` (raises CT win probability)
- `lag_14__T_B_site_active_smokes`: coefficient `0.001006` (raises CT win probability)
- `lag_06__CT_A_site_active_smokes`: coefficient `0.000979` (raises CT win probability)
- `lag_10__T1__smoke`: coefficient `-0.000971` (lowers CT win probability)
- `lag_09__T_active_infernos`: coefficient `-0.000956` (lowers CT win probability)
- `lag_14__T_A_site_active_smokes`: coefficient `0.000948` (raises CT win probability)
- `lag_06__active_smokes_total`: coefficient `0.000945` (raises CT win probability)
- `lag_02__CT1__flash`: coefficient `-0.000846` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.008672` (lowers CT win probability)
- `lag_01__T_place_CONNECTOR`: coefficient `0.007889` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.007189` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.007098` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.006797` (lowers CT win probability)
- `lag_00__CT4__alive`: coefficient `0.006178` (raises CT win probability)
- `lag_00__CT4__hp`: coefficient `0.005389` (raises CT win probability)
- `lag_00__CT4__armor`: coefficient `0.005378` (raises CT win probability)
- `lag_00__CT4__duck_amount`: coefficient `0.004344` (raises CT win probability)
- `lag_00__CT_spread_xy`: coefficient `0.004096` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `119434`, seconds `88.50`, LSTM delta `-0.3190`

Top all feature movements:
- `lag_01__T_place_CONNECTOR`: contribution `-0.038202`
- `lag_00__T_kills_last_3s`: contribution `-0.027475`
- `lag_00__kill_diff_last_3s`: contribution `-0.017085`
- `lag_00__CT4__alive`: contribution `-0.015153`
- `lag_00__T_damage_last_5s`: contribution `-0.014343`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `115178`, seconds `22.00`, LSTM delta `-0.2373`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.027475`
- `lag_00__damage_diff_last_5s`: contribution `-0.020759`
- `lag_00__kill_diff_last_3s`: contribution `-0.017085`
- `lag_00__T_damage_last_5s`: contribution `-0.016298`
- `lag_02__T_kills_last_3s`: contribution `-0.007926`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `115434`, seconds `26.00`, LSTM delta `+0.2285`

Top all feature movements:
- `lag_00__damage_diff_last_5s`: contribution `+0.023192`
- `lag_10__CT_place_CONNECTOR`: contribution `+0.018201`
- `lag_05__T_bomb_zone_count`: contribution `+0.013321`
- `lag_02__CT_place_ENTRANCE`: contribution `+0.011790`
- `lag_00__T_damage_last_5s`: contribution `+0.011572`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `115530`, seconds `27.50`, LSTM delta `-0.1884`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.027475`
- `lag_00__kill_diff_last_3s`: contribution `-0.017085`
- `lag_00__damage_diff_last_5s`: contribution `-0.016867`
- `lag_00__T_damage_last_5s`: contribution `-0.014343`
- `lag_08__T_bomb_zone_count`: contribution `-0.012140`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `115242`, seconds `23.00`, LSTM delta `+0.1420`

Top all feature movements:
- `lag_04__CT_place_CONNECTOR`: contribution `+0.017169`
- `lag_00__kill_diff_last_3s`: contribution `+0.017085`
- `lag_13__CT_place_BACKOFB`: contribution `+0.016223`
- `lag_14__CT_place_BACKOFB`: contribution `+0.011858`
- `lag_12__CT_place_LONGDOG`: contribution `+0.009875`

Top utility-only movements:
- No utility movement among the top local contributors.
