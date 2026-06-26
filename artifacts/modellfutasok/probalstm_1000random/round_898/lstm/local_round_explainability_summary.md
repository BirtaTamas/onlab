# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-furia-vs-fluxo-bo3-cy88FeSpEinhT8XDRxQGHo/furia-vs-fluxo-m2-mirage.csv`
- round_num: `15`

## Largest probability jumps

- tick `105773`, seconds `80.50`, LSTM `0.7701`, delta `+0.3182`
- tick `105453`, seconds `75.50`, LSTM `0.4853`, delta `-0.2768`
- tick `104237`, seconds `56.50`, LSTM `0.6091`, delta `+0.1789`
- tick `105037`, seconds `69.00`, LSTM `0.6566`, delta `-0.1773`
- tick `103757`, seconds `49.00`, LSTM `0.6542`, delta `-0.1769`
- tick `105101`, seconds `70.00`, LSTM `0.7939`, delta `+0.1301`
- tick `101229`, seconds `9.50`, LSTM `0.7883`, delta `+0.1235`
- tick `105965`, seconds `83.50`, LSTM `0.8893`, delta `+0.1022`
- tick `102957`, seconds `36.50`, LSTM `0.8908`, delta `+0.0939`
- tick `102829`, seconds `34.50`, LSTM `0.8232`, delta `+0.0872`

## Top 15 local ridge features

- `lag_11__T_place_STAIRS`: coefficient `0.003846`, |coef| `0.003846`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.003531`, |coef| `0.003531`
- `lag_00__T_place_JUNGLE`: coefficient `-0.003345`, |coef| `0.003345`
- `lag_00__kill_diff_last_3s`: coefficient `0.003178`, |coef| `0.003178`
- `lag_00__CT_duck_amount_mean`: coefficient `0.003095`, |coef| `0.003095`
- `lag_09__T_duck_amount_mean`: coefficient `-0.002940`, |coef| `0.002940`
- `lag_07__CT_place_PALACEALLEY`: coefficient `-0.002936`, |coef| `0.002936`
- `lag_00__damage_diff_last_5s`: coefficient `0.002792`, |coef| `0.002792`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002770`, |coef| `0.002770`
- `lag_10__T_duck_amount_mean`: coefficient `-0.002571`, |coef| `0.002571`
- `lag_03__CT_duck_amount_mean`: coefficient `-0.002555`, |coef| `0.002555`
- `lag_00__CT_defusing_count`: coefficient `0.002380`, |coef| `0.002380`
- `lag_05__CT_place_TRAMP`: coefficient `0.002355`, |coef| `0.002355`
- `lag_01__CT_place_JUNGLE`: coefficient `0.002331`, |coef| `0.002331`
- `lag_00__CT_kills_last_3s`: coefficient `0.002298`, |coef| `0.002298`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.003531` (lowers CT win probability)
- `lag_06__T_flash_alpha_mean`: coefficient `-0.001462` (lowers CT win probability)
- `lag_10__CT_utility_damage_last_5s`: coefficient `0.001411` (raises CT win probability)
- `lag_05__CT4__flash_duration`: coefficient `-0.001386` (lowers CT win probability)
- `lag_03__T_flash_alpha_mean`: coefficient `-0.001297` (lowers CT win probability)
- `lag_10__CT4__molly`: coefficient `-0.001211` (lowers CT win probability)
- `lag_10__utility_damage_diff_last_5s`: coefficient `0.001169` (raises CT win probability)
- `lag_01__CT4__flash_duration`: coefficient `0.001116` (raises CT win probability)
- `lag_07__CT_utility_damage_last_5s`: coefficient `-0.001055` (lowers CT win probability)
- `lag_10__CT4__flash_duration`: coefficient `-0.000926` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_11__T_place_STAIRS`: coefficient `0.003846` (raises CT win probability)
- `lag_00__T_place_JUNGLE`: coefficient `-0.003345` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003178` (raises CT win probability)
- `lag_00__CT_duck_amount_mean`: coefficient `0.003095` (raises CT win probability)
- `lag_09__T_duck_amount_mean`: coefficient `-0.002940` (lowers CT win probability)
- `lag_07__CT_place_PALACEALLEY`: coefficient `-0.002936` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002792` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002770` (raises CT win probability)
- `lag_10__T_duck_amount_mean`: coefficient `-0.002571` (lowers CT win probability)
- `lag_03__CT_duck_amount_mean`: coefficient `-0.002555` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `105773`, seconds `80.50`, LSTM delta `+0.3182`

Top all feature movements:
- `lag_00__T_place_JUNGLE`: contribution `+0.043326`
- `lag_00__T_flash_alpha_mean`: contribution `+0.021422`
- `lag_00__CT_duck_amount_mean`: contribution `+0.018534`
- `lag_09__T_duck_amount_mean`: contribution `+0.017096`
- `lag_03__CT_duck_amount_mean`: contribution `+0.015298`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.021422`

### tick `105453`, seconds `75.50`, LSTM delta `-0.2768`

Top all feature movements:
- `lag_11__T_place_STAIRS`: contribution `-0.073628`
- `lag_10__T_duck_amount_mean`: contribution `-0.014954`
- `lag_00__CT_duck_amount_mean`: contribution `-0.009267`
- `lag_10__CT_utility_damage_last_5s`: contribution `-0.009009`
- `lag_00__kill_diff_last_3s`: contribution `-0.007648`

Top utility-only movements:
- `lag_10__CT_utility_damage_last_5s`: contribution `-0.009009`
- `lag_10__utility_damage_diff_last_5s`: contribution `-0.006121`

### tick `104237`, seconds `56.50`, LSTM delta `+0.1789`

Top all feature movements:
- `lag_05__CT_place_TRAMP`: contribution `+0.031722`
- `lag_05__CT_place_PALACEALLEY`: contribution `+0.020946`
- `lag_05__CT4__flash_duration`: contribution `+0.010690`
- `lag_00__kill_diff_last_3s`: contribution `+0.007648`
- `lag_10__CT2__duck_amount`: contribution `+0.006800`

Top utility-only movements:
- `lag_05__CT4__flash_duration`: contribution `+0.010690`

### tick `105037`, seconds `69.00`, LSTM delta `-0.1773`

Top all feature movements:
- `lag_14__CT_place_TRAMP`: contribution `-0.027626`
- `lag_03__T_shots_fired_sum`: contribution `-0.007866`
- `lag_00__kill_diff_last_3s`: contribution `-0.007648`
- `lag_07__CT_utility_damage_last_5s`: contribution `-0.006738`
- `lag_00__damage_diff_last_5s`: contribution `-0.006299`

Top utility-only movements:
- `lag_07__CT_utility_damage_last_5s`: contribution `-0.006738`
- `lag_07__utility_damage_diff_last_5s`: contribution `-0.004484`

### tick `103757`, seconds `49.00`, LSTM delta `-0.1769`

Top all feature movements:
- `lag_07__CT_place_PALACEALLEY`: contribution `-0.044820`
- `lag_09__T_place_JUNGLE`: contribution `-0.019541`
- `lag_07__CT_place_TSPAWN`: contribution `-0.010810`
- `lag_00__kill_diff_last_3s`: contribution `-0.007648`
- `lag_10__CT4__flash_duration`: contribution `-0.006984`

Top utility-only movements:
- `lag_10__CT4__flash_duration`: contribution `-0.006984`
- `lag_01__T_A_site_active_infernos`: contribution `-0.001934`
