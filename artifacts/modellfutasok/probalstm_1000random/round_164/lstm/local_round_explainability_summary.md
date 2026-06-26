# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-g2-vs-fluxo-bo3-IhqycqXYyOA3DyfY0xuGyX/g2-vs-fluxo-m3-mirage.csv`
- round_num: `10`

## Largest probability jumps

- tick `75323`, seconds `133.50`, LSTM `0.3642`, delta `-0.3024`
- tick `73819`, seconds `110.00`, LSTM `0.4288`, delta `+0.2823`
- tick `73051`, seconds `98.00`, LSTM `0.1092`, delta `-0.2761`
- tick `75739`, seconds `140.00`, LSTM `0.4632`, delta `+0.2548`
- tick `73915`, seconds `111.50`, LSTM `0.6358`, delta `+0.1665`
- tick `72635`, seconds `91.50`, LSTM `0.6388`, delta `-0.1513`
- tick `72667`, seconds `92.00`, LSTM `0.4883`, delta `-0.1505`
- tick `67771`, seconds `15.50`, LSTM `0.7940`, delta `+0.1296`
- tick `75771`, seconds `140.50`, LSTM `0.3480`, delta `-0.1153`
- tick `73531`, seconds `105.50`, LSTM `0.1924`, delta `-0.1147`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005193`, |coef| `0.005193`
- `lag_00__T_kills_last_3s`: coefficient `-0.004558`, |coef| `0.004558`
- `lag_09__T_bomb_zone_count`: coefficient `-0.003232`, |coef| `0.003232`
- `lag_03__CT_duck_amount_mean`: coefficient `-0.003224`, |coef| `0.003224`
- `lag_08__CT_duck_amount_mean`: coefficient `-0.003015`, |coef| `0.003015`
- `lag_00__damage_diff_last_5s`: coefficient `0.002962`, |coef| `0.002962`
- `lag_00__CT3__alive`: coefficient `0.002864`, |coef| `0.002864`
- `lag_03__CT2__duck_amount`: coefficient `-0.002752`, |coef| `0.002752`
- `lag_00__T_shots_fired_sum`: coefficient `-0.002727`, |coef| `0.002727`
- `lag_15__CT_place_JUNGLE`: coefficient `0.002665`, |coef| `0.002665`
- `lag_08__T_duck_amount_mean`: coefficient `-0.002645`, |coef| `0.002645`
- `lag_14__damage_diff_last_5s`: coefficient `0.002642`, |coef| `0.002642`
- `lag_01__T_kills_last_3s`: coefficient `-0.002576`, |coef| `0.002576`
- `lag_02__CT_duck_amount_mean`: coefficient `-0.002571`, |coef| `0.002571`
- `lag_01__CT3__is_scoped`: coefficient `-0.002569`, |coef| `0.002569`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.002534` (lowers CT win probability)
- `lag_09__CT2__flash`: coefficient `0.001843` (raises CT win probability)
- `lag_07__T1__flash_duration`: coefficient `-0.001294` (lowers CT win probability)
- `lag_10__CT2__flash`: coefficient `0.001100` (raises CT win probability)
- `lag_04__T_flash_alpha_mean`: coefficient `-0.001073` (lowers CT win probability)
- `lag_07__CT2__flash`: coefficient `0.001069` (raises CT win probability)
- `lag_02__T_flash_alpha_mean`: coefficient `-0.001021` (lowers CT win probability)
- `lag_03__T_flash_alpha_mean`: coefficient `-0.001018` (lowers CT win probability)
- `lag_09__CT2__utility_total`: coefficient `0.000931` (raises CT win probability)
- `lag_06__CT2__flash`: coefficient `0.000930` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005193` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.004558` (lowers CT win probability)
- `lag_09__T_bomb_zone_count`: coefficient `-0.003232` (lowers CT win probability)
- `lag_03__CT_duck_amount_mean`: coefficient `-0.003224` (lowers CT win probability)
- `lag_08__CT_duck_amount_mean`: coefficient `-0.003015` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002962` (raises CT win probability)
- `lag_00__CT3__alive`: coefficient `0.002864` (raises CT win probability)
- `lag_03__CT2__duck_amount`: coefficient `-0.002752` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.002727` (lowers CT win probability)
- `lag_15__CT_place_JUNGLE`: coefficient `0.002665` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `75323`, seconds `133.50`, LSTM delta `-0.3024`

Top all feature movements:
- `lag_08__T_duck_amount_mean`: contribution `-0.015385`
- `lag_00__T_kills_last_3s`: contribution `-0.014441`
- `lag_00__kill_diff_last_3s`: contribution `-0.012499`
- `lag_02__T_duck_amount_mean`: contribution `-0.011805`
- `lag_01__CT3__is_scoped`: contribution `-0.011685`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `73819`, seconds `110.00`, LSTM delta `+0.2823`

Top all feature movements:
- `lag_09__T_bomb_zone_count`: contribution `+0.018812`
- `lag_00__kill_diff_last_3s`: contribution `+0.012499`
- `lag_14__damage_diff_last_5s`: contribution `+0.011800`
- `lag_11__CT2__is_scoped`: contribution `+0.009081`
- `lag_00__damage_diff_last_5s`: contribution `+0.006416`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `73051`, seconds `98.00`, LSTM delta `-0.2761`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.014441`
- `lag_00__kill_diff_last_3s`: contribution `-0.012499`
- `lag_00__T_shots_fired_sum`: contribution `-0.010222`
- `lag_00__CT_place_UNDERPASS`: contribution `-0.007718`
- `lag_04__T_place_CONNECTOR`: contribution `-0.006966`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `75739`, seconds `140.00`, LSTM delta `+0.2548`

Top all feature movements:
- `lag_08__CT_duck_amount_mean`: contribution `+0.018053`
- `lag_03__CT_duck_amount_mean`: contribution `+0.016575`
- `lag_00__T_flash_alpha_mean`: contribution `+0.015373`
- `lag_00__kill_diff_last_3s`: contribution `+0.012499`
- `lag_03__CT2__duck_amount`: contribution `+0.009003`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.015373`

### tick `73915`, seconds `111.50`, LSTM delta `+0.1665`

Top all feature movements:
- `lag_12__T_bomb_zone_count`: contribution `+0.013954`
- `lag_00__T_duck_amount_mean`: contribution `+0.011606`
- `lag_03__CT2__duck_amount`: contribution `+0.010486`
- `lag_03__CT_duck_amount_mean`: contribution `+0.009652`
- `lag_00__CT2__is_scoped`: contribution `-0.005812`

Top utility-only movements:
- No utility movement among the top local contributors.
