# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-faze-vs-inner-circle-bo3-runM3q2zOKSAHTeRui0Q2h/faze-vs-inner-circle-m2-nuke.csv`
- round_num: `13`

## Largest probability jumps

- tick `113168`, seconds `33.00`, LSTM `0.8852`, delta `+0.3699`
- tick `112752`, seconds `26.50`, LSTM `0.6939`, delta `+0.1843`
- tick `113712`, seconds `41.50`, LSTM `0.7259`, delta `-0.1593`
- tick `112496`, seconds `22.50`, LSTM `0.5167`, delta `-0.1412`
- tick `113872`, seconds `44.00`, LSTM `0.8891`, delta `+0.1182`
- tick `112880`, seconds `28.50`, LSTM `0.5702`, delta `-0.0994`
- tick `112848`, seconds `28.00`, LSTM `0.6696`, delta `-0.0758`
- tick `112816`, seconds `27.50`, LSTM `0.7454`, delta `+0.0695`
- tick `112272`, seconds `19.00`, LSTM `0.6602`, delta `+0.0660`
- tick `114288`, seconds `50.50`, LSTM `0.9451`, delta `+0.0585`

## Top 15 local ridge features

- `lag_00__CT_defusing_count`: coefficient `0.003457`, |coef| `0.003457`
- `lag_15__CT_place_GARAGE`: coefficient `-0.003152`, |coef| `0.003152`
- `lag_00__kill_diff_last_3s`: coefficient `0.003021`, |coef| `0.003021`
- `lag_00__damage_diff_last_5s`: coefficient `0.002364`, |coef| `0.002364`
- `lag_09__T_place_HUT`: coefficient `0.002338`, |coef| `0.002338`
- `lag_06__T_place_SQUEAKY`: coefficient `0.002311`, |coef| `0.002311`
- `lag_01__CT_place_MINI`: coefficient `-0.002284`, |coef| `0.002284`
- `lag_06__CT_defusing_count`: coefficient `0.002201`, |coef| `0.002201`
- `lag_07__CT_defusing_count`: coefficient `-0.002197`, |coef| `0.002197`
- `lag_08__CT_utility_damage_last_5s`: coefficient `-0.002113`, |coef| `0.002113`
- `lag_00__CT_kills_last_3s`: coefficient `0.002075`, |coef| `0.002075`
- `lag_08__T_bomb_zone_count`: coefficient `-0.002070`, |coef| `0.002070`
- `lag_03__T_place_SQUEAKY`: coefficient `-0.002054`, |coef| `0.002054`
- `lag_12__CT_defusing_count`: coefficient `0.002024`, |coef| `0.002024`
- `lag_00__T_place_SQUEAKY`: coefficient `-0.002017`, |coef| `0.002017`

## Top 10 utility ridge features

- `lag_08__CT_utility_damage_last_5s`: coefficient `-0.002113` (lowers CT win probability)
- `lag_08__utility_damage_diff_last_5s`: coefficient `-0.001727` (lowers CT win probability)
- `lag_12__T_A_site_active_infernos`: coefficient `-0.001092` (lowers CT win probability)
- `lag_12__T_B_site_active_infernos`: coefficient `-0.001039` (lowers CT win probability)
- `lag_12__T_active_infernos`: coefficient `-0.000777` (lowers CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.000723` (lowers CT win probability)
- `lag_09__CT4__flash`: coefficient `-0.000664` (lowers CT win probability)
- `lag_03__T_flash_alpha_mean`: coefficient `0.000587` (raises CT win probability)
- `lag_12__active_infernos_total`: coefficient `-0.000550` (lowers CT win probability)
- `lag_09__CT_utility_damage_last_5s`: coefficient `-0.000437` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_defusing_count`: coefficient `0.003457` (raises CT win probability)
- `lag_15__CT_place_GARAGE`: coefficient `-0.003152` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003021` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002364` (raises CT win probability)
- `lag_09__T_place_HUT`: coefficient `0.002338` (raises CT win probability)
- `lag_06__T_place_SQUEAKY`: coefficient `0.002311` (raises CT win probability)
- `lag_01__CT_place_MINI`: coefficient `-0.002284` (lowers CT win probability)
- `lag_06__CT_defusing_count`: coefficient `0.002201` (raises CT win probability)
- `lag_07__CT_defusing_count`: coefficient `-0.002197` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002075` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `113168`, seconds `33.00`, LSTM delta `+0.3699`

Top all feature movements:
- `lag_15__CT_place_GARAGE`: contribution `+0.022656`
- `lag_00__kill_diff_last_3s`: contribution `+0.014542`
- `lag_06__T_place_SQUEAKY`: contribution `+0.014390`
- `lag_01__CT_place_MINI`: contribution `+0.014005`
- `lag_03__T_place_SQUEAKY`: contribution `+0.012786`

Top utility-only movements:
- `lag_08__CT_utility_damage_last_5s`: contribution `+0.012558`
- `lag_08__utility_damage_diff_last_5s`: contribution `+0.008420`

### tick `112752`, seconds `26.50`, LSTM delta `+0.1843`

Top all feature movements:
- `lag_15__CT_place_GARAGE`: contribution `+0.022656`
- `lag_09__T_place_HUT`: contribution `+0.021797`
- `lag_05__T_place_HUT`: contribution `+0.019020`
- `lag_01__CT_place_RAFTERS`: contribution `+0.009229`
- `lag_05__T_place_SQUEAKY`: contribution `-0.009100`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `113712`, seconds `41.50`, LSTM delta `-0.1593`

Top all feature movements:
- `lag_06__CT_defusing_count`: contribution `-0.021334`
- `lag_07__CT_defusing_count`: contribution `-0.021294`
- `lag_04__T_place_HUT`: contribution `-0.010929`
- `lag_11__CT_kills_last_3s`: contribution `-0.007920`
- `lag_13__CT_place_RAFTERS`: contribution `-0.007713`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `112496`, seconds `22.50`, LSTM delta `-0.1412`

Top all feature movements:
- `lag_01__CT_place_MINI`: contribution `-0.014005`
- `lag_13__T_place_TROPHY`: contribution `-0.010732`
- `lag_02__CT_place_HEAVEN`: contribution `-0.009317`
- `lag_13__CT_place_RAFTERS`: contribution `-0.007713`
- `lag_00__kill_diff_last_3s`: contribution `-0.007271`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `113872`, seconds `44.00`, LSTM delta `+0.1182`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `+0.033507`
- `lag_09__T_place_HUT`: contribution `+0.021797`
- `lag_12__CT_defusing_count`: contribution `+0.019621`
- `lag_01__CT_duck_amount_mean`: contribution `+0.003549`
- `lag_06__CT1__duck_amount`: contribution `+0.003192`

Top utility-only movements:
- No utility movement among the top local contributors.
