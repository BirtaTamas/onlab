# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-g2-vs-fluxo-bo3-IhqycqXYyOA3DyfY0xuGyX/g2-vs-fluxo-m2-inferno.csv`
- round_num: `1`

## Largest probability jumps

- tick `4286`, seconds `47.50`, LSTM `0.1567`, delta `-0.2453`
- tick `4318`, seconds `48.00`, LSTM `0.0276`, delta `-0.1291`
- tick `4254`, seconds `47.00`, LSTM `0.4020`, delta `-0.0869`
- tick `4222`, seconds `46.50`, LSTM `0.4889`, delta `-0.0243`
- tick `3358`, seconds `33.00`, LSTM `0.4826`, delta `-0.0192`
- tick `3838`, seconds `40.50`, LSTM `0.5157`, delta `+0.0153`
- tick `4350`, seconds `48.50`, LSTM `0.0136`, delta `-0.0140`
- tick `3518`, seconds `35.50`, LSTM `0.4870`, delta `+0.0139`
- tick `3614`, seconds `37.00`, LSTM `0.4858`, delta `-0.0102`
- tick `2974`, seconds `27.00`, LSTM `0.5130`, delta `-0.0096`

## Top 15 local ridge features

- `lag_11__CT_place_UNDERPASS`: coefficient `0.002637`, |coef| `0.002637`
- `lag_00__T_kills_last_3s`: coefficient `-0.002122`, |coef| `0.002122`
- `lag_02__T_flashed_players`: coefficient `-0.001884`, |coef| `0.001884`
- `lag_00__T_damage_last_5s`: coefficient `-0.001814`, |coef| `0.001814`
- `lag_13__CT2__duck_amount`: coefficient `-0.001720`, |coef| `0.001720`
- `lag_00__damage_diff_last_5s`: coefficient `0.001679`, |coef| `0.001679`
- `lag_00__kill_diff_last_3s`: coefficient `0.001630`, |coef| `0.001630`
- `lag_02__T_place_BOMBSITEB`: coefficient `-0.001580`, |coef| `0.001580`
- `lag_02__T_macro_B`: coefficient `-0.001580`, |coef| `0.001580`
- `lag_14__CT1__duck_amount`: coefficient `-0.001568`, |coef| `0.001568`
- `lag_03__CT2__duck_amount`: coefficient `0.001514`, |coef| `0.001514`
- `lag_03__T_place_BOMBSITEB`: coefficient `-0.001488`, |coef| `0.001488`
- `lag_03__T_macro_B`: coefficient `-0.001488`, |coef| `0.001488`
- `lag_10__CT_place_UNDERPASS`: coefficient `0.001461`, |coef| `0.001461`
- `lag_00__CT_place_RUINS`: coefficient `0.001455`, |coef| `0.001455`

## Top 10 utility ridge features

- `lag_02__T_B_site_active_infernos`: coefficient `-0.001381` (lowers CT win probability)
- `lag_05__T2__molly`: coefficient `0.001099` (raises CT win probability)
- `lag_04__T4__smoke`: coefficient `0.001077` (raises CT win probability)
- `lag_02__T_active_infernos`: coefficient `-0.001006` (lowers CT win probability)
- `lag_02__T4__flash_duration`: coefficient `-0.000996` (lowers CT win probability)
- `lag_06__T3__flash`: coefficient `0.000770` (raises CT win probability)
- `lag_01__T_B_site_active_infernos`: coefficient `-0.000730` (lowers CT win probability)
- `lag_01__T4__flash_duration`: coefficient `-0.000724` (lowers CT win probability)
- `lag_03__T_B_site_active_infernos`: coefficient `-0.000723` (lowers CT win probability)
- `lag_02__active_infernos_total`: coefficient `-0.000682` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_11__CT_place_UNDERPASS`: coefficient `0.002637` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002122` (lowers CT win probability)
- `lag_02__T_flashed_players`: coefficient `-0.001884` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001814` (lowers CT win probability)
- `lag_13__CT2__duck_amount`: coefficient `-0.001720` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001679` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001630` (raises CT win probability)
- `lag_02__T_place_BOMBSITEB`: coefficient `-0.001580` (lowers CT win probability)
- `lag_02__T_macro_B`: coefficient `-0.001580` (lowers CT win probability)
- `lag_14__CT1__duck_amount`: coefficient `-0.001568` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `4286`, seconds `47.50`, LSTM delta `-0.2453`

Top all feature movements:
- `lag_11__CT_place_UNDERPASS`: contribution `-0.015292`
- `lag_02__T_flashed_players`: contribution `-0.010904`
- `lag_00__T_kills_last_3s`: contribution `-0.006722`
- `lag_13__CT2__duck_amount`: contribution `-0.006553`
- `lag_14__CT1__duck_amount`: contribution `-0.005981`

Top utility-only movements:
- `lag_02__T_B_site_active_infernos`: contribution `-0.003904`

### tick `4318`, seconds `48.00`, LSTM delta `-0.1291`

Top all feature movements:
- `lag_12__CT_place_UNDERPASS`: contribution `-0.008177`
- `lag_00__T_kills_last_3s`: contribution `-0.006722`
- `lag_03__T_flashed_players`: contribution `-0.006182`
- `lag_00__T_place_BOMBSITEB`: contribution `-0.004390`
- `lag_00__T_macro_B`: contribution `-0.004390`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `4254`, seconds `47.00`, LSTM delta `-0.0869`

Top all feature movements:
- `lag_10__CT_place_UNDERPASS`: contribution `-0.008471`
- `lag_13__CT1__duck_amount`: contribution `-0.003553`
- `lag_12__CT2__duck_amount`: contribution `-0.003456`
- `lag_02__CT2__duck_amount`: contribution `-0.003235`
- `lag_01__CT4__duck_amount`: contribution `+0.003152`

Top utility-only movements:
- `lag_01__T_B_site_active_infernos`: contribution `-0.002065`

### tick `4222`, seconds `46.50`, LSTM delta `-0.0243`

Top all feature movements:
- `lag_15__CT_place_SECONDMID`: contribution `-0.008274`
- `lag_09__CT_place_UNDERPASS`: contribution `-0.003861`
- `lag_09__CT_place_RUINS`: contribution `+0.003419`
- `lag_00__T_flashed_players`: contribution `+0.003164`
- `lag_00__T_macro_B`: contribution `-0.002195`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `3358`, seconds `33.00`, LSTM delta `-0.0192`

Top all feature movements:
- `lag_07__CT_place_SECONDMID`: contribution `-0.004171`
- `lag_15__T5__duck_amount`: contribution `-0.003951`
- `lag_14__CT2__duck_amount`: contribution `-0.003480`
- `lag_12__CT2__duck_amount`: contribution `+0.003456`
- `lag_03__T5__is_walking`: contribution `-0.002409`

Top utility-only movements:
- No utility movement among the top local contributors.
