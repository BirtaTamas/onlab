# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-g2-vs-fluxo-bo3-IhqycqXYyOA3DyfY0xuGyX/g2-vs-fluxo-m2-inferno.csv`
- round_num: `8`

## Largest probability jumps

- tick `53179`, seconds `23.00`, LSTM `0.1374`, delta `-0.1954`
- tick `53211`, seconds `23.50`, LSTM `0.0309`, delta `-0.1065`
- tick `52411`, seconds `11.00`, LSTM `0.3180`, delta `-0.0610`
- tick `52091`, seconds `6.00`, LSTM `0.3970`, delta `+0.0571`
- tick `51771`, seconds `1.00`, LSTM `0.2562`, delta `-0.0525`
- tick `52539`, seconds `13.00`, LSTM `0.3200`, delta `+0.0369`
- tick `52027`, seconds `5.00`, LSTM `0.3325`, delta `+0.0360`
- tick `51835`, seconds `2.00`, LSTM `0.3190`, delta `+0.0351`
- tick `52443`, seconds `11.50`, LSTM `0.2841`, delta `-0.0339`
- tick `52315`, seconds `9.50`, LSTM `0.3785`, delta `-0.0299`

## Top 15 local ridge features

- `lag_00__CT_place_APARTMENTS`: coefficient `0.001685`, |coef| `0.001685`
- `lag_00__CT2__is_scoped`: coefficient `0.001608`, |coef| `0.001608`
- `lag_13__T_A_site_active_infernos`: coefficient `0.001551`, |coef| `0.001551`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001396`, |coef| `0.001396`
- `lag_00__CT2__flash`: coefficient `0.001346`, |coef| `0.001346`
- `lag_00__T_kills_last_3s`: coefficient `-0.001178`, |coef| `0.001178`
- `lag_00__T1__shots_fired`: coefficient `-0.001163`, |coef| `0.001163`
- `lag_13__T_active_infernos`: coefficient `0.001122`, |coef| `0.001122`
- `lag_13__T3__is_scoped`: coefficient `-0.001101`, |coef| `0.001101`
- `lag_12__T_place_UNDERPASS`: coefficient `-0.001084`, |coef| `0.001084`
- `lag_10__CT_mollies_last_5s`: coefficient `0.001051`, |coef| `0.001051`
- `lag_10__T1__duck_amount`: coefficient `0.001020`, |coef| `0.001020`
- `lag_01__CT_place_APARTMENTS`: coefficient `0.000944`, |coef| `0.000944`
- `lag_00__CT_mollies_last_5s`: coefficient `-0.000923`, |coef| `0.000923`
- `lag_00__kill_diff_last_3s`: coefficient `0.000919`, |coef| `0.000919`

## Top 10 utility ridge features

- `lag_13__T_A_site_active_infernos`: coefficient `0.001551` (raises CT win probability)
- `lag_00__CT2__flash`: coefficient `0.001346` (raises CT win probability)
- `lag_13__T_active_infernos`: coefficient `0.001122` (raises CT win probability)
- `lag_10__CT_mollies_last_5s`: coefficient `0.001051` (raises CT win probability)
- `lag_00__CT_mollies_last_5s`: coefficient `-0.000923` (lowers CT win probability)
- `lag_13__active_infernos_total`: coefficient `0.000790` (raises CT win probability)
- `lag_02__CT_mollies_last_5s`: coefficient `0.000744` (raises CT win probability)
- `lag_14__T_A_site_active_infernos`: coefficient `0.000719` (raises CT win probability)
- `lag_00__CT2__utility_total`: coefficient `0.000702` (raises CT win probability)
- `lag_04__CT_mollies_last_5s`: coefficient `-0.000679` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_APARTMENTS`: coefficient `0.001685` (raises CT win probability)
- `lag_00__CT2__is_scoped`: coefficient `0.001608` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001396` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001178` (lowers CT win probability)
- `lag_00__T1__shots_fired`: coefficient `-0.001163` (lowers CT win probability)
- `lag_13__T3__is_scoped`: coefficient `-0.001101` (lowers CT win probability)
- `lag_12__T_place_UNDERPASS`: coefficient `-0.001084` (lowers CT win probability)
- `lag_10__T1__duck_amount`: coefficient `0.001020` (raises CT win probability)
- `lag_01__CT_place_APARTMENTS`: coefficient `0.000944` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000919` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `53179`, seconds `23.00`, LSTM delta `-0.1954`

Top all feature movements:
- `lag_00__CT2__is_scoped`: contribution `-0.009840`
- `lag_13__T_A_site_active_infernos`: contribution `-0.009233`
- `lag_00__T_shots_fired_sum`: contribution `-0.007324`
- `lag_13__T3__is_scoped`: contribution `-0.007065`
- `lag_00__CT_place_APARTMENTS`: contribution `-0.006473`

Top utility-only movements:
- `lag_13__T_A_site_active_infernos`: contribution `-0.009233`
- `lag_00__CT2__flash`: contribution `-0.004868`
- `lag_13__T_active_infernos`: contribution `-0.004673`
- `lag_13__active_infernos_total`: contribution `-0.002270`

### tick `53211`, seconds `23.50`, LSTM delta `-0.1065`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.011509`
- `lag_00__CT_place_APARTMENTS`: contribution `-0.006473`
- `lag_01__CT2__is_scoped`: contribution `-0.004411`
- `lag_14__T_A_site_active_infernos`: contribution `-0.004279`
- `lag_00__T1__shots_fired`: contribution `-0.004171`

Top utility-only movements:
- `lag_14__T_A_site_active_infernos`: contribution `-0.004279`
- `lag_14__T_active_infernos`: contribution `-0.002436`
- `lag_01__CT2__flash`: contribution `-0.001918`

### tick `52411`, seconds `11.00`, LSTM delta `-0.0610`

Top all feature movements:
- `lag_10__CT_mollies_last_5s`: contribution `-0.034842`
- `lag_13__T_place_LOWERMID`: contribution `-0.002729`
- `lag_12__T_place_LOWERMID`: contribution `-0.002199`
- `lag_00__CT_place_BALCONY`: contribution `-0.001679`
- `lag_12__CT_place_LIBRARY`: contribution `+0.001608`

Top utility-only movements:
- `lag_10__CT_mollies_last_5s`: contribution `-0.034842`
- `lag_03__T_A_site_active_infernos`: contribution `-0.000644`

### tick `52091`, seconds `6.00`, LSTM delta `+0.0571`

Top all feature movements:
- `lag_10__CT_mollies_last_5s`: contribution `+0.034842`
- `lag_00__CT_mollies_last_5s`: contribution `+0.030602`
- `lag_07__CT_place_LIBRARY`: contribution `-0.001341`
- `lag_06__CT_place_LIBRARY`: contribution `-0.001019`
- `lag_02__CT_place_RUINS`: contribution `-0.000737`

Top utility-only movements:
- `lag_10__CT_mollies_last_5s`: contribution `+0.034842`
- `lag_00__CT_mollies_last_5s`: contribution `+0.030602`
- `lag_02__T4__smoke`: contribution `-0.000417`
- `lag_12__T3__flash`: contribution `-0.000407`
- `lag_12__T2__flash`: contribution `-0.000403`

### tick `51771`, seconds `1.00`, LSTM delta `-0.0525`

Top all feature movements:
- `lag_00__CT_mollies_last_5s`: contribution `-0.030602`
- `lag_02__CT_place_CTSPAWN`: contribution `-0.001166`
- `lag_02__T_closest_enemy_dist`: contribution `-0.001001`
- `lag_02__CT_closest_enemy_dist`: contribution `-0.000997`
- `lag_02__centroid_distance_xy`: contribution `-0.000950`

Top utility-only movements:
- `lag_00__CT_mollies_last_5s`: contribution `-0.030602`
- `lag_00__CT5__molly`: contribution `+0.000750`
- `lag_02__T3__flash`: contribution `-0.000547`
- `lag_02__T2__flash`: contribution `-0.000541`
- `lag_02__CT2__flash`: contribution `+0.000505`
