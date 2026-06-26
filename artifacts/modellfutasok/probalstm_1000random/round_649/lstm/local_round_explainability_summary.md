# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-faze-vs-inner-circle-bo3-runM3q2zOKSAHTeRui0Q2h/faze-vs-inner-circle-m2-nuke.csv`
- round_num: `2`

## Largest probability jumps

- tick `5821`, seconds `18.50`, LSTM `0.2120`, delta `-0.1494`
- tick `5341`, seconds `11.00`, LSTM `0.3013`, delta `+0.0755`
- tick `4669`, seconds `0.50`, LSTM `0.2046`, delta `-0.0741`
- tick `5565`, seconds `14.50`, LSTM `0.3280`, delta `-0.0644`
- tick `6237`, seconds `25.00`, LSTM `0.0616`, delta `-0.0588`
- tick `5789`, seconds `18.00`, LSTM `0.3614`, delta `-0.0442`
- tick `5469`, seconds `13.00`, LSTM `0.3842`, delta `+0.0359`
- tick `5917`, seconds `20.00`, LSTM `0.1567`, delta `-0.0337`
- tick `5405`, seconds `12.00`, LSTM `0.3569`, delta `+0.0324`
- tick `4989`, seconds `5.50`, LSTM `0.2009`, delta `-0.0267`

## Top 15 local ridge features

- `lag_00__CT2__is_scoped`: coefficient `0.001734`, |coef| `0.001734`
- `lag_00__CT_he_last_5s`: coefficient `-0.001510`, |coef| `0.001510`
- `lag_03__CT_place_VENTS`: coefficient `-0.001351`, |coef| `0.001351`
- `lag_11__CT_he_last_5s`: coefficient `-0.001325`, |coef| `0.001325`
- `lag_15__T_place_SILO`: coefficient `-0.001281`, |coef| `0.001281`
- `lag_08__CT_place_CONTROL`: coefficient `-0.001168`, |coef| `0.001168`
- `lag_15__T_place_ROOF`: coefficient `0.001111`, |coef| `0.001111`
- `lag_13__CT2__is_scoped`: coefficient `-0.000935`, |coef| `0.000935`
- `lag_09__CT2__is_scoped`: coefficient `0.000931`, |coef| `0.000931`
- `lag_08__CT2__is_scoped`: coefficient `0.000926`, |coef| `0.000926`
- `lag_05__CT2__is_scoped`: coefficient `-0.000897`, |coef| `0.000897`
- `lag_10__CT_he_last_5s`: coefficient `-0.000870`, |coef| `0.000870`
- `lag_00__T_shots_fired_sum`: coefficient `-0.000859`, |coef| `0.000859`
- `lag_01__CT_he_last_5s`: coefficient `-0.000829`, |coef| `0.000829`
- `lag_01__CT_place_CONTROL`: coefficient `0.000802`, |coef| `0.000802`

## Top 10 utility ridge features

- `lag_00__CT_he_last_5s`: coefficient `-0.001510` (lowers CT win probability)
- `lag_11__CT_he_last_5s`: coefficient `-0.001325` (lowers CT win probability)
- `lag_10__CT_he_last_5s`: coefficient `-0.000870` (lowers CT win probability)
- `lag_01__CT_he_last_5s`: coefficient `-0.000829` (lowers CT win probability)
- `lag_13__CT_he_last_5s`: coefficient `-0.000553` (lowers CT win probability)
- `lag_00__T2__molly`: coefficient `0.000550` (raises CT win probability)
- `lag_01__CT_flash_alpha_mean`: coefficient `0.000537` (raises CT win probability)
- `lag_08__T1__smoke`: coefficient `0.000510` (raises CT win probability)
- `lag_05__T_B_site_active_smokes`: coefficient `-0.000471` (lowers CT win probability)
- `lag_07__CT_he_last_5s`: coefficient `0.000458` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT2__is_scoped`: coefficient `0.001734` (raises CT win probability)
- `lag_03__CT_place_VENTS`: coefficient `-0.001351` (lowers CT win probability)
- `lag_15__T_place_SILO`: coefficient `-0.001281` (lowers CT win probability)
- `lag_08__CT_place_CONTROL`: coefficient `-0.001168` (lowers CT win probability)
- `lag_15__T_place_ROOF`: coefficient `0.001111` (raises CT win probability)
- `lag_13__CT2__is_scoped`: coefficient `-0.000935` (lowers CT win probability)
- `lag_09__CT2__is_scoped`: coefficient `0.000931` (raises CT win probability)
- `lag_08__CT2__is_scoped`: coefficient `0.000926` (raises CT win probability)
- `lag_05__CT2__is_scoped`: coefficient `-0.000897` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.000859` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `5821`, seconds `18.50`, LSTM delta `-0.1494`

Top all feature movements:
- `lag_08__CT_place_CONTROL`: contribution `-0.012127`
- `lag_03__CT_place_VENTS`: contribution `-0.011338`
- `lag_00__CT2__is_scoped`: contribution `-0.010611`
- `lag_15__T_place_SILO`: contribution `-0.008705`
- `lag_15__T_place_ROOF`: contribution `-0.006294`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `5341`, seconds `11.00`, LSTM delta `+0.0755`

Top all feature movements:
- `lag_11__CT_he_last_5s`: contribution `+0.024306`
- `lag_05__CT2__is_scoped`: contribution `-0.005488`
- `lag_00__T_place_ROOF`: contribution `+0.003622`
- `lag_00__T_place_SILO`: contribution `+0.003495`
- `lag_12__CT_place_HELL`: contribution `+0.002881`

Top utility-only movements:
- `lag_11__CT_he_last_5s`: contribution `+0.024306`

### tick `4669`, seconds `0.50`, LSTM delta `-0.0741`

Top all feature movements:
- `lag_00__CT_he_last_5s`: contribution `-0.027707`
- `lag_01__CT_place_CTSPAWN`: contribution `-0.003451`
- `lag_01__T_closest_enemy_dist`: contribution `-0.002891`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.002886`
- `lag_01__centroid_distance_xy`: contribution `-0.002558`

Top utility-only movements:
- `lag_00__CT_he_last_5s`: contribution `-0.027707`
- `lag_01__T_smoke_inv`: contribution `-0.000831`
- `lag_01__smoke_inv_diff`: contribution `-0.000705`
- `lag_01__CT_flash_alpha_mean`: contribution `-0.000700`
- `lag_01__T1__utility_total`: contribution `-0.000565`

### tick `5565`, seconds `14.50`, LSTM delta `-0.0644`

Top all feature movements:
- `lag_00__CT2__is_scoped`: contribution `-0.010611`
- `lag_09__CT2__is_scoped`: contribution `-0.005698`
- `lag_05__CT2__is_scoped`: contribution `-0.005488`
- `lag_13__CT_place_RAFTERS`: contribution `+0.003852`
- `lag_13__CT_place_ADMIN`: contribution `-0.003207`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `6237`, seconds `25.00`, LSTM delta `-0.0588`

Top all feature movements:
- `lag_04__T_place_GARAGE`: contribution `-0.009111`
- `lag_03__CT_place_OBSERVATION`: contribution `-0.008235`
- `lag_07__CT_place_CONTROL`: contribution `-0.008037`
- `lag_13__CT2__is_scoped`: contribution `+0.005724`
- `lag_05__T_place_GARAGE`: contribution `+0.004607`

Top utility-only movements:
- No utility movement among the top local contributors.
