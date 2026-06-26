# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-nemiga-bo3-HBPh0RFmxqP1tE9QMaq3nA/heroic-vs-nemiga-m2-mirage.csv`
- round_num: `4`

## Largest probability jumps

- tick `22038`, seconds `32.00`, LSTM `0.0291`, delta `-0.1107`
- tick `20022`, seconds `0.50`, LSTM `0.0992`, delta `-0.0673`
- tick `21366`, seconds `21.50`, LSTM `0.0756`, delta `-0.0499`
- tick `21590`, seconds `25.00`, LSTM `0.1177`, delta `+0.0455`
- tick `22006`, seconds `31.50`, LSTM `0.1398`, delta `+0.0332`
- tick `21718`, seconds `27.00`, LSTM `0.1309`, delta `+0.0303`
- tick `20406`, seconds `6.50`, LSTM `0.1347`, delta `+0.0291`
- tick `21526`, seconds `24.00`, LSTM `0.0630`, delta `-0.0223`
- tick `21398`, seconds `22.00`, LSTM `0.0961`, delta `+0.0205`
- tick `21814`, seconds `28.50`, LSTM `0.1153`, delta `-0.0199`

## Top 15 local ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000801`, |coef| `0.000801`
- `lag_15__CT_place_TRUCK`: coefficient `-0.000784`, |coef| `0.000784`
- `lag_14__CT_place_SCAFFOLDING`: coefficient `0.000765`, |coef| `0.000765`
- `lag_00__CT_place_SCAFFOLDING`: coefficient `-0.000691`, |coef| `0.000691`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000627`, |coef| `0.000627`
- `lag_00__CT_velocity_mean`: coefficient `-0.000623`, |coef| `0.000623`
- `lag_08__T_place_CONNECTOR`: coefficient `-0.000591`, |coef| `0.000591`
- `lag_00__T_he_last_5s`: coefficient `-0.000548`, |coef| `0.000548`
- `lag_08__CT_place_JUNGLE`: coefficient `-0.000489`, |coef| `0.000489`
- `lag_12__CT_place_TRUCK`: coefficient `0.000488`, |coef| `0.000488`
- `lag_08__CT_place_SCAFFOLDING`: coefficient `0.000483`, |coef| `0.000483`
- `lag_00__T_velocity_mean`: coefficient `-0.000457`, |coef| `0.000457`
- `lag_01__CT_place_TRUCK`: coefficient `0.000440`, |coef| `0.000440`
- `lag_01__centroid_distance_xy`: coefficient `-0.000415`, |coef| `0.000415`
- `lag_10__T_he_last_5s`: coefficient `0.000405`, |coef| `0.000405`

## Top 10 utility ridge features

- `lag_00__T_he_last_5s`: coefficient `-0.000548` (lowers CT win probability)
- `lag_10__T_he_last_5s`: coefficient `0.000405` (raises CT win probability)
- `lag_01__T2__flash`: coefficient `-0.000381` (lowers CT win probability)
- `lag_00__T1__smoke`: coefficient `0.000367` (raises CT win probability)
- `lag_01__T1__flash`: coefficient `-0.000356` (lowers CT win probability)
- `lag_00__CT2__smoke`: coefficient `0.000351` (raises CT win probability)
- `lag_01__T2__utility_total`: coefficient `-0.000349` (lowers CT win probability)
- `lag_15__T_A_site_active_infernos`: coefficient `0.000343` (raises CT win probability)
- `lag_01__flash_inv_diff`: coefficient `0.000306` (raises CT win probability)
- `lag_01__CT_flash_alpha_mean`: coefficient `0.000303` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000801` (lowers CT win probability)
- `lag_15__CT_place_TRUCK`: coefficient `-0.000784` (lowers CT win probability)
- `lag_14__CT_place_SCAFFOLDING`: coefficient `0.000765` (raises CT win probability)
- `lag_00__CT_place_SCAFFOLDING`: coefficient `-0.000691` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000627` (lowers CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000623` (lowers CT win probability)
- `lag_08__T_place_CONNECTOR`: coefficient `-0.000591` (lowers CT win probability)
- `lag_08__CT_place_JUNGLE`: coefficient `-0.000489` (lowers CT win probability)
- `lag_12__CT_place_TRUCK`: coefficient `0.000488` (raises CT win probability)
- `lag_08__CT_place_SCAFFOLDING`: coefficient `0.000483` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `22038`, seconds `32.00`, LSTM delta `-0.1107`

Top all feature movements:
- `lag_14__CT_place_SCAFFOLDING`: contribution `-0.015967`
- `lag_15__CT_place_TRUCK`: contribution `-0.005056`
- `lag_12__CT_place_TRUCK`: contribution `-0.003151`
- `lag_08__CT_place_JUNGLE`: contribution `-0.003137`
- `lag_08__T_place_CONNECTOR`: contribution `-0.002863`

Top utility-only movements:
- `lag_15__T_A_site_active_infernos`: contribution `-0.001022`

### tick `20022`, seconds `0.50`, LSTM delta `-0.0673`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.003829`
- `lag_01__T_place_TSPAWN`: contribution `-0.002776`
- `lag_00__CT_velocity_mean`: contribution `-0.002166`
- `lag_00__T_velocity_mean`: contribution `-0.001331`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.001100`

Top utility-only movements:
- `lag_01__T2__flash`: contribution `-0.000848`
- `lag_01__T2__utility_total`: contribution `-0.000841`
- `lag_00__T1__smoke`: contribution `-0.000793`
- `lag_01__T1__flash`: contribution `-0.000730`
- `lag_01__utility_inv_diff`: contribution `-0.000648`

### tick `21366`, seconds `21.50`, LSTM delta `-0.0499`

Top all feature movements:
- `lag_00__CT_place_SCAFFOLDING`: contribution `-0.014420`
- `lag_15__CT_place_TRUCK`: contribution `-0.005056`
- `lag_01__CT_place_TRUCK`: contribution `-0.002836`
- `lag_10__CT_place_JUNGLE`: contribution `-0.001856`
- `lag_03__CT_place_TRUCK`: contribution `-0.001856`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `21590`, seconds `25.00`, LSTM delta `+0.0455`

Top all feature movements:
- `lag_00__CT_place_SCAFFOLDING`: contribution `+0.014420`
- `lag_07__CT_place_SCAFFOLDING`: contribution `+0.007657`
- `lag_15__CT_place_TRUCK`: contribution `+0.005056`
- `lag_04__CT_place_SCAFFOLDING`: contribution `+0.004076`
- `lag_01__CT_place_TRUCK`: contribution `+0.002836`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `22006`, seconds `31.50`, LSTM delta `+0.0332`

Top all feature movements:
- `lag_13__CT_place_SCAFFOLDING`: contribution `-0.006677`
- `lag_15__CT_place_TRUCK`: contribution `+0.005056`
- `lag_11__CT_place_TRUCK`: contribution `-0.001760`
- `lag_15__CT_place_JUNGLE`: contribution `+0.001451`
- `lag_08__T_shots_fired_sum`: contribution `+0.001394`

Top utility-only movements:
- No utility movement among the top local contributors.
