# Local Round Explainability

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-flyquest-vs-legacy-bo3-FlEa8e0vdBrf1ft_mNbThh/flyquest-vs-legacy-m2-nuke.csv`
- round_num: `6`

## Largest probability jumps

- tick `65755`, seconds `0.50`, LSTM `0.0391`, delta `-0.0527`
- tick `66971`, seconds `19.50`, LSTM `0.0377`, delta `-0.0489`
- tick `66939`, seconds `19.00`, LSTM `0.0866`, delta `-0.0448`
- tick `67579`, seconds `29.00`, LSTM `0.1064`, delta `+0.0306`
- tick `66491`, seconds `12.00`, LSTM `0.0986`, delta `+0.0283`
- tick `67835`, seconds `33.00`, LSTM `0.0646`, delta `-0.0244`
- tick `66779`, seconds `16.50`, LSTM `0.1042`, delta `+0.0242`
- tick `67739`, seconds `31.50`, LSTM `0.1075`, delta `+0.0237`
- tick `66459`, seconds `11.50`, LSTM `0.0703`, delta `+0.0235`
- tick `67867`, seconds `33.50`, LSTM `0.0452`, delta `-0.0194`

## Top 15 local ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.000590`, |coef| `0.000590`
- `lag_13__CT_place_HUTROOF`: coefficient `-0.000518`, |coef| `0.000518`
- `lag_14__CT_place_HUTROOF`: coefficient `-0.000499`, |coef| `0.000499`
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000441`, |coef| `0.000441`
- `lag_01__T_walking_count`: coefficient `0.000438`, |coef| `0.000438`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000422`, |coef| `0.000422`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000417`, |coef| `0.000417`
- `lag_08__CT_place_HUTROOF`: coefficient `0.000415`, |coef| `0.000415`
- `lag_12__CT2__duck_amount`: coefficient `0.000415`, |coef| `0.000415`
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000412`, |coef| `0.000412`
- `lag_05__T_shots_fired_sum`: coefficient `0.000402`, |coef| `0.000402`
- `lag_11__CT2__duck_amount`: coefficient `0.000389`, |coef| `0.000389`
- `lag_01__centroid_distance_xy`: coefficient `-0.000389`, |coef| `0.000389`
- `lag_05__T3__shots_fired`: coefficient `0.000379`, |coef| `0.000379`
- `lag_14__CT_place_RAFTERS`: coefficient `0.000374`, |coef| `0.000374`

## Top 10 utility ridge features

- `lag_14__CT1__flash_duration`: coefficient `0.000293` (raises CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000276` (raises CT win probability)
- `lag_15__CT1__flash_duration`: coefficient `0.000270` (raises CT win probability)
- `lag_03__T_A_site_active_infernos`: coefficient `0.000247` (raises CT win probability)
- `lag_01__CT_B_site_active_smokes`: coefficient `0.000243` (raises CT win probability)
- `lag_03__T_B_site_active_infernos`: coefficient `0.000235` (raises CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000233` (raises CT win probability)
- `lag_04__T_A_site_active_infernos`: coefficient `0.000231` (raises CT win probability)
- `lag_01__CT_A_site_active_smokes`: coefficient `0.000228` (raises CT win probability)
- `lag_01__T5__flash`: coefficient `-0.000227` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.000590` (lowers CT win probability)
- `lag_13__CT_place_HUTROOF`: coefficient `-0.000518` (lowers CT win probability)
- `lag_14__CT_place_HUTROOF`: coefficient `-0.000499` (lowers CT win probability)
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000441` (lowers CT win probability)
- `lag_01__T_walking_count`: coefficient `0.000438` (raises CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000422` (lowers CT win probability)
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000417` (lowers CT win probability)
- `lag_08__CT_place_HUTROOF`: coefficient `0.000415` (raises CT win probability)
- `lag_12__CT2__duck_amount`: coefficient `0.000415` (raises CT win probability)
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000412` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `65755`, seconds `0.50`, LSTM delta `-0.0527`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.002110`
- `lag_01__T_place_TSPAWN`: contribution `-0.001869`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.001788`
- `lag_01__T_closest_enemy_dist`: contribution `-0.001768`
- `lag_01__centroid_distance_xy`: contribution `-0.001539`

Top utility-only movements:
- `lag_01__molly_inv_diff`: contribution `-0.000769`
- `lag_01__utility_inv_diff`: contribution `-0.000511`
- `lag_01__T5__utility_total`: contribution `-0.000501`
- `lag_01__T_molly_inv`: contribution `-0.000495`
- `lag_01__T5__flash`: contribution `-0.000477`

### tick `66971`, seconds `19.50`, LSTM delta `-0.0489`

Top all feature movements:
- `lag_14__CT_place_HUTROOF`: contribution `-0.003489`
- `lag_00__T_shots_fired_sum`: contribution `-0.002212`
- `lag_14__CT_place_RAFTERS`: contribution `-0.001997`
- `lag_12__CT2__duck_amount`: contribution `-0.001580`
- `lag_06__T_shots_fired_sum`: contribution `-0.001536`

Top utility-only movements:
- `lag_15__CT1__flash_duration`: contribution `-0.000781`
- `lag_04__T_A_site_active_infernos`: contribution `-0.000686`

### tick `66939`, seconds `19.00`, LSTM delta `-0.0448`

Top all feature movements:
- `lag_13__CT_place_HUTROOF`: contribution `-0.003623`
- `lag_05__T_shots_fired_sum`: contribution `-0.003315`
- `lag_00__T_shots_fired_sum`: contribution `-0.002655`
- `lag_05__T3__shots_fired`: contribution `-0.002527`
- `lag_13__CT_place_RAFTERS`: contribution `-0.001828`

Top utility-only movements:
- `lag_14__CT1__flash_duration`: contribution `-0.000848`
- `lag_03__T_A_site_active_infernos`: contribution `-0.000735`

### tick `67579`, seconds `29.00`, LSTM delta `+0.0306`

Top all feature movements:
- `lag_10__CT_place_VENTS`: contribution `+0.003075`
- `lag_00__CT_place_HUTROOF`: contribution `+0.002144`
- `lag_11__CT2__duck_amount`: contribution `+0.000932`
- `lag_01__T_place_VENDING`: contribution `+0.000901`
- `lag_01__T4__is_walking`: contribution `+0.000842`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `66491`, seconds `12.00`, LSTM delta `+0.0283`

Top all feature movements:
- `lag_10__CT_place_HELL`: contribution `+0.002872`
- `lag_03__T_place_TROPHY`: contribution `+0.001019`
- `lag_15__CT5__duck_amount`: contribution `-0.000973`
- `lag_12__CT_place_HELL`: contribution `+0.000972`
- `lag_11__CT_place_ADMIN`: contribution `+0.000964`

Top utility-only movements:
- `lag_03__T_A_site_active_infernos`: contribution `+0.000735`
- `lag_03__T_B_site_active_infernos`: contribution `+0.000664`
