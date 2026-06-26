# Local Round Explainability

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-ninjas-in-pyjamas-vs-natus-vincere-bo3-NyGFAYn592Hu0YfljTVG0N/ninjas-in-pyjamas-vs-natus-vincere-m3-train.csv`
- round_num: `15`

## Largest probability jumps

- tick `126623`, seconds `0.50`, LSTM `0.0199`, delta `-0.0329`
- tick `127295`, seconds `11.00`, LSTM `0.0094`, delta `-0.0090`
- tick `128127`, seconds `24.00`, LSTM `0.0408`, delta `+0.0083`
- tick `128287`, seconds `26.50`, LSTM `0.0180`, delta `-0.0077`
- tick `128159`, seconds `24.50`, LSTM `0.0334`, delta `-0.0074`
- tick `128191`, seconds `25.00`, LSTM `0.0262`, delta `-0.0073`
- tick `128415`, seconds `28.50`, LSTM `0.0074`, delta `-0.0059`
- tick `128031`, seconds `22.50`, LSTM `0.0318`, delta `-0.0056`
- tick `126655`, seconds `1.00`, LSTM `0.0150`, delta `-0.0048`
- tick `128383`, seconds `28.00`, LSTM `0.0133`, delta `-0.0046`

## Top 15 local ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000271`, |coef| `0.000271`
- `lag_00__CT_velocity_mean`: coefficient `-0.000253`, |coef| `0.000253`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000224`, |coef| `0.000224`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000223`, |coef| `0.000223`
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000222`, |coef| `0.000222`
- `lag_00__T_velocity_mean`: coefficient `-0.000208`, |coef| `0.000208`
- `lag_01__centroid_distance_xy`: coefficient `-0.000204`, |coef| `0.000204`
- `lag_01__utility_inv_diff`: coefficient `0.000192`, |coef| `0.000192`
- `lag_01__armor_diff`: coefficient `0.000181`, |coef| `0.000181`
- `lag_01__smoke_inv_diff`: coefficient `0.000180`, |coef| `0.000180`
- `lag_01__CT_armor_sum`: coefficient `0.000161`, |coef| `0.000161`
- `lag_14__T2__flash_duration`: coefficient `0.000156`, |coef| `0.000156`
- `lag_00__T5__smoke`: coefficient `0.000151`, |coef| `0.000151`
- `lag_01__molly_inv_diff`: coefficient `0.000146`, |coef| `0.000146`
- `lag_01__flash_inv_diff`: coefficient `0.000145`, |coef| `0.000145`

## Top 10 utility ridge features

- `lag_01__utility_inv_diff`: coefficient `0.000192` (raises CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000180` (raises CT win probability)
- `lag_14__T2__flash_duration`: coefficient `0.000156` (raises CT win probability)
- `lag_00__T5__smoke`: coefficient `0.000151` (raises CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000146` (raises CT win probability)
- `lag_01__flash_inv_diff`: coefficient `0.000145` (raises CT win probability)
- `lag_01__T4__utility_total`: coefficient `-0.000132` (lowers CT win probability)
- `lag_13__T2__flash_duration`: coefficient `0.000131` (raises CT win probability)
- `lag_01__T4__flash`: coefficient `-0.000129` (lowers CT win probability)
- `lag_01__T_utility_inv`: coefficient `-0.000127` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000271` (lowers CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000253` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000224` (lowers CT win probability)
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000223` (lowers CT win probability)
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000222` (lowers CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000208` (lowers CT win probability)
- `lag_01__centroid_distance_xy`: coefficient `-0.000204` (lowers CT win probability)
- `lag_01__armor_diff`: coefficient `0.000181` (raises CT win probability)
- `lag_01__CT_armor_sum`: coefficient `0.000161` (raises CT win probability)
- `lag_01__T5__has_bomb`: coefficient `-0.000138` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `126623`, seconds `0.50`, LSTM delta `-0.0329`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.001297`
- `lag_01__T_place_TSPAWN`: contribution `-0.000991`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.000912`
- `lag_00__CT_velocity_mean`: contribution `-0.000901`
- `lag_01__T_closest_enemy_dist`: contribution `-0.000889`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `-0.000632`
- `lag_01__smoke_inv_diff`: contribution `-0.000572`
- `lag_01__molly_inv_diff`: contribution `-0.000408`
- `lag_01__flash_inv_diff`: contribution `-0.000327`
- `lag_00__T5__smoke`: contribution `-0.000326`

### tick `127295`, seconds `11.00`, LSTM delta `-0.0090`

Top all feature movements:
- `lag_01__CT_place_ELECTRICALBOX`: contribution `-0.001005`
- `lag_15__CT_place_ENTRANCE`: contribution `-0.000529`
- `lag_06__T_place_DUMPSTER`: contribution `+0.000526`
- `lag_12__T_place_TSTAIRS`: contribution `-0.000472`
- `lag_00__CT_place_BACKOFB`: contribution `-0.000398`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `128127`, seconds `24.00`, LSTM delta `+0.0083`

Top all feature movements:
- `lag_01__T_place_DUMPSTER`: contribution `+0.000795`
- `lag_12__T2__flash_duration`: contribution `+0.000377`
- `lag_08__CT_place_BACKOFB`: contribution `+0.000322`
- `lag_01__T_place_TSPAWN`: contribution `-0.000212`
- `lag_06__T2__duck_amount`: contribution `+0.000202`

Top utility-only movements:
- `lag_12__T2__flash_duration`: contribution `+0.000377`

### tick `128287`, seconds `26.50`, LSTM delta `-0.0077`

Top all feature movements:
- `lag_06__T_place_DUMPSTER`: contribution `-0.000526`
- `lag_01__CT_place_BACKOFB`: contribution `-0.000507`
- `lag_00__T_place_LONGDOG`: contribution `-0.000467`
- `lag_13__CT_place_BACKOFB`: contribution `-0.000352`
- `lag_05__T_flashed_players`: contribution `-0.000212`

Top utility-only movements:
- `lag_10__T_A_site_active_infernos`: contribution `-0.000207`

### tick `128159`, seconds `24.50`, LSTM delta `-0.0074`

Top all feature movements:
- `lag_13__T2__flash_duration`: contribution `-0.000945`
- `lag_02__T_place_DUMPSTER`: contribution `-0.000753`
- `lag_09__CT_place_BACKOFB`: contribution `-0.000309`
- `lag_06__T2__duck_amount`: contribution `-0.000202`
- `lag_04__T1__is_walking`: contribution `-0.000160`

Top utility-only movements:
- `lag_13__T2__flash_duration`: contribution `-0.000945`
- `lag_13__T_flash_duration_sum`: contribution `-0.000150`
- `lag_06__T_A_site_active_infernos`: contribution `-0.000127`
