# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-mouz-vs-vitality-bo3-pYxpz34IEN-t8y4DgB-MSD/mouz-vs-vitality-m3-train.csv`
- round_num: `14`

## Largest probability jumps

- tick `115857`, seconds `5.00`, LSTM `0.9355`, delta `+0.0202`
- tick `115569`, seconds `0.50`, LSTM `0.9049`, delta `+0.0185`
- tick `115665`, seconds `2.00`, LSTM `0.9217`, delta `+0.0175`
- tick `116145`, seconds `9.50`, LSTM `0.9073`, delta `-0.0162`
- tick `121073`, seconds `86.50`, LSTM `0.9684`, delta `+0.0122`
- tick `120785`, seconds `82.00`, LSTM `0.9531`, delta `+0.0108`
- tick `117649`, seconds `33.00`, LSTM `0.9409`, delta `+0.0084`
- tick `120817`, seconds `82.50`, LSTM `0.9450`, delta `-0.0081`
- tick `120753`, seconds `81.50`, LSTM `0.9422`, delta `-0.0076`
- tick `121265`, seconds `89.50`, LSTM `0.9796`, delta `+0.0071`

## Top 15 local ridge features

- `lag_00__CT_place_ENTRANCE`: coefficient `0.000443`, |coef| `0.000443`
- `lag_00__T5__is_walking`: coefficient `-0.000263`, |coef| `0.000263`
- `lag_00__T3__is_walking`: coefficient `-0.000238`, |coef| `0.000238`
- `lag_00__T3__duck_amount`: coefficient `0.000237`, |coef| `0.000237`
- `lag_07__CT5__is_walking`: coefficient `0.000233`, |coef| `0.000233`
- `lag_13__T_place_IVY`: coefficient `0.000224`, |coef| `0.000224`
- `lag_00__T_walking_count`: coefficient `-0.000216`, |coef| `0.000216`
- `lag_05__CT_place_ENTRANCE`: coefficient `0.000199`, |coef| `0.000199`
- `lag_00__CT5__is_walking`: coefficient `-0.000196`, |coef| `0.000196`
- `lag_00__CT3__is_walking`: coefficient `-0.000185`, |coef| `0.000185`
- `lag_08__T_place_IVY`: coefficient `0.000184`, |coef| `0.000184`
- `lag_14__T_place_IVY`: coefficient `0.000176`, |coef| `0.000176`
- `lag_00__CT5__duck_amount`: coefficient `0.000174`, |coef| `0.000174`
- `lag_00__CT3__duck_amount`: coefficient `0.000171`, |coef| `0.000171`
- `lag_01__CT_place_CTSPAWN`: coefficient `0.000166`, |coef| `0.000166`

## Top 10 utility ridge features

- `lag_13__CT5__smoke`: coefficient `-0.000100` (lowers CT win probability)
- `lag_03__T_flash_duration_sum`: coefficient `0.000100` (raises CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000094` (raises CT win probability)
- `lag_04__smoke_inv_diff`: coefficient `0.000087` (raises CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000085` (raises CT win probability)
- `lag_01__CT4__molly`: coefficient `0.000084` (raises CT win probability)
- `lag_00__T3__flash_duration`: coefficient `-0.000082` (lowers CT win probability)
- `lag_12__CT5__smoke`: coefficient `-0.000081` (lowers CT win probability)
- `lag_03__T4__flash_duration`: coefficient `0.000080` (raises CT win probability)
- `lag_09__T_flash_duration_sum`: coefficient `0.000079` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_ENTRANCE`: coefficient `0.000443` (raises CT win probability)
- `lag_00__T5__is_walking`: coefficient `-0.000263` (lowers CT win probability)
- `lag_00__T3__is_walking`: coefficient `-0.000238` (lowers CT win probability)
- `lag_00__T3__duck_amount`: coefficient `0.000237` (raises CT win probability)
- `lag_07__CT5__is_walking`: coefficient `0.000233` (raises CT win probability)
- `lag_13__T_place_IVY`: coefficient `0.000224` (raises CT win probability)
- `lag_00__T_walking_count`: coefficient `-0.000216` (lowers CT win probability)
- `lag_05__CT_place_ENTRANCE`: coefficient `0.000199` (raises CT win probability)
- `lag_00__CT5__is_walking`: coefficient `-0.000196` (lowers CT win probability)
- `lag_00__CT3__is_walking`: coefficient `-0.000185` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `115857`, seconds `5.00`, LSTM delta `+0.0202`

Top all feature movements:
- `lag_00__CT_place_ENTRANCE`: contribution `+0.003927`
- `lag_05__CT_place_ENTRANCE`: contribution `+0.001764`
- `lag_06__CT_place_ENTRANCE`: contribution `+0.001344`
- `lag_04__CT_place_ENTRANCE`: contribution `+0.000776`
- `lag_03__CT_place_ENTRANCE`: contribution `+0.000622`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `115569`, seconds `0.50`, LSTM delta `+0.0185`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `+0.000794`
- `lag_01__T_place_TSPAWN`: contribution `+0.000726`
- `lag_01__CT_closest_enemy_dist`: contribution `+0.000532`
- `lag_01__T_closest_enemy_dist`: contribution `+0.000520`
- `lag_01__centroid_distance_xy`: contribution `+0.000471`

Top utility-only movements:
- `lag_01__smoke_inv_diff`: contribution `+0.000244`
- `lag_01__utility_inv_diff`: contribution `+0.000187`
- `lag_01__CT4__molly`: contribution `+0.000165`
- `lag_01__CT2__utility_total`: contribution `+0.000118`

### tick `115665`, seconds `2.00`, LSTM delta `+0.0175`

Top all feature movements:
- `lag_00__CT_place_ENTRANCE`: contribution `+0.003927`
- `lag_00__T3__duck_amount`: contribution `-0.000894`
- `lag_04__CT_place_CTSPAWN`: contribution `+0.000570`
- `lag_03__T_velocity_mean`: contribution `+0.000542`
- `lag_01__T3__duck_amount`: contribution `+0.000466`

Top utility-only movements:
- `lag_04__smoke_inv_diff`: contribution `+0.000223`
- `lag_04__utility_inv_diff`: contribution `+0.000155`
- `lag_04__CT4__molly`: contribution `+0.000131`

### tick `116145`, seconds `9.50`, LSTM delta `-0.0162`

Top all feature movements:
- `lag_04__T_place_DUMPSTER`: contribution `-0.001450`
- `lag_14__CT_place_ENTRANCE`: contribution `-0.000907`
- `lag_12__CT_place_ENTRANCE`: contribution `-0.000787`
- `lag_11__CT_place_TUNNELS`: contribution `-0.000750`
- `lag_00__CT5__duck_amount`: contribution `-0.000604`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `121073`, seconds `86.50`, LSTM delta `+0.0122`

Top all feature movements:
- `lag_13__T_place_IVY`: contribution `+0.001199`
- `lag_03__T_flash_duration_sum`: contribution `+0.001125`
- `lag_00__T3__duck_amount`: contribution `-0.000894`
- `lag_11__T_place_IVY`: contribution `+0.000832`
- `lag_00__CT_shots_fired_sum`: contribution `+0.000718`

Top utility-only movements:
- `lag_03__T_flash_duration_sum`: contribution `+0.001125`
- `lag_00__T3__flash_duration`: contribution `+0.000586`
- `lag_03__T4__flash_duration`: contribution `+0.000552`
- `lag_03__T1__flash_duration`: contribution `+0.000384`
- `lag_03__T3__flash_duration`: contribution `+0.000365`
