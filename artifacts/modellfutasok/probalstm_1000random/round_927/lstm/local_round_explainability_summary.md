# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-saw-bo3-tIR5RlOpBrnlpEe6MBVyNd/heroic-vs-saw-m2-train.csv`
- round_num: `3`

## Largest probability jumps

- tick `10446`, seconds `0.50`, LSTM `0.0189`, delta `-0.0245`
- tick `14542`, seconds `64.50`, LSTM `0.0150`, delta `-0.0087`
- tick `14350`, seconds `61.50`, LSTM `0.0221`, delta `+0.0079`
- tick `11214`, seconds `12.50`, LSTM `0.0058`, delta `-0.0056`
- tick `10766`, seconds `5.50`, LSTM `0.0128`, delta `-0.0049`
- tick `14798`, seconds `68.50`, LSTM `0.0038`, delta `-0.0044`
- tick `11918`, seconds `23.50`, LSTM `0.0169`, delta `+0.0040`
- tick `10478`, seconds `1.00`, LSTM `0.0152`, delta `-0.0037`
- tick `12526`, seconds `33.00`, LSTM `0.0164`, delta `-0.0036`
- tick `10574`, seconds `2.50`, LSTM `0.0205`, delta `+0.0032`

## Top 15 local ridge features

- `lag_00__CT_flashes_last_5s`: coefficient `-0.000319`, |coef| `0.000319`
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000172`, |coef| `0.000172`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000160`, |coef| `0.000160`
- `lag_00__CT_velocity_mean`: coefficient `-0.000141`, |coef| `0.000141`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000139`, |coef| `0.000139`
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000135`, |coef| `0.000135`
- `lag_01__centroid_distance_xy`: coefficient `-0.000131`, |coef| `0.000131`
- `lag_00__T_velocity_mean`: coefficient `-0.000130`, |coef| `0.000130`
- `lag_01__armor_diff`: coefficient `0.000129`, |coef| `0.000129`
- `lag_00__CT1__is_walking`: coefficient `0.000128`, |coef| `0.000128`
- `lag_05__T1__flash_duration`: coefficient `-0.000114`, |coef| `0.000114`
- `lag_01__CT_armor_sum`: coefficient `0.000108`, |coef| `0.000108`
- `lag_01__smoke_inv_diff`: coefficient `0.000102`, |coef| `0.000102`
- `lag_01__equip_diff`: coefficient `0.000102`, |coef| `0.000102`
- `lag_01__CT_flashes_last_5s`: coefficient `-0.000099`, |coef| `0.000099`

## Top 10 utility ridge features

- `lag_00__CT_flashes_last_5s`: coefficient `-0.000319` (lowers CT win probability)
- `lag_05__T1__flash_duration`: coefficient `-0.000114` (lowers CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000102` (raises CT win probability)
- `lag_01__CT_flashes_last_5s`: coefficient `-0.000099` (lowers CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000096` (raises CT win probability)
- `lag_01__T5__utility_total`: coefficient `-0.000087` (lowers CT win probability)
- `lag_01__T5__flash`: coefficient `-0.000086` (lowers CT win probability)
- `lag_01__T_smoke_inv`: coefficient `-0.000082` (lowers CT win probability)
- `lag_13__T1__flash_duration`: coefficient `-0.000081` (lowers CT win probability)
- `lag_01__T1__utility_total`: coefficient `-0.000080` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000172` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000160` (lowers CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000141` (lowers CT win probability)
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000139` (lowers CT win probability)
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000135` (lowers CT win probability)
- `lag_01__centroid_distance_xy`: coefficient `-0.000131` (lowers CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000130` (lowers CT win probability)
- `lag_01__armor_diff`: coefficient `0.000129` (raises CT win probability)
- `lag_00__CT1__is_walking`: coefficient `0.000128` (raises CT win probability)
- `lag_01__CT_armor_sum`: coefficient `0.000108` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `10446`, seconds `0.50`, LSTM delta `-0.0245`

Top all feature movements:
- `lag_00__CT_flashes_last_5s`: contribution `-0.003504`
- `lag_01__CT_place_CTSPAWN`: contribution `-0.000824`
- `lag_01__T_place_TSPAWN`: contribution `-0.000710`
- `lag_01__T_closest_enemy_dist`: contribution `-0.000556`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.000550`

Top utility-only movements:
- `lag_00__CT_flashes_last_5s`: contribution `-0.003504`
- `lag_01__smoke_inv_diff`: contribution `-0.000260`
- `lag_01__utility_inv_diff`: contribution `-0.000254`
- `lag_01__T5__utility_total`: contribution `-0.000201`
- `lag_01__T_smoke_inv`: contribution `-0.000187`

### tick `14542`, seconds `64.50`, LSTM delta `-0.0087`

Top all feature movements:
- `lag_05__T1__flash_duration`: contribution `-0.000711`
- `lag_00__CT_place_BACKOFB`: contribution `-0.000354`
- `lag_08__CT_flashed_players`: contribution `-0.000312`
- `lag_00__CT1__is_walking`: contribution `+0.000299`
- `lag_08__CT4__flash_duration`: contribution `-0.000287`

Top utility-only movements:
- `lag_05__T1__flash_duration`: contribution `-0.000711`
- `lag_08__CT4__flash_duration`: contribution `-0.000287`
- `lag_07__T1__flash_duration`: contribution `-0.000249`
- `lag_08__CT_flash_duration_sum`: contribution `-0.000225`
- `lag_00__CT2__flash_duration`: contribution `-0.000176`

### tick `14350`, seconds `61.50`, LSTM delta `+0.0079`

Top all feature movements:
- `lag_02__CT_flashed_players`: contribution `+0.000517`
- `lag_02__CT_flash_duration_sum`: contribution `+0.000353`
- `lag_02__CT4__flash_duration`: contribution `+0.000320`
- `lag_00__CT1__is_walking`: contribution `+0.000299`
- `lag_02__CT2__flash_duration`: contribution `+0.000225`

Top utility-only movements:
- `lag_02__CT_flash_duration_sum`: contribution `+0.000353`
- `lag_02__CT4__flash_duration`: contribution `+0.000320`
- `lag_02__CT2__flash_duration`: contribution `+0.000225`
- `lag_00__T1__flash_duration`: contribution `+0.000191`

### tick `11214`, seconds `12.50`, LSTM delta `-0.0056`

Top all feature movements:
- `lag_02__CT_place_ELECTRICALBOX`: contribution `-0.000447`
- `lag_12__T_place_DUMPSTER`: contribution `-0.000431`
- `lag_14__CT_flashes_last_5s`: contribution `-0.000428`
- `lag_11__T_place_TSTAIRS`: contribution `-0.000305`
- `lag_08__T_place_TSTAIRS`: contribution `-0.000216`

Top utility-only movements:
- `lag_14__CT_flashes_last_5s`: contribution `-0.000428`
- `lag_03__T_A_site_active_infernos`: contribution `-0.000098`

### tick `10766`, seconds `5.50`, LSTM delta `-0.0049`

Top all feature movements:
- `lag_00__CT_flashes_last_5s`: contribution `+0.003504`
- `lag_10__CT_flashes_last_5s`: contribution `-0.000622`
- `lag_07__CT_place_ENTRANCE`: contribution `-0.000505`
- `lag_05__CT_place_ENTRANCE`: contribution `-0.000478`
- `lag_02__CT_place_ENTRANCE`: contribution `-0.000380`

Top utility-only movements:
- `lag_00__CT_flashes_last_5s`: contribution `+0.003504`
- `lag_10__CT_flashes_last_5s`: contribution `-0.000622`
