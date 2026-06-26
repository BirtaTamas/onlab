# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-tyloo-vs-metizport-mirage-uJE2h4ym3PvBPopNN8-YOA/tyloo-vs-metizport-mirage.csv`
- round_num: `3`

## Largest probability jumps

- tick `22522`, seconds `0.50`, LSTM `0.0151`, delta `-0.0277`
- tick `24026`, seconds `24.00`, LSTM `0.0281`, delta `-0.0086`
- tick `24666`, seconds `34.00`, LSTM `0.0045`, delta `-0.0068`
- tick `23450`, seconds `15.00`, LSTM `0.0298`, delta `-0.0062`
- tick `23418`, seconds `14.50`, LSTM `0.0360`, delta `+0.0060`
- tick `24506`, seconds `31.50`, LSTM `0.0153`, delta `-0.0053`
- tick `22554`, seconds `1.00`, LSTM `0.0098`, delta `-0.0053`
- tick `23194`, seconds `11.00`, LSTM `0.0318`, delta `+0.0052`
- tick `22938`, seconds `7.00`, LSTM `0.0201`, delta `+0.0051`
- tick `23002`, seconds `8.00`, LSTM `0.0265`, delta `+0.0051`

## Top 15 local ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000269`, |coef| `0.000269`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000257`, |coef| `0.000257`
- `lag_00__CT_velocity_mean`: coefficient `-0.000213`, |coef| `0.000213`
- `lag_01__T_flash_alpha_mean`: coefficient `0.000198`, |coef| `0.000198`
- `lag_00__T_velocity_mean`: coefficient `-0.000188`, |coef| `0.000188`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000168`, |coef| `0.000168`
- `lag_00__T2__has_bomb`: coefficient `-0.000165`, |coef| `0.000165`
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000164`, |coef| `0.000164`
- `lag_01__armor_diff`: coefficient `0.000163`, |coef| `0.000163`
- `lag_01__centroid_distance_xy`: coefficient `-0.000161`, |coef| `0.000161`
- `lag_01__CT_armor_sum`: coefficient `0.000145`, |coef| `0.000145`
- `lag_00__bomb_events_last_5s`: coefficient `-0.000125`, |coef| `0.000125`
- `lag_02__CT_place_CTSPAWN`: coefficient `-0.000122`, |coef| `0.000122`
- `lag_01__CT_mean_Y`: coefficient `0.000119`, |coef| `0.000119`
- `lag_01__molly_inv_diff`: coefficient `0.000117`, |coef| `0.000117`

## Top 10 utility ridge features

- `lag_01__T_flash_alpha_mean`: coefficient `0.000198` (raises CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000117` (raises CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000107` (raises CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000107` (raises CT win probability)
- `lag_01__T1__molly`: coefficient `-0.000099` (lowers CT win probability)
- `lag_02__T_flash_alpha_mean`: coefficient `0.000092` (raises CT win probability)
- `lag_00__T3__smoke`: coefficient `0.000082` (raises CT win probability)
- `lag_01__T_molly_inv`: coefficient `-0.000081` (lowers CT win probability)
- `lag_01__T4__molly`: coefficient `-0.000078` (lowers CT win probability)
- `lag_01__T2__molly`: coefficient `-0.000078` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000269` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000257` (lowers CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000213` (lowers CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000188` (lowers CT win probability)
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000168` (lowers CT win probability)
- `lag_00__T2__has_bomb`: coefficient `-0.000165` (lowers CT win probability)
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000164` (lowers CT win probability)
- `lag_01__armor_diff`: coefficient `0.000163` (raises CT win probability)
- `lag_01__centroid_distance_xy`: coefficient `-0.000161` (lowers CT win probability)
- `lag_01__CT_armor_sum`: coefficient `0.000145` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `22522`, seconds `0.50`, LSTM delta `-0.0277`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.001285`
- `lag_01__T_place_TSPAWN`: contribution `-0.001137`
- `lag_00__CT_velocity_mean`: contribution `-0.000735`
- `lag_01__T_flash_alpha_mean`: contribution `-0.000675`
- `lag_00__T_velocity_mean`: contribution `-0.000545`

Top utility-only movements:
- `lag_01__T_flash_alpha_mean`: contribution `-0.000675`
- `lag_01__smoke_inv_diff`: contribution `-0.000272`
- `lag_01__molly_inv_diff`: contribution `-0.000255`
- `lag_01__utility_inv_diff`: contribution `-0.000234`

### tick `24026`, seconds `24.00`, LSTM delta `-0.0086`

Top all feature movements:
- `lag_11__CT_place_SNIPERSNEST`: contribution `-0.000434`
- `lag_00__T3__duck_amount`: contribution `-0.000268`
- `lag_00__T1__is_walking`: contribution `-0.000257`
- `lag_07__T_place_CATWALK`: contribution `-0.000245`
- `lag_00__CT3__is_walking`: contribution `-0.000236`

Top utility-only movements:
- `lag_13__T_B_site_active_infernos`: contribution `-0.000152`

### tick `24666`, seconds `34.00`, LSTM delta `-0.0068`

Top all feature movements:
- `lag_11__T2__flash_duration`: contribution `-0.000517`
- `lag_06__T_place_CONNECTOR`: contribution `-0.000376`
- `lag_09__CT_place_PALACEINTERIOR`: contribution `-0.000317`
- `lag_00__CT_place_APARTMENTS`: contribution `-0.000190`
- `lag_10__CT5__duck_amount`: contribution `-0.000180`

Top utility-only movements:
- `lag_11__T2__flash_duration`: contribution `-0.000517`
- `lag_15__T1__smoke`: contribution `-0.000133`
- `lag_00__T_B_site_active_infernos`: contribution `-0.000106`

### tick `23450`, seconds `15.00`, LSTM delta `-0.0062`

Top all feature movements:
- `lag_04__T_place_HOUSE`: contribution `-0.000404`
- `lag_04__CT_place_TRUCK`: contribution `-0.000341`
- `lag_00__CT1__is_scoped`: contribution `-0.000316`
- `lag_15__CT_place_SNIPERSNEST`: contribution `+0.000260`
- `lag_00__T1__is_walking`: contribution `-0.000257`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `23418`, seconds `14.50`, LSTM delta `+0.0060`

Top all feature movements:
- `lag_03__CT_place_TRUCK`: contribution `+0.000383`
- `lag_14__CT_place_SNIPERSNEST`: contribution `+0.000333`
- `lag_00__CT1__is_scoped`: contribution `+0.000316`
- `lag_10__CT1__is_scoped`: contribution `+0.000307`
- `lag_04__CT_place_PALACEINTERIOR`: contribution `+0.000278`

Top utility-only movements:
- No utility movement among the top local contributors.
