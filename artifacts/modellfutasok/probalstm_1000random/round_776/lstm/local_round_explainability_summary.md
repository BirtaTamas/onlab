# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-natus-vincere-vs-spirit-bo3-cW0x-KCT4cbPLaZUAvb08Z/natus-vincere-vs-spirit-m2-ancient.csv`
- round_num: `7`

## Largest probability jumps

- tick `46169`, seconds `0.50`, LSTM `0.0278`, delta `-0.0378`
- tick `47321`, seconds `18.50`, LSTM `0.0139`, delta `-0.0181`
- tick `46201`, seconds `1.00`, LSTM `0.0223`, delta `-0.0055`
- tick `46713`, seconds `9.00`, LSTM `0.0193`, delta `-0.0054`
- tick `47097`, seconds `15.00`, LSTM `0.0246`, delta `+0.0049`
- tick `47033`, seconds `14.00`, LSTM `0.0227`, delta `+0.0046`
- tick `46777`, seconds `10.00`, LSTM `0.0214`, delta `+0.0044`
- tick `46937`, seconds `12.50`, LSTM `0.0194`, delta `-0.0040`
- tick `47417`, seconds `20.00`, LSTM `0.0053`, delta `-0.0031`
- tick `47225`, seconds `17.00`, LSTM `0.0316`, delta `+0.0030`

## Top 15 local ridge features

- `lag_01__CT_place_TSIDEUPPER`: coefficient `-0.000423`, |coef| `0.000423`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000284`, |coef| `0.000284`
- `lag_01__CT_place_MIDDLE`: coefficient `-0.000242`, |coef| `0.000242`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000236`, |coef| `0.000236`
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000231`, |coef| `0.000231`
- `lag_01__centroid_distance_xy`: coefficient `-0.000216`, |coef| `0.000216`
- `lag_00__T_velocity_mean`: coefficient `-0.000202`, |coef| `0.000202`
- `lag_01__utility_inv_diff`: coefficient `0.000191`, |coef| `0.000191`
- `lag_01__molly_inv_diff`: coefficient `0.000184`, |coef| `0.000184`
- `lag_01__smoke_inv_diff`: coefficient `0.000180`, |coef| `0.000180`
- `lag_08__T_shots_fired_sum`: coefficient `0.000176`, |coef| `0.000176`
- `lag_01__equip_diff`: coefficient `0.000171`, |coef| `0.000171`
- `lag_00__CT_velocity_mean`: coefficient `-0.000167`, |coef| `0.000167`
- `lag_01__armor_diff`: coefficient `0.000162`, |coef| `0.000162`
- `lag_00__CT_place_MIDDLE`: coefficient `0.000150`, |coef| `0.000150`

## Top 10 utility ridge features

- `lag_01__utility_inv_diff`: coefficient `0.000191` (raises CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000184` (raises CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000180` (raises CT win probability)
- `lag_01__T1__utility_total`: coefficient `-0.000149` (lowers CT win probability)
- `lag_01__T_utility_inv`: coefficient `-0.000148` (lowers CT win probability)
- `lag_01__T_smoke_inv`: coefficient `-0.000148` (lowers CT win probability)
- `lag_01__T_molly_inv`: coefficient `-0.000141` (lowers CT win probability)
- `lag_01__T5__utility_total`: coefficient `-0.000139` (lowers CT win probability)
- `lag_01__T5__flash`: coefficient `-0.000129` (lowers CT win probability)
- `lag_01__T1__flash`: coefficient `-0.000127` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_TSIDEUPPER`: coefficient `-0.000423` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000284` (lowers CT win probability)
- `lag_01__CT_place_MIDDLE`: coefficient `-0.000242` (lowers CT win probability)
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000236` (lowers CT win probability)
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000231` (lowers CT win probability)
- `lag_01__centroid_distance_xy`: coefficient `-0.000216` (lowers CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000202` (lowers CT win probability)
- `lag_08__T_shots_fired_sum`: coefficient `0.000176` (raises CT win probability)
- `lag_01__equip_diff`: coefficient `0.000171` (raises CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000167` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `46169`, seconds `0.50`, LSTM delta `-0.0378`

Top all feature movements:
- `lag_01__CT_place_TSIDEUPPER`: contribution `-0.003132`
- `lag_01__T_place_TSPAWN`: contribution `-0.001260`
- `lag_01__CT_place_MIDDLE`: contribution `-0.001207`
- `lag_01__T_closest_enemy_dist`: contribution `-0.000858`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.000854`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `-0.000546`
- `lag_01__molly_inv_diff`: contribution `-0.000514`
- `lag_01__smoke_inv_diff`: contribution `-0.000459`
- `lag_01__T_utility_inv`: contribution `-0.000351`
- `lag_01__T1__utility_total`: contribution `-0.000337`

### tick `47321`, seconds `18.50`, LSTM delta `-0.0181`

Top all feature movements:
- `lag_08__T_shots_fired_sum`: contribution `-0.003439`
- `lag_08__T4__shots_fired`: contribution `-0.001561`
- `lag_01__CT_flashed_players`: contribution `-0.000393`
- `lag_12__CT2__flash_duration`: contribution `-0.000320`
- `lag_00__T2__shots_fired`: contribution `-0.000310`

Top utility-only movements:
- `lag_12__CT2__flash_duration`: contribution `-0.000320`
- `lag_01__CT_flash_duration_sum`: contribution `-0.000307`
- `lag_13__CT4__flash_duration`: contribution `-0.000284`
- `lag_01__CT2__flash_duration`: contribution `-0.000274`
- `lag_01__T2__flash_duration`: contribution `-0.000263`

### tick `46201`, seconds `1.00`, LSTM delta `-0.0055`

Top all feature movements:
- `lag_02__CT_place_TSIDEUPPER`: contribution `-0.000827`
- `lag_00__CT_place_MIDDLE`: contribution `-0.000785`
- `lag_00__CT_macro_MID`: contribution `-0.000440`
- `lag_02__T_place_TSPAWN`: contribution `-0.000399`
- `lag_02__CT_place_MIDDLE`: contribution `-0.000363`

Top utility-only movements:
- `lag_02__utility_inv_diff`: contribution `-0.000194`
- `lag_02__molly_inv_diff`: contribution `-0.000186`
- `lag_02__smoke_inv_diff`: contribution `-0.000166`
- `lag_02__T_molly_inv`: contribution `-0.000110`

### tick `46713`, seconds `9.00`, LSTM delta `-0.0054`

Top all feature movements:
- `lag_14__CT_place_TSIDEUPPER`: contribution `-0.000482`
- `lag_07__T_place_WATER`: contribution `-0.000429`
- `lag_05__T_place_WATER`: contribution `-0.000389`
- `lag_07__T_place_TUNNEL`: contribution `-0.000353`
- `lag_12__T_place_TUNNEL`: contribution `-0.000278`

Top utility-only movements:
- `lag_01__T3__molly`: contribution `+0.000177`
- `lag_01__molly_inv_diff`: contribution `+0.000112`
- `lag_00__T4__molly`: contribution `-0.000097`

### tick `47097`, seconds `15.00`, LSTM delta `+0.0049`

Top all feature movements:
- `lag_01__T_shots_fired_sum`: contribution `+0.001299`
- `lag_01__T2__shots_fired`: contribution `+0.000422`
- `lag_02__T2__shots_fired`: contribution `-0.000263`
- `lag_00__CT_macro_MID`: contribution `+0.000220`
- `lag_00__CT_place_TOPOFMID`: contribution `+0.000215`

Top utility-only movements:
- `lag_00__T3__flash_duration`: contribution `+0.000148`
- `lag_05__CT2__flash_duration`: contribution `+0.000073`
