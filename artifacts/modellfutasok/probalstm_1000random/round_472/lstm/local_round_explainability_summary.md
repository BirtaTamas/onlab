# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-vitality-vs-the-mongolz-bo3-vhm_UWcBfYfYcOeLh9JIDA/vitality-vs-the-mongolz-m1-mirage.csv`
- round_num: `6`

## Largest probability jumps

- tick `56056`, seconds `0.50`, LSTM `0.0142`, delta `-0.0354`
- tick `57016`, seconds `15.50`, LSTM `0.0291`, delta `+0.0059`
- tick `58744`, seconds `42.50`, LSTM `0.0078`, delta `-0.0057`
- tick `56088`, seconds `1.00`, LSTM `0.0087`, delta `-0.0055`
- tick `56824`, seconds `12.50`, LSTM `0.0198`, delta `+0.0046`
- tick `58008`, seconds `31.00`, LSTM `0.0231`, delta `-0.0044`
- tick `58072`, seconds `32.00`, LSTM `0.0257`, delta `+0.0041`
- tick `58136`, seconds `33.00`, LSTM `0.0236`, delta `-0.0040`
- tick `57624`, seconds `25.00`, LSTM `0.0213`, delta `-0.0039`
- tick `57816`, seconds `28.00`, LSTM `0.0305`, delta `+0.0039`

## Top 15 local ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000326`, |coef| `0.000326`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000324`, |coef| `0.000324`
- `lag_00__T_velocity_mean`: coefficient `-0.000271`, |coef| `0.000271`
- `lag_00__CT3__duck_amount`: coefficient `0.000249`, |coef| `0.000249`
- `lag_00__CT_velocity_mean`: coefficient `-0.000226`, |coef| `0.000226`
- `lag_01__armor_diff`: coefficient `0.000211`, |coef| `0.000211`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000200`, |coef| `0.000200`
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000193`, |coef| `0.000193`
- `lag_01__centroid_distance_xy`: coefficient `-0.000182`, |coef| `0.000182`
- `lag_01__CT_armor_sum`: coefficient `0.000178`, |coef| `0.000178`
- `lag_00__T2__smoke`: coefficient `0.000168`, |coef| `0.000168`
- `lag_01__utility_inv_diff`: coefficient `0.000167`, |coef| `0.000167`
- `lag_01__T4__has_bomb`: coefficient `-0.000166`, |coef| `0.000166`
- `lag_01__equip_diff`: coefficient `0.000162`, |coef| `0.000162`
- `lag_01__smoke_inv_diff`: coefficient `0.000147`, |coef| `0.000147`

## Top 10 utility ridge features

- `lag_00__T2__smoke`: coefficient `0.000168` (raises CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000167` (raises CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000147` (raises CT win probability)
- `lag_01__flash_inv_diff`: coefficient `0.000138` (raises CT win probability)
- `lag_01__T2__molly`: coefficient `-0.000123` (lowers CT win probability)
- `lag_01__T3__molly`: coefficient `-0.000122` (lowers CT win probability)
- `lag_01__T5__molly`: coefficient `-0.000121` (lowers CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000121` (raises CT win probability)
- `lag_01__T3__utility_total`: coefficient `-0.000107` (lowers CT win probability)
- `lag_01__T4__smoke`: coefficient `-0.000104` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000326` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000324` (lowers CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000271` (lowers CT win probability)
- `lag_00__CT3__duck_amount`: coefficient `0.000249` (raises CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000226` (lowers CT win probability)
- `lag_01__armor_diff`: coefficient `0.000211` (raises CT win probability)
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000200` (lowers CT win probability)
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000193` (lowers CT win probability)
- `lag_01__centroid_distance_xy`: coefficient `-0.000182` (lowers CT win probability)
- `lag_01__CT_armor_sum`: coefficient `0.000178` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `56056`, seconds `0.50`, LSTM delta `-0.0354`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.001559`
- `lag_01__T_place_TSPAWN`: contribution `-0.001433`
- `lag_00__CT3__duck_amount`: contribution `-0.000926`
- `lag_00__T_velocity_mean`: contribution `-0.000821`
- `lag_00__CT_velocity_mean`: contribution `-0.000662`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `-0.000403`
- `lag_01__smoke_inv_diff`: contribution `-0.000373`
- `lag_00__T2__smoke`: contribution `-0.000370`
- `lag_01__flash_inv_diff`: contribution `-0.000253`

### tick `57016`, seconds `15.50`, LSTM delta `+0.0059`

Top all feature movements:
- `lag_12__CT_place_SNIPERSNEST`: contribution `+0.000335`
- `lag_00__T1__is_walking`: contribution `+0.000306`
- `lag_10__CT_place_JUNGLE`: contribution `+0.000287`
- `lag_00__T5__is_walking`: contribution `+0.000264`
- `lag_02__T1__is_walking`: contribution `+0.000208`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `58744`, seconds `42.50`, LSTM delta `-0.0057`

Top all feature movements:
- `lag_02__CT_place_UNDERPASS`: contribution `-0.000297`
- `lag_10__CT3__duck_amount`: contribution `-0.000295`
- `lag_00__T_kills_last_3s`: contribution `-0.000288`
- `lag_00__T5__is_walking`: contribution `+0.000264`
- `lag_12__CT3__duck_amount`: contribution `+0.000261`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `56088`, seconds `1.00`, LSTM delta `-0.0055`

Top all feature movements:
- `lag_02__CT_place_CTSPAWN`: contribution `-0.000641`
- `lag_02__T_place_TSPAWN`: contribution `-0.000623`
- `lag_01__CT3__duck_amount`: contribution `+0.000395`
- `lag_02__armor_diff`: contribution `-0.000250`
- `lag_02__T_closest_enemy_dist`: contribution `-0.000233`

Top utility-only movements:
- `lag_02__utility_inv_diff`: contribution `-0.000168`
- `lag_02__smoke_inv_diff`: contribution `-0.000146`
- `lag_02__flash_inv_diff`: contribution `-0.000106`

### tick `56824`, seconds `12.50`, LSTM delta `+0.0046`

Top all feature movements:
- `lag_11__T_he_last_5s`: contribution `+0.000654`
- `lag_04__CT_place_JUNGLE`: contribution `-0.000351`
- `lag_01__CT_place_CTSPAWN`: contribution `-0.000339`
- `lag_00__CT1__is_walking`: contribution `+0.000335`
- `lag_12__CT_place_SNIPERSNEST`: contribution `-0.000335`

Top utility-only movements:
- `lag_11__T_he_last_5s`: contribution `+0.000654`
