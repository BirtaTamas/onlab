# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-faze-vs-virtuspro-bo3-YDlVsCnS6YPgcr85bBYoPq/faze-vs-virtus-pro-m2-mirage.csv`
- round_num: `2`

## Largest probability jumps

- tick `6977`, seconds `0.50`, LSTM `0.0126`, delta `-0.0299`
- tick `11201`, seconds `66.50`, LSTM `0.0061`, delta `-0.0075`
- tick `7745`, seconds `12.50`, LSTM `0.0255`, delta `-0.0065`
- tick `8673`, seconds `27.00`, LSTM `0.0312`, delta `-0.0064`
- tick `8865`, seconds `30.00`, LSTM `0.0278`, delta `+0.0060`
- tick `7649`, seconds `11.00`, LSTM `0.0264`, delta `+0.0055`
- tick `7713`, seconds `12.00`, LSTM `0.0320`, delta `+0.0050`
- tick `8193`, seconds `19.50`, LSTM `0.0309`, delta `-0.0049`
- tick `9057`, seconds `33.00`, LSTM `0.0200`, delta `-0.0048`
- tick `8097`, seconds `18.00`, LSTM `0.0321`, delta `+0.0046`

## Top 15 local ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000306`, |coef| `0.000306`
- `lag_00__T_velocity_mean`: coefficient `-0.000269`, |coef| `0.000269`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000257`, |coef| `0.000257`
- `lag_00__CT_velocity_mean`: coefficient `-0.000201`, |coef| `0.000201`
- `lag_01__armor_diff`: coefficient `0.000181`, |coef| `0.000181`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000166`, |coef| `0.000166`
- `lag_15__T_place_HOUSE`: coefficient `0.000165`, |coef| `0.000165`
- `lag_01__centroid_distance_xy`: coefficient `-0.000160`, |coef| `0.000160`
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000157`, |coef| `0.000157`
- `lag_01__CT_armor_sum`: coefficient `0.000149`, |coef| `0.000149`
- `lag_01__smoke_inv_diff`: coefficient `0.000145`, |coef| `0.000145`
- `lag_01__T_flash_alpha_mean`: coefficient `0.000144`, |coef| `0.000144`
- `lag_01__T_walking_count`: coefficient `0.000144`, |coef| `0.000144`
- `lag_01__T3__flash`: coefficient `-0.000141`, |coef| `0.000141`
- `lag_01__utility_inv_diff`: coefficient `0.000137`, |coef| `0.000137`

## Top 10 utility ridge features

- `lag_01__smoke_inv_diff`: coefficient `0.000145` (raises CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `0.000144` (raises CT win probability)
- `lag_01__T3__flash`: coefficient `-0.000141` (lowers CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000137` (raises CT win probability)
- `lag_01__flash_inv_diff`: coefficient `0.000118` (raises CT win probability)
- `lag_01__T2__utility_total`: coefficient `-0.000099` (lowers CT win probability)
- `lag_01__T2__molly`: coefficient `-0.000092` (lowers CT win probability)
- `lag_01__T2__flash`: coefficient `-0.000089` (lowers CT win probability)
- `lag_01__T_smoke_inv`: coefficient `-0.000085` (lowers CT win probability)
- `lag_01__CT_flash_alpha_mean`: coefficient `0.000081` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000306` (lowers CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000269` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000257` (lowers CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000201` (lowers CT win probability)
- `lag_01__armor_diff`: coefficient `0.000181` (raises CT win probability)
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000166` (lowers CT win probability)
- `lag_15__T_place_HOUSE`: coefficient `0.000165` (raises CT win probability)
- `lag_01__centroid_distance_xy`: coefficient `-0.000160` (lowers CT win probability)
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000157` (lowers CT win probability)
- `lag_01__CT_armor_sum`: coefficient `0.000149` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `6977`, seconds `0.50`, LSTM delta `-0.0299`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.001464`
- `lag_01__T_place_TSPAWN`: contribution `-0.001137`
- `lag_00__T_velocity_mean`: contribution `-0.000868`
- `lag_00__CT_velocity_mean`: contribution `-0.000563`
- `lag_01__armor_diff`: contribution `-0.000519`

Top utility-only movements:
- `lag_01__smoke_inv_diff`: contribution `-0.000461`
- `lag_01__utility_inv_diff`: contribution `-0.000391`
- `lag_01__T_flash_alpha_mean`: contribution `-0.000316`
- `lag_01__T3__flash`: contribution `-0.000315`
- `lag_01__flash_inv_diff`: contribution `-0.000265`

### tick `11201`, seconds `66.50`, LSTM delta `-0.0075`

Top all feature movements:
- `lag_11__CT_place_SIDEALLEY`: contribution `-0.001875`
- `lag_04__T_place_LADDER`: contribution `-0.001695`
- `lag_14__CT_place_SIDEALLEY`: contribution `-0.000515`
- `lag_00__CT_place_SIDEALLEY`: contribution `-0.000409`
- `lag_14__T_place_LADDER`: contribution `-0.000404`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `7745`, seconds `12.50`, LSTM delta `-0.0065`

Top all feature movements:
- `lag_06__CT_place_JUNGLE`: contribution `-0.000855`
- `lag_14__CT_place_SHOP`: contribution `-0.000625`
- `lag_03__CT_place_STAIRS`: contribution `-0.000481`
- `lag_12__CT_place_SNIPERSNEST`: contribution `-0.000390`
- `lag_13__T_place_HOUSE`: contribution `+0.000373`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `8673`, seconds `27.00`, LSTM delta `-0.0064`

Top all feature movements:
- `lag_15__T_place_HOUSE`: contribution `-0.000725`
- `lag_02__CT_place_JUNGLE`: contribution `-0.000475`
- `lag_15__CT_place_JUNGLE`: contribution `-0.000471`
- `lag_01__T_place_BACKALLEY`: contribution `-0.000334`
- `lag_14__CT2__duck_amount`: contribution `-0.000215`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `8865`, seconds `30.00`, LSTM delta `+0.0060`

Top all feature movements:
- `lag_08__CT_place_JUNGLE`: contribution `-0.000362`
- `lag_01__CT2__is_walking`: contribution `+0.000269`
- `lag_05__CT_place_SHOP`: contribution `+0.000242`
- `lag_01__CT2__duck_amount`: contribution `+0.000239`
- `lag_05__CT3__duck_amount`: contribution `+0.000235`

Top utility-only movements:
- No utility movement among the top local contributors.
