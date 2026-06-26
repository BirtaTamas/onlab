# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-spirit-vs-heroic-bo3-NSK1XoxyhXK-sIe0204dzp/spirit-vs-heroic-m3-mirage.csv`
- round_num: `18`

## Largest probability jumps

- tick `136267`, seconds `0.50`, LSTM `0.0656`, delta `-0.0680`
- tick `138155`, seconds `30.00`, LSTM `0.0632`, delta `-0.0384`
- tick `136811`, seconds `9.00`, LSTM `0.1268`, delta `+0.0215`
- tick `137323`, seconds `17.00`, LSTM `0.1008`, delta `-0.0214`
- tick `138187`, seconds `30.50`, LSTM `0.0442`, delta `-0.0190`
- tick `138507`, seconds `35.50`, LSTM `0.0263`, delta `-0.0151`
- tick `137963`, seconds `27.00`, LSTM `0.1137`, delta `+0.0137`
- tick `138731`, seconds `39.00`, LSTM `0.0083`, delta `-0.0129`
- tick `137835`, seconds `25.00`, LSTM `0.0959`, delta `-0.0124`
- tick `137099`, seconds `13.50`, LSTM `0.1126`, delta `+0.0124`

## Top 15 local ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000619`, |coef| `0.000619`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000603`, |coef| `0.000603`
- `lag_00__T_velocity_mean`: coefficient `-0.000498`, |coef| `0.000498`
- `lag_00__CT_velocity_mean`: coefficient `-0.000494`, |coef| `0.000494`
- `lag_01__T_money_sum`: coefficient `-0.000478`, |coef| `0.000478`
- `lag_06__CT_place_BACKALLEY`: coefficient `-0.000478`, |coef| `0.000478`
- `lag_01__T_start_balance_sum`: coefficient `-0.000476`, |coef| `0.000476`
- `lag_00__T_place_JUNGLE`: coefficient `-0.000451`, |coef| `0.000451`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000371`, |coef| `0.000371`
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000369`, |coef| `0.000369`
- `lag_01__centroid_distance_xy`: coefficient `-0.000348`, |coef| `0.000348`
- `lag_01__utility_inv_diff`: coefficient `0.000346`, |coef| `0.000346`
- `lag_02__CT_place_SHOP`: coefficient `-0.000335`, |coef| `0.000335`
- `lag_01__molly_inv_diff`: coefficient `0.000333`, |coef| `0.000333`
- `lag_01__money_diff`: coefficient `0.000314`, |coef| `0.000314`

## Top 10 utility ridge features

- `lag_01__utility_inv_diff`: coefficient `0.000346` (raises CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000333` (raises CT win probability)
- `lag_01__T2__flash`: coefficient `-0.000313` (lowers CT win probability)
- `lag_00__T4__smoke`: coefficient `0.000305` (raises CT win probability)
- `lag_01__T2__utility_total`: coefficient `-0.000294` (lowers CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000259` (raises CT win probability)
- `lag_01__flash_inv_diff`: coefficient `0.000258` (raises CT win probability)
- `lag_01__T_molly_inv`: coefficient `-0.000257` (lowers CT win probability)
- `lag_01__T_utility_inv`: coefficient `-0.000221` (lowers CT win probability)
- `lag_01__T4__molly`: coefficient `-0.000214` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000619` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000603` (lowers CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000498` (lowers CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000494` (lowers CT win probability)
- `lag_01__T_money_sum`: coefficient `-0.000478` (lowers CT win probability)
- `lag_06__CT_place_BACKALLEY`: coefficient `-0.000478` (lowers CT win probability)
- `lag_01__T_start_balance_sum`: coefficient `-0.000476` (lowers CT win probability)
- `lag_00__T_place_JUNGLE`: coefficient `-0.000451` (lowers CT win probability)
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000371` (lowers CT win probability)
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000369` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `136267`, seconds `0.50`, LSTM delta `-0.0680`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.002958`
- `lag_01__T_place_TSPAWN`: contribution `-0.002670`
- `lag_00__CT_velocity_mean`: contribution `-0.001753`
- `lag_00__T_velocity_mean`: contribution `-0.001635`
- `lag_01__T_money_sum`: contribution `-0.001621`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `-0.000989`
- `lag_01__molly_inv_diff`: contribution `-0.000928`
- `lag_01__T2__utility_total`: contribution `-0.000709`
- `lag_01__T2__flash`: contribution `-0.000696`
- `lag_00__T4__smoke`: contribution `-0.000664`

### tick `138155`, seconds `30.00`, LSTM delta `-0.0384`

Top all feature movements:
- `lag_06__CT_place_BACKALLEY`: contribution `-0.007169`
- `lag_00__T_place_JUNGLE`: contribution `-0.005840`
- `lag_01__T_place_CONNECTOR`: contribution `-0.001486`
- `lag_00__T_place_CONNECTOR`: contribution `-0.001332`
- `lag_13__T_place_CONNECTOR`: contribution `-0.001211`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `136811`, seconds `9.00`, LSTM delta `+0.0215`

Top all feature movements:
- `lag_07__CT_place_SHOP`: contribution `+0.002425`
- `lag_00__CT_place_LADDER`: contribution `+0.002296`
- `lag_03__CT_place_SHOP`: contribution `+0.001988`
- `lag_02__CT_place_SHOP`: contribution `+0.001682`
- `lag_08__CT_place_SHOP`: contribution `+0.001430`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `137323`, seconds `17.00`, LSTM delta `-0.0214`

Top all feature movements:
- `lag_00__CT_place_TRUCK`: contribution `-0.001728`
- `lag_00__CT_place_APARTMENTS`: contribution `-0.000806`
- `lag_02__CT_place_TRUCK`: contribution `-0.000699`
- `lag_06__T1__is_scoped`: contribution `+0.000648`
- `lag_11__CT_place_APARTMENTS`: contribution `-0.000629`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `138187`, seconds `30.50`, LSTM delta `-0.0190`

Top all feature movements:
- `lag_07__CT_place_BACKALLEY`: contribution `-0.003023`
- `lag_01__T_place_JUNGLE`: contribution `-0.001843`
- `lag_01__T_place_CONNECTOR`: contribution `-0.001486`
- `lag_00__T_place_CONNECTOR`: contribution `-0.001332`
- `lag_07__CT_place_APARTMENTS`: contribution `-0.001092`

Top utility-only movements:
- No utility movement among the top local contributors.
