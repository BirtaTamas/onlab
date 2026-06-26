# Local Round Explainability

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-vitality-vs-faze-bo3-cXlexQJNK-6GX9ddkYcv53/vitality-vs-faze-m1-mirage.csv`
- round_num: `3`

## Largest probability jumps

- tick `27545`, seconds `0.50`, LSTM `0.0246`, delta `-0.0366`
- tick `28025`, seconds `8.00`, LSTM `0.0573`, delta `+0.0144`
- tick `29785`, seconds `35.50`, LSTM `0.0106`, delta `-0.0132`
- tick `28089`, seconds `9.00`, LSTM `0.0763`, delta `+0.0101`
- tick `28057`, seconds `8.50`, LSTM `0.0662`, delta `+0.0089`
- tick `28697`, seconds `18.50`, LSTM `0.0501`, delta `-0.0084`
- tick `30809`, seconds `51.50`, LSTM `0.0099`, delta `-0.0080`
- tick `28601`, seconds `17.00`, LSTM `0.0674`, delta `-0.0077`
- tick `28633`, seconds `17.50`, LSTM `0.0602`, delta `-0.0072`
- tick `27577`, seconds `1.00`, LSTM `0.0185`, delta `-0.0061`

## Top 15 local ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000360`, |coef| `0.000360`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000343`, |coef| `0.000343`
- `lag_04__CT_place_SHOP`: coefficient `0.000306`, |coef| `0.000306`
- `lag_00__CT_place_LADDER`: coefficient `0.000303`, |coef| `0.000303`
- `lag_00__T_velocity_mean`: coefficient `-0.000262`, |coef| `0.000262`
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.000236`, |coef| `0.000236`
- `lag_05__CT_place_SHOP`: coefficient `0.000221`, |coef| `0.000221`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000217`, |coef| `0.000217`
- `lag_01__smoke_inv_diff`: coefficient `0.000208`, |coef| `0.000208`
- `lag_01__utility_inv_diff`: coefficient `0.000207`, |coef| `0.000207`
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000198`, |coef| `0.000198`
- `lag_01__T3__has_bomb`: coefficient `-0.000187`, |coef| `0.000187`
- `lag_01__armor_diff`: coefficient `0.000186`, |coef| `0.000186`
- `lag_01__molly_inv_diff`: coefficient `0.000184`, |coef| `0.000184`
- `lag_01__centroid_distance_xy`: coefficient `-0.000183`, |coef| `0.000183`

## Top 10 utility ridge features

- `lag_00__T_utility_damage_last_5s`: coefficient `-0.000236` (lowers CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000208` (raises CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000207` (raises CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000184` (raises CT win probability)
- `lag_02__T_utility_damage_last_5s`: coefficient `-0.000173` (lowers CT win probability)
- `lag_01__T1__utility_total`: coefficient `-0.000164` (lowers CT win probability)
- `lag_01__T1__flash`: coefficient `-0.000160` (lowers CT win probability)
- `lag_01__T4__molly`: coefficient `-0.000157` (lowers CT win probability)
- `lag_08__T5__flash_duration`: coefficient `0.000150` (raises CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000149` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000360` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000343` (lowers CT win probability)
- `lag_04__CT_place_SHOP`: coefficient `0.000306` (raises CT win probability)
- `lag_00__CT_place_LADDER`: coefficient `0.000303` (raises CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000262` (lowers CT win probability)
- `lag_05__CT_place_SHOP`: coefficient `0.000221` (raises CT win probability)
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000217` (lowers CT win probability)
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000198` (lowers CT win probability)
- `lag_01__T3__has_bomb`: coefficient `-0.000187` (lowers CT win probability)
- `lag_01__armor_diff`: coefficient `0.000186` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `27545`, seconds `0.50`, LSTM delta `-0.0366`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.001724`
- `lag_01__T_place_TSPAWN`: contribution `-0.001518`
- `lag_00__T_velocity_mean`: contribution `-0.000877`
- `lag_01__smoke_inv_diff`: contribution `-0.000662`
- `lag_01__utility_inv_diff`: contribution `-0.000638`

Top utility-only movements:
- `lag_01__smoke_inv_diff`: contribution `-0.000662`
- `lag_01__utility_inv_diff`: contribution `-0.000638`
- `lag_01__molly_inv_diff`: contribution `-0.000513`
- `lag_01__T1__utility_total`: contribution `-0.000369`
- `lag_01__T1__flash`: contribution `-0.000328`

### tick `28025`, seconds `8.00`, LSTM delta `+0.0144`

Top all feature movements:
- `lag_04__CT_place_SHOP`: contribution `+0.003074`
- `lag_00__CT_place_SHOP`: contribution `+0.000831`
- `lag_01__T1__duck_amount`: contribution `+0.000535`
- `lag_12__T_place_SIDEALLEY`: contribution `+0.000515`
- `lag_03__CT_place_SNIPERSNEST`: contribution `+0.000493`

Top utility-only movements:
- `lag_01__T4__molly`: contribution `+0.000343`

### tick `29785`, seconds `35.50`, LSTM delta `-0.0132`

Top all feature movements:
- `lag_06__CT_place_LADDER`: contribution `-0.001577`
- `lag_10__CT_place_UNDERPASS`: contribution `-0.000636`
- `lag_06__CT_place_SNIPERSNEST`: contribution `-0.000462`
- `lag_02__T_utility_damage_last_5s`: contribution `-0.000420`
- `lag_00__CT1__is_walking`: contribution `-0.000416`

Top utility-only movements:
- `lag_02__T_utility_damage_last_5s`: contribution `-0.000420`

### tick `28089`, seconds `9.00`, LSTM delta `+0.0101`

Top all feature movements:
- `lag_00__CT_place_LADDER`: contribution `+0.003148`
- `lag_06__CT_place_SHOP`: contribution `+0.001563`
- `lag_01__CT_place_SHOP`: contribution `+0.000447`
- `lag_00__T1__is_walking`: contribution `+0.000401`
- `lag_00__CT_walking_count`: contribution `+0.000390`

Top utility-only movements:
- `lag_01__T_B_site_active_smokes`: contribution `+0.000150`

### tick `28057`, seconds `8.50`, LSTM delta `+0.0089`

Top all feature movements:
- `lag_05__CT_place_SHOP`: contribution `+0.002217`
- `lag_00__CT_place_SHOP`: contribution `+0.000831`
- `lag_01__CT3__duck_amount`: contribution `+0.000585`
- `lag_13__T_place_SIDEALLEY`: contribution `+0.000517`
- `lag_01__CT_place_SHOP`: contribution `+0.000447`

Top utility-only movements:
- `lag_02__T4__molly`: contribution `+0.000148`
