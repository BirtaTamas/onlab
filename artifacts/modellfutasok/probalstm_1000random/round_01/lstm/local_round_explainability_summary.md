# Local Round Explainability

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-vitality-vs-faze-bo3-cXlexQJNK-6GX9ddkYcv53/vitality-vs-faze-m1-mirage.csv`
- round_num: `15`

## Largest probability jumps

- tick `130103`, seconds `56.50`, LSTM `0.0836`, delta `-0.2945`
- tick `128919`, seconds `38.00`, LSTM `0.2573`, delta `-0.2293`
- tick `129687`, seconds `50.00`, LSTM `0.1805`, delta `+0.1189`
- tick `129527`, seconds `47.50`, LSTM `0.1441`, delta `-0.1114`
- tick `129239`, seconds `43.00`, LSTM `0.1410`, delta `+0.0954`
- tick `129719`, seconds `50.50`, LSTM `0.2671`, delta `+0.0866`
- tick `129367`, seconds `45.00`, LSTM `0.2804`, delta `+0.0791`
- tick `129495`, seconds `47.00`, LSTM `0.2555`, delta `-0.0752`
- tick `129111`, seconds `41.00`, LSTM `0.0717`, delta `-0.0527`
- tick `129751`, seconds `51.00`, LSTM `0.3195`, delta `+0.0525`

## Top 15 local ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.002410`, |coef| `0.002410`
- `lag_14__T2__flash_duration`: coefficient `0.002180`, |coef| `0.002180`
- `lag_04__CT_place_JUNGLE`: coefficient `-0.002139`, |coef| `0.002139`
- `lag_00__kill_diff_last_3s`: coefficient `0.002096`, |coef| `0.002096`
- `lag_15__CT2__flash_duration`: coefficient `0.002086`, |coef| `0.002086`
- `lag_00__damage_diff_last_5s`: coefficient `0.002040`, |coef| `0.002040`
- `lag_12__T_bomb_zone_count`: coefficient `0.002030`, |coef| `0.002030`
- `lag_13__CT_place_SHOP`: coefficient `-0.002002`, |coef| `0.002002`
- `lag_00__T_place_CONNECTOR`: coefficient `-0.001883`, |coef| `0.001883`
- `lag_00__T_place_CTSPAWN`: coefficient `-0.001849`, |coef| `0.001849`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001803`, |coef| `0.001803`
- `lag_00__T_damage_last_5s`: coefficient `-0.001798`, |coef| `0.001798`
- `lag_00__CT_place_JUNGLE`: coefficient `0.001755`, |coef| `0.001755`
- `lag_13__T_bomb_zone_count`: coefficient `-0.001737`, |coef| `0.001737`
- `lag_00__CT_place_CONNECTOR`: coefficient `0.001681`, |coef| `0.001681`

## Top 10 utility ridge features

- `lag_14__T2__flash_duration`: coefficient `0.002180` (raises CT win probability)
- `lag_15__CT2__flash_duration`: coefficient `0.002086` (raises CT win probability)
- `lag_04__T_A_site_active_infernos`: coefficient `-0.001305` (lowers CT win probability)
- `lag_13__T2__flash`: coefficient `0.001285` (raises CT win probability)
- `lag_00__CT3__smoke`: coefficient `0.001264` (raises CT win probability)
- `lag_00__T_A_site_active_infernos`: coefficient `0.001231` (raises CT win probability)
- `lag_13__T2__flash_duration`: coefficient `0.001231` (raises CT win probability)
- `lag_00__CT3__utility_total`: coefficient `0.001191` (raises CT win probability)
- `lag_00__CT4__smoke`: coefficient `0.001160` (raises CT win probability)
- `lag_15__T2__flash_duration`: coefficient `0.001148` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.002410` (lowers CT win probability)
- `lag_04__CT_place_JUNGLE`: coefficient `-0.002139` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002096` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002040` (raises CT win probability)
- `lag_12__T_bomb_zone_count`: coefficient `0.002030` (raises CT win probability)
- `lag_13__CT_place_SHOP`: coefficient `-0.002002` (lowers CT win probability)
- `lag_00__T_place_CONNECTOR`: coefficient `-0.001883` (lowers CT win probability)
- `lag_00__T_place_CTSPAWN`: coefficient `-0.001849` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001803` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001798` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `130103`, seconds `56.50`, LSTM delta `-0.2945`

Top all feature movements:
- `lag_04__CT_place_JUNGLE`: contribution `-0.013725`
- `lag_12__T_bomb_zone_count`: contribution `-0.011820`
- `lag_00__CT_place_JUNGLE`: contribution `-0.011262`
- `lag_13__T_bomb_zone_count`: contribution `-0.010113`
- `lag_13__CT_place_SHOP`: contribution `-0.010041`

Top utility-only movements:
- `lag_04__T_A_site_active_infernos`: contribution `-0.003883`
- `lag_13__T2__flash`: contribution `-0.003784`
- `lag_00__T_A_site_active_infernos`: contribution `-0.003663`
- `lag_03__T_A_site_active_infernos`: contribution `-0.002987`

### tick `128919`, seconds `38.00`, LSTM delta `-0.2293`

Top all feature movements:
- `lag_14__T2__flash_duration`: contribution `-0.012921`
- `lag_15__CT2__flash_duration`: contribution `-0.011832`
- `lag_00__T_kills_last_3s`: contribution `-0.007634`
- `lag_00__T_shots_fired_sum`: contribution `-0.006757`
- `lag_00__CT_place_CONNECTOR`: contribution `-0.006011`

Top utility-only movements:
- `lag_14__T2__flash_duration`: contribution `-0.012921`
- `lag_15__CT2__flash_duration`: contribution `-0.011832`
- `lag_00__CT3__smoke`: contribution `-0.002797`

### tick `129687`, seconds `50.00`, LSTM delta `+0.1189`

Top all feature movements:
- `lag_10__T_place_JUNGLE`: contribution `+0.017192`
- `lag_13__CT_place_SHOP`: contribution `+0.010041`
- `lag_14__CT_place_LADDER`: contribution `+0.009359`
- `lag_13__T_place_JUNGLE`: contribution `+0.007080`
- `lag_15__CT_utility_damage_last_5s`: contribution `+0.006230`

Top utility-only movements:
- `lag_15__CT_utility_damage_last_5s`: contribution `+0.006230`
- `lag_06__CT_utility_damage_last_5s`: contribution `+0.006207`
- `lag_15__utility_damage_diff_last_5s`: contribution `+0.004064`
- `lag_06__utility_damage_diff_last_5s`: contribution `+0.003693`

### tick `129527`, seconds `47.50`, LSTM delta `-0.1114`

Top all feature movements:
- `lag_05__T_place_JUNGLE`: contribution `-0.012234`
- `lag_08__T_place_JUNGLE`: contribution `-0.008188`
- `lag_01__CT_utility_damage_last_5s`: contribution `-0.007606`
- `lag_10__CT_utility_damage_last_5s`: contribution `-0.006140`
- `lag_01__utility_damage_diff_last_5s`: contribution `-0.005613`

Top utility-only movements:
- `lag_01__CT_utility_damage_last_5s`: contribution `-0.007606`
- `lag_10__CT_utility_damage_last_5s`: contribution `-0.006140`
- `lag_01__utility_damage_diff_last_5s`: contribution `-0.005613`
- `lag_11__CT_utility_damage_last_5s`: contribution `-0.004936`
- `lag_10__utility_damage_diff_last_5s`: contribution `-0.003116`

### tick `129239`, seconds `43.00`, LSTM delta `+0.0954`

Top all feature movements:
- `lag_02__T_shots_fired_sum`: contribution `+0.016123`
- `lag_00__CT_place_LADDER`: contribution `+0.013328`
- `lag_00__T_place_CONNECTOR`: contribution `+0.009119`
- `lag_02__T_place_CONNECTOR`: contribution `+0.007094`
- `lag_02__CT_utility_damage_last_5s`: contribution `+0.006568`

Top utility-only movements:
- `lag_02__CT_utility_damage_last_5s`: contribution `+0.006568`
- `lag_01__CT_utility_damage_last_5s`: contribution `+0.005158`
- `lag_02__utility_damage_diff_last_5s`: contribution `+0.004514`
- `lag_01__utility_damage_diff_last_5s`: contribution `+0.003807`
