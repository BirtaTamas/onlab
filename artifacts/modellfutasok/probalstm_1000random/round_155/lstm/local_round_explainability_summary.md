# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-b8-vs-flyquest-bo3-ROTxQXIIApwC88KHLMMjQT/b8-vs-flyquest-m3-inferno.csv`
- round_num: `17`

## Largest probability jumps

- tick `139795`, seconds `55.50`, LSTM `0.1152`, delta `-0.1703`
- tick `137843`, seconds `25.00`, LSTM `0.1524`, delta `-0.0792`
- tick `140243`, seconds `62.50`, LSTM `0.0226`, delta `-0.0760`
- tick `137779`, seconds `24.00`, LSTM `0.2611`, delta `-0.0599`
- tick `137651`, seconds `22.00`, LSTM `0.4234`, delta `-0.0587`
- tick `137619`, seconds `21.50`, LSTM `0.4821`, delta `-0.0584`
- tick `139411`, seconds `49.50`, LSTM `0.2153`, delta `-0.0541`
- tick `137971`, seconds `27.00`, LSTM `0.0800`, delta `-0.0532`
- tick `140051`, seconds `59.50`, LSTM `0.0915`, delta `-0.0407`
- tick `139539`, seconds `51.50`, LSTM `0.2491`, delta `+0.0360`

## Top 15 local ridge features

- `lag_12__CT_place_QUAD`: coefficient `0.001783`, |coef| `0.001783`
- `lag_00__CT_place_QUAD`: coefficient `0.001586`, |coef| `0.001586`
- `lag_07__CT_place_TOPOFMID`: coefficient `-0.001363`, |coef| `0.001363`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001236`, |coef| `0.001236`
- `lag_00__CT_place_ARCH`: coefficient `0.001143`, |coef| `0.001143`
- `lag_08__CT_place_QUAD`: coefficient `-0.001141`, |coef| `0.001141`
- `lag_00__T_kills_last_3s`: coefficient `-0.001088`, |coef| `0.001088`
- `lag_11__T2__is_walking`: coefficient `-0.001010`, |coef| `0.001010`
- `lag_00__CT2__alive`: coefficient `0.000997`, |coef| `0.000997`
- `lag_00__T1__shots_fired`: coefficient `-0.000969`, |coef| `0.000969`
- `lag_00__T_place_UPSTAIRS`: coefficient `0.000967`, |coef| `0.000967`
- `lag_00__CT2__utility_total`: coefficient `0.000917`, |coef| `0.000917`
- `lag_08__CT_place_TOPOFMID`: coefficient `-0.000911`, |coef| `0.000911`
- `lag_00__CT2__smoke`: coefficient `0.000893`, |coef| `0.000893`
- `lag_12__T_place_DECK`: coefficient `-0.000873`, |coef| `0.000873`

## Top 10 utility ridge features

- `lag_00__CT2__utility_total`: coefficient `0.000917` (raises CT win probability)
- `lag_00__CT2__smoke`: coefficient `0.000893` (raises CT win probability)
- `lag_05__T_utility_damage_last_5s`: coefficient `-0.000859` (lowers CT win probability)
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.000844` (lowers CT win probability)
- `lag_04__T_utility_damage_last_5s`: coefficient `-0.000841` (lowers CT win probability)
- `lag_00__CT2__flash`: coefficient `0.000745` (raises CT win probability)
- `lag_15__T1__smoke`: coefficient `0.000702` (raises CT win probability)
- `lag_03__T_utility_damage_last_5s`: coefficient `-0.000658` (lowers CT win probability)
- `lag_01__T_utility_damage_last_5s`: coefficient `-0.000629` (lowers CT win probability)
- `lag_12__T_utility_damage_last_5s`: coefficient `-0.000622` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_12__CT_place_QUAD`: coefficient `0.001783` (raises CT win probability)
- `lag_00__CT_place_QUAD`: coefficient `0.001586` (raises CT win probability)
- `lag_07__CT_place_TOPOFMID`: coefficient `-0.001363` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001236` (lowers CT win probability)
- `lag_00__CT_place_ARCH`: coefficient `0.001143` (raises CT win probability)
- `lag_08__CT_place_QUAD`: coefficient `-0.001141` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001088` (lowers CT win probability)
- `lag_11__T2__is_walking`: coefficient `-0.001010` (lowers CT win probability)
- `lag_00__CT2__alive`: coefficient `0.000997` (raises CT win probability)
- `lag_00__T1__shots_fired`: coefficient `-0.000969` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `139795`, seconds `55.50`, LSTM delta `-0.1703`

Top all feature movements:
- `lag_12__CT_place_QUAD`: contribution `-0.014049`
- `lag_08__CT_place_QUAD`: contribution `-0.008993`
- `lag_07__CT_place_TOPOFMID`: contribution `-0.004945`
- `lag_00__CT_place_ARCH`: contribution `-0.004665`
- `lag_00__T_shots_fired_sum`: contribution `-0.004632`

Top utility-only movements:
- `lag_00__CT2__smoke`: contribution `-0.001938`

### tick `137843`, seconds `25.00`, LSTM delta `-0.0792`

Top all feature movements:
- `lag_12__T_place_DECK`: contribution `-0.021185`
- `lag_03__T_place_DECK`: contribution `-0.011727`
- `lag_12__T_place_KITCHEN`: contribution `-0.008153`
- `lag_00__T_utility_damage_last_5s`: contribution `-0.005063`
- `lag_01__T_utility_damage_last_5s`: contribution `-0.002875`

Top utility-only movements:
- `lag_00__T_utility_damage_last_5s`: contribution `-0.005063`
- `lag_01__T_utility_damage_last_5s`: contribution `-0.002875`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.001623`
- `lag_00__T1__molly`: contribution `-0.000822`

### tick `140243`, seconds `62.50`, LSTM delta `-0.0760`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.003446`
- `lag_12__T_shots_fired_sum`: contribution `-0.003433`
- `lag_07__T2__duck_amount`: contribution `+0.003018`
- `lag_06__CT_place_TOPOFMID`: contribution `-0.002533`
- `lag_06__CT_place_QUAD`: contribution `-0.002524`

Top utility-only movements:
- `lag_13__T_A_site_active_infernos`: contribution `+0.001556`

### tick `137779`, seconds `24.00`, LSTM delta `-0.0599`

Top all feature movements:
- `lag_10__T_place_DECK`: contribution `-0.020675`
- `lag_14__T_place_KITCHEN`: contribution `-0.016543`
- `lag_01__T_place_DECK`: contribution `-0.007968`
- `lag_10__T_place_KITCHEN`: contribution `+0.004621`
- `lag_14__T_place_UPSTAIRS`: contribution `-0.003492`

Top utility-only movements:
- `lag_08__CT_A_site_active_infernos`: contribution `-0.000870`

### tick `137651`, seconds `22.00`, LSTM delta `-0.0587`

Top all feature movements:
- `lag_06__T_place_KITCHEN`: contribution `-0.019003`
- `lag_06__T_place_DECK`: contribution `-0.013337`
- `lag_10__T_place_UPSTAIRS`: contribution `-0.010326`
- `lag_10__T_place_KITCHEN`: contribution `-0.004621`
- `lag_00__T_shots_fired_sum`: contribution `+0.002779`

Top utility-only movements:
- `lag_15__CT_A_site_active_infernos`: contribution `-0.000896`
