# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-og-vs-nrg-bo3-GH6ZBFOA9sfdeCxgnhHN9f/og-vs-nrg-m2-nuke.csv`
- round_num: `8`

## Largest probability jumps

- tick `58280`, seconds `22.50`, LSTM `0.3356`, delta `-0.1736`
- tick `58472`, seconds `25.50`, LSTM `0.0837`, delta `-0.1473`
- tick `59112`, seconds `35.50`, LSTM `0.0380`, delta `-0.1022`
- tick `58376`, seconds `24.00`, LSTM `0.2887`, delta `-0.0577`
- tick `58312`, seconds `23.00`, LSTM `0.3822`, delta `+0.0465`
- tick `58440`, seconds `25.00`, LSTM `0.2310`, delta `-0.0458`
- tick `58344`, seconds `23.50`, LSTM `0.3464`, delta `-0.0358`
- tick `58920`, seconds `32.50`, LSTM `0.1432`, delta `+0.0313`
- tick `57800`, seconds `15.00`, LSTM `0.4825`, delta `-0.0249`
- tick `57864`, seconds `16.00`, LSTM `0.4606`, delta `-0.0176`

## Top 15 local ridge features

- `lag_04__T_place_SILO`: coefficient `-0.001169`, |coef| `0.001169`
- `lag_15__CT2__flash_duration`: coefficient `0.001100`, |coef| `0.001100`
- `lag_14__CT_place_CONTROL`: coefficient `-0.001069`, |coef| `0.001069`
- `lag_10__T_place_SILO`: coefficient `-0.000958`, |coef| `0.000958`
- `lag_00__T_kills_last_3s`: coefficient `-0.000885`, |coef| `0.000885`
- `lag_13__T_place_ROOF`: coefficient `-0.000872`, |coef| `0.000872`
- `lag_06__T2__duck_amount`: coefficient `-0.000835`, |coef| `0.000835`
- `lag_00__CT5__flash`: coefficient `0.000823`, |coef| `0.000823`
- `lag_13__T1__duck_amount`: coefficient `0.000822`, |coef| `0.000822`
- `lag_01__CT2__shots_fired`: coefficient `-0.000790`, |coef| `0.000790`
- `lag_07__CT2__shots_fired`: coefficient `-0.000774`, |coef| `0.000774`
- `lag_01__T_shots_fired_sum`: coefficient `-0.000767`, |coef| `0.000767`
- `lag_00__T_shots_fired_sum`: coefficient `-0.000760`, |coef| `0.000760`
- `lag_06__CT2__shots_fired`: coefficient `-0.000760`, |coef| `0.000760`
- `lag_00__T_place_MINI`: coefficient `-0.000741`, |coef| `0.000741`

## Top 10 utility ridge features

- `lag_15__CT2__flash_duration`: coefficient `0.001100` (raises CT win probability)
- `lag_00__CT5__flash`: coefficient `0.000823` (raises CT win probability)
- `lag_13__T_A_site_active_infernos`: coefficient `-0.000694` (lowers CT win probability)
- `lag_00__CT5__utility_total`: coefficient `0.000657` (raises CT win probability)
- `lag_13__T_B_site_active_infernos`: coefficient `-0.000656` (lowers CT win probability)
- `lag_00__T_A_site_active_infernos`: coefficient `0.000650` (raises CT win probability)
- `lag_00__T_B_site_active_infernos`: coefficient `0.000618` (raises CT win probability)
- `lag_13__CT_A_site_active_infernos`: coefficient `0.000598` (raises CT win probability)
- `lag_00__CT2__smoke`: coefficient `0.000597` (raises CT win probability)
- `lag_13__CT_B_site_active_infernos`: coefficient `0.000585` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_04__T_place_SILO`: coefficient `-0.001169` (lowers CT win probability)
- `lag_14__CT_place_CONTROL`: coefficient `-0.001069` (lowers CT win probability)
- `lag_10__T_place_SILO`: coefficient `-0.000958` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.000885` (lowers CT win probability)
- `lag_13__T_place_ROOF`: coefficient `-0.000872` (lowers CT win probability)
- `lag_06__T2__duck_amount`: coefficient `-0.000835` (lowers CT win probability)
- `lag_13__T1__duck_amount`: coefficient `0.000822` (raises CT win probability)
- `lag_01__CT2__shots_fired`: coefficient `-0.000790` (lowers CT win probability)
- `lag_07__CT2__shots_fired`: coefficient `-0.000774` (lowers CT win probability)
- `lag_01__T_shots_fired_sum`: coefficient `-0.000767` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `58280`, seconds `22.50`, LSTM delta `-0.1736`

Top all feature movements:
- `lag_04__T_place_SILO`: contribution `-0.007942`
- `lag_15__CT2__flash_duration`: contribution `-0.006729`
- `lag_13__T_place_ROOF`: contribution `-0.004939`
- `lag_04__CT_place_HEAVEN`: contribution `-0.003445`
- `lag_12__CT_place_RAFTERS`: contribution `-0.003368`

Top utility-only movements:
- `lag_15__CT2__flash_duration`: contribution `-0.006729`
- `lag_13__CT_A_site_active_infernos`: contribution `-0.002111`

### tick `58472`, seconds `25.50`, LSTM delta `-0.1473`

Top all feature movements:
- `lag_03__T_place_MINI`: contribution `-0.006971`
- `lag_10__T_place_SILO`: contribution `-0.006506`
- `lag_01__T_shots_fired_sum`: contribution `-0.004024`
- `lag_13__T1__duck_amount`: contribution `-0.003218`
- `lag_05__T_shots_fired_sum`: contribution `-0.002913`

Top utility-only movements:
- `lag_13__T_A_site_active_infernos`: contribution `-0.002066`
- `lag_00__T_A_site_active_infernos`: contribution `-0.001935`
- `lag_13__T_B_site_active_infernos`: contribution `-0.001855`

### tick `59112`, seconds `35.50`, LSTM delta `-0.1022`

Top all feature movements:
- `lag_14__CT_place_CONTROL`: contribution `-0.011092`
- `lag_11__T_place_MINI`: contribution `-0.008486`
- `lag_06__T_place_MINI`: contribution `-0.003052`
- `lag_00__CT5__flash`: contribution `-0.002921`
- `lag_00__T_kills_last_3s`: contribution `-0.002803`

Top utility-only movements:
- `lag_00__CT5__flash`: contribution `-0.002921`
- `lag_00__CT5__utility_total`: contribution `-0.001862`
- `lag_00__CT5__smoke`: contribution `-0.001115`

### tick `58376`, seconds `24.00`, LSTM delta `-0.0577`

Top all feature movements:
- `lag_00__T_place_MINI`: contribution `-0.010305`
- `lag_07__T_place_SILO`: contribution `-0.003836`
- `lag_06__CT1__duck_amount`: contribution `+0.002813`
- `lag_00__T2__duck_amount`: contribution `-0.002419`
- `lag_07__CT_place_HEAVEN`: contribution `-0.002290`

Top utility-only movements:
- `lag_11__T_A_site_active_infernos`: contribution `-0.001363`
- `lag_10__T_A_site_active_infernos`: contribution `-0.001304`
- `lag_11__T_B_site_active_infernos`: contribution `-0.001221`
- `lag_10__T_B_site_active_infernos`: contribution `-0.001168`

### tick `58312`, seconds `23.00`, LSTM delta `+0.0465`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.005698`
- `lag_06__T2__duck_amount`: contribution `+0.003192`
- `lag_01__T_shots_fired_sum`: contribution `-0.002874`
- `lag_03__CT1__duck_amount`: contribution `+0.002674`
- `lag_13__CT_place_HEAVEN`: contribution `+0.002497`

Top utility-only movements:
- No utility movement among the top local contributors.
