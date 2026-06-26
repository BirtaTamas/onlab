# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-pain-bo3-6mWraId8pA69o5etX6dmBT/falcons-vs-pain-m1-inferno.csv`
- round_num: `14`

## Largest probability jumps

- tick `101048`, seconds `40.00`, LSTM `0.1627`, delta `-0.2082`
- tick `99768`, seconds `20.00`, LSTM `0.2830`, delta `+0.0532`
- tick `103352`, seconds `76.00`, LSTM `0.0441`, delta `-0.0490`
- tick `103320`, seconds `75.50`, LSTM `0.0931`, delta `-0.0488`
- tick `98808`, seconds `5.00`, LSTM `0.3298`, delta `+0.0480`
- tick `99704`, seconds `19.00`, LSTM `0.2353`, delta `-0.0438`
- tick `103192`, seconds `73.50`, LSTM `0.1795`, delta `-0.0375`
- tick `102744`, seconds `66.50`, LSTM `0.2204`, delta `+0.0365`
- tick `99064`, seconds `9.00`, LSTM `0.3116`, delta `-0.0358`
- tick `103128`, seconds `72.50`, LSTM `0.2038`, delta `-0.0350`

## Top 15 local ridge features

- `lag_10__T_place_SECONDMID`: coefficient `0.002753`, |coef| `0.002753`
- `lag_00__T_kills_last_3s`: coefficient `-0.002680`, |coef| `0.002680`
- `lag_00__CT5__alive`: coefficient `0.002055`, |coef| `0.002055`
- `lag_00__kill_diff_last_3s`: coefficient `0.002036`, |coef| `0.002036`
- `lag_00__CT5__hp`: coefficient `0.002028`, |coef| `0.002028`
- `lag_04__CT4__is_walking`: coefficient `-0.002018`, |coef| `0.002018`
- `lag_00__T_damage_last_5s`: coefficient `-0.001934`, |coef| `0.001934`
- `lag_00__CT5__armor`: coefficient `0.001924`, |coef| `0.001924`
- `lag_00__CT5__smoke`: coefficient `0.001854`, |coef| `0.001854`
- `lag_15__T3__molly`: coefficient `0.001813`, |coef| `0.001813`
- `lag_00__damage_diff_last_5s`: coefficient `0.001805`, |coef| `0.001805`
- `lag_09__T4__duck_amount`: coefficient `-0.001795`, |coef| `0.001795`
- `lag_00__T1__duck_amount`: coefficient `-0.001763`, |coef| `0.001763`
- `lag_00__CT5__has_helmet`: coefficient `0.001698`, |coef| `0.001698`
- `lag_00__T_place_TOPOFMID`: coefficient `-0.001664`, |coef| `0.001664`

## Top 10 utility ridge features

- `lag_00__CT5__smoke`: coefficient `0.001854` (raises CT win probability)
- `lag_15__T3__molly`: coefficient `0.001813` (raises CT win probability)
- `lag_13__T_A_site_active_infernos`: coefficient `-0.001574` (lowers CT win probability)
- `lag_06__T3__flash`: coefficient `0.001229` (raises CT win probability)
- `lag_13__T_active_infernos`: coefficient `-0.000986` (lowers CT win probability)
- `lag_01__T_A_site_active_infernos`: coefficient `0.000867` (raises CT win probability)
- `lag_00__CT5__utility_total`: coefficient `0.000798` (raises CT win probability)
- `lag_01__T_active_infernos`: coefficient `0.000748` (raises CT win probability)
- `lag_06__T3__utility_total`: coefficient `0.000723` (raises CT win probability)
- `lag_01__CT5__smoke`: coefficient `0.000704` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_10__T_place_SECONDMID`: coefficient `0.002753` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002680` (lowers CT win probability)
- `lag_00__CT5__alive`: coefficient `0.002055` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002036` (raises CT win probability)
- `lag_00__CT5__hp`: coefficient `0.002028` (raises CT win probability)
- `lag_04__CT4__is_walking`: coefficient `-0.002018` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001934` (lowers CT win probability)
- `lag_00__CT5__armor`: coefficient `0.001924` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001805` (raises CT win probability)
- `lag_09__T4__duck_amount`: coefficient `-0.001795` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `101048`, seconds `40.00`, LSTM delta `-0.2082`

Top all feature movements:
- `lag_10__T_place_SECONDMID`: contribution `-0.009014`
- `lag_00__T_kills_last_3s`: contribution `-0.008491`
- `lag_00__T1__duck_amount`: contribution `-0.006903`
- `lag_09__T4__duck_amount`: contribution `-0.006636`
- `lag_04__T1__duck_amount`: contribution `-0.006258`

Top utility-only movements:
- `lag_13__T_A_site_active_infernos`: contribution `-0.004685`
- `lag_00__CT5__smoke`: contribution `-0.004066`

### tick `99768`, seconds `20.00`, LSTM delta `+0.0532`

Top all feature movements:
- `lag_02__T_place_BALCONY`: contribution `+0.009544`
- `lag_00__T_place_BALCONY`: contribution `+0.007285`
- `lag_12__CT2__is_walking`: contribution `-0.003357`
- `lag_05__T3__duck_amount`: contribution `+0.003243`
- `lag_00__CT3__is_walking`: contribution `+0.002798`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `103352`, seconds `76.00`, LSTM delta `-0.0490`

Top all feature movements:
- `lag_05__T_place_QUAD`: contribution `-0.011038`
- `lag_00__T_kills_last_3s`: contribution `-0.008491`
- `lag_01__T_place_QUAD`: contribution `+0.007478`
- `lag_09__T4__duck_amount`: contribution `-0.006636`
- `lag_03__T_place_ARCH`: contribution `-0.005746`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `103320`, seconds `75.50`, LSTM delta `-0.0488`

Top all feature movements:
- `lag_04__T_place_QUAD`: contribution `-0.022874`
- `lag_00__T_place_QUAD`: contribution `+0.014326`
- `lag_02__T_place_ARCH`: contribution `-0.006711`
- `lag_05__T_place_ARCH`: contribution `-0.003874`
- `lag_00__T_place_TOPOFMID`: contribution `-0.003388`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `98808`, seconds `5.00`, LSTM delta `+0.0480`

Top all feature movements:
- `lag_09__CT_smokes_last_5s`: contribution `+0.006318`
- `lag_05__CT_place_LIBRARY`: contribution `+0.003317`
- `lag_00__CT_place_LIBRARY`: contribution `+0.003302`
- `lag_00__T_place_LOWERMID`: contribution `+0.002823`
- `lag_00__CT_place_RUINS`: contribution `+0.002499`

Top utility-only movements:
- `lag_09__CT_smokes_last_5s`: contribution `+0.006318`
- `lag_03__CT_smokes_last_5s`: contribution `+0.002403`
- `lag_07__T_he_last_5s`: contribution `+0.001540`
- `lag_09__CT_flashes_last_5s`: contribution `+0.001134`
