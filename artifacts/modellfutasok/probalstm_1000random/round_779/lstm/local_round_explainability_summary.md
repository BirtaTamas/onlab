# Local Round Explainability

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-pain-vs-rare-atom-bo3-Rmb0_mvtIpTmOfUIJVjwOw/pain-vs-rare-atom-m2-dust2.csv`
- round_num: `12`

## Largest probability jumps

- tick `91017`, seconds `26.50`, LSTM `0.0843`, delta `-0.1368`
- tick `89641`, seconds `5.00`, LSTM `0.1682`, delta `-0.0415`
- tick `90761`, seconds `22.50`, LSTM `0.1816`, delta `+0.0363`
- tick `89353`, seconds `0.50`, LSTM `0.2303`, delta `-0.0361`
- tick `91209`, seconds `29.50`, LSTM `0.0244`, delta `-0.0326`
- tick `89897`, seconds `9.00`, LSTM `0.1758`, delta `+0.0324`
- tick `90985`, seconds `26.00`, LSTM `0.2211`, delta `-0.0272`
- tick `90921`, seconds `25.00`, LSTM `0.2335`, delta `+0.0244`
- tick `90793`, seconds `23.00`, LSTM `0.2060`, delta `+0.0244`
- tick `89993`, seconds `10.50`, LSTM `0.1822`, delta `+0.0210`

## Top 15 local ridge features

- `lag_07__CT_place_ARAMP`: coefficient `0.001349`, |coef| `0.001349`
- `lag_01__CT_shots_fired_sum`: coefficient `0.001114`, |coef| `0.001114`
- `lag_15__T_place_MIDDOORS`: coefficient `-0.001041`, |coef| `0.001041`
- `lag_03__T_place_EXTENDEDA`: coefficient `-0.000989`, |coef| `0.000989`
- `lag_00__CT_shots_fired_sum`: coefficient `0.000924`, |coef| `0.000924`
- `lag_01__CT1__shots_fired`: coefficient `0.000895`, |coef| `0.000895`
- `lag_09__CT2__duck_amount`: coefficient `0.000823`, |coef| `0.000823`
- `lag_07__T4__is_scoped`: coefficient `-0.000798`, |coef| `0.000798`
- `lag_05__CT1__duck_amount`: coefficient `-0.000794`, |coef| `0.000794`
- `lag_03__CT_shots_fired_sum`: coefficient `-0.000723`, |coef| `0.000723`
- `lag_00__CT_place_LONGA`: coefficient `0.000715`, |coef| `0.000715`
- `lag_09__T_place_MIDDOORS`: coefficient `-0.000713`, |coef| `0.000713`
- `lag_00__CT2__alive`: coefficient `0.000682`, |coef| `0.000682`
- `lag_00__CT1__shots_fired`: coefficient `0.000677`, |coef| `0.000677`
- `lag_00__CT2__hp`: coefficient `0.000674`, |coef| `0.000674`

## Top 10 utility ridge features

- `lag_03__T5__smoke`: coefficient `0.000509` (raises CT win probability)
- `lag_13__T1__flash`: coefficient `0.000323` (raises CT win probability)
- `lag_01__T3__utility_total`: coefficient `-0.000267` (lowers CT win probability)
- `lag_01__T3__flash`: coefficient `-0.000250` (lowers CT win probability)
- `lag_00__T_active_smokes`: coefficient `-0.000240` (lowers CT win probability)
- `lag_02__T5__smoke`: coefficient `0.000224` (raises CT win probability)
- `lag_04__T3__utility_total`: coefficient `-0.000208` (lowers CT win probability)
- `lag_10__T3__utility_total`: coefficient `-0.000205` (lowers CT win probability)
- `lag_04__T5__smoke`: coefficient `0.000204` (raises CT win probability)
- `lag_04__T3__flash`: coefficient `-0.000195` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_07__CT_place_ARAMP`: coefficient `0.001349` (raises CT win probability)
- `lag_01__CT_shots_fired_sum`: coefficient `0.001114` (raises CT win probability)
- `lag_15__T_place_MIDDOORS`: coefficient `-0.001041` (lowers CT win probability)
- `lag_03__T_place_EXTENDEDA`: coefficient `-0.000989` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.000924` (raises CT win probability)
- `lag_01__CT1__shots_fired`: coefficient `0.000895` (raises CT win probability)
- `lag_09__CT2__duck_amount`: coefficient `0.000823` (raises CT win probability)
- `lag_07__T4__is_scoped`: coefficient `-0.000798` (lowers CT win probability)
- `lag_05__CT1__duck_amount`: coefficient `-0.000794` (lowers CT win probability)
- `lag_03__CT_shots_fired_sum`: coefficient `-0.000723` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `91017`, seconds `26.50`, LSTM delta `-0.1368`

Top all feature movements:
- `lag_01__CT_shots_fired_sum`: contribution `-0.008512`
- `lag_07__CT_place_ARAMP`: contribution `-0.008404`
- `lag_01__CT1__shots_fired`: contribution `-0.005203`
- `lag_03__T_place_EXTENDEDA`: contribution `-0.004904`
- `lag_15__T_place_MIDDOORS`: contribution `-0.004424`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `89641`, seconds `5.00`, LSTM delta `-0.0415`

Top all feature movements:
- `lag_00__T1__is_scoped`: contribution `-0.002399`
- `lag_04__CT_place_MIDDOORS`: contribution `-0.002080`
- `lag_02__T_place_OUTSIDETUNNEL`: contribution `-0.002012`
- `lag_00__CT_place_LONGA`: contribution `+0.001910`
- `lag_10__T_place_TSPAWN`: contribution `-0.001693`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `90761`, seconds `22.50`, LSTM delta `+0.0363`

Top all feature movements:
- `lag_14__CT_place_ARAMP`: contribution `+0.002735`
- `lag_03__T_place_LOWERTUNNEL`: contribution `+0.001702`
- `lag_08__T_place_LOWERTUNNEL`: contribution `+0.001477`
- `lag_01__T_place_MIDDOORS`: contribution `+0.001308`
- `lag_09__T_place_SHORTSTAIRS`: contribution `+0.001268`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `89353`, seconds `0.50`, LSTM delta `-0.0361`

Top all feature movements:
- `lag_00__CT_velocity_mean`: contribution `-0.001391`
- `lag_00__T_velocity_mean`: contribution `-0.001159`
- `lag_01__T_place_TSPAWN`: contribution `-0.001038`
- `lag_00__T1__has_bomb`: contribution `-0.000914`
- `lag_00__CT2__armor`: contribution `+0.000872`

Top utility-only movements:
- `lag_01__T3__utility_total`: contribution `-0.000638`
- `lag_01__T3__flash`: contribution `-0.000561`
- `lag_01__T4__utility_total`: contribution `-0.000414`
- `lag_01__utility_inv_diff`: contribution `-0.000377`
- `lag_01__T_utility_inv`: contribution `-0.000366`

### tick `91209`, seconds `29.50`, LSTM delta `-0.0326`

Top all feature movements:
- `lag_15__T_place_MIDDOORS`: contribution `-0.004424`
- `lag_07__CT_shots_fired_sum`: contribution `-0.001547`
- `lag_13__CT_place_ARAMP`: contribution `-0.001496`
- `lag_09__T_place_EXTENDEDA`: contribution `-0.001430`
- `lag_00__T_shots_fired_sum`: contribution `-0.001376`

Top utility-only movements:
- No utility movement among the top local contributors.
