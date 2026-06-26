# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-virtuspro-bo3-qivzNI2LmnWi0RrHw-7sxj/falcons-vs-virtus-pro-m1-mirage.csv`
- round_num: `11`

## Largest probability jumps

- tick `78444`, seconds `98.50`, LSTM `0.5011`, delta `-0.2010`
- tick `78348`, seconds `97.00`, LSTM `0.7716`, delta `-0.1738`
- tick `76204`, seconds `63.50`, LSTM `0.6686`, delta `+0.1464`
- tick `77196`, seconds `79.00`, LSTM `0.8058`, delta `+0.1327`
- tick `78188`, seconds `94.50`, LSTM `0.9262`, delta `+0.1086`
- tick `78412`, seconds `98.00`, LSTM `0.7021`, delta `-0.0884`
- tick `78092`, seconds `93.00`, LSTM `0.8702`, delta `-0.0627`
- tick `73196`, seconds `16.50`, LSTM `0.6324`, delta `-0.0573`
- tick `76396`, seconds `66.50`, LSTM `0.7301`, delta `-0.0537`
- tick `72908`, seconds `12.00`, LSTM `0.6578`, delta `+0.0471`

## Top 15 local ridge features

- `lag_05__T_place_TRUCK`: coefficient `0.002875`, |coef| `0.002875`
- `lag_00__kill_diff_last_3s`: coefficient `0.002776`, |coef| `0.002776`
- `lag_00__T_place_TRUCK`: coefficient `-0.002520`, |coef| `0.002520`
- `lag_08__T_place_TRUCK`: coefficient `0.002451`, |coef| `0.002451`
- `lag_00__T_kills_last_3s`: coefficient `-0.002111`, |coef| `0.002111`
- `lag_09__CT_place_BACKALLEY`: coefficient `0.001958`, |coef| `0.001958`
- `lag_07__T_place_TRUCK`: coefficient `0.001848`, |coef| `0.001848`
- `lag_00__damage_diff_last_5s`: coefficient `0.001761`, |coef| `0.001761`
- `lag_10__CT5__is_scoped`: coefficient `-0.001754`, |coef| `0.001754`
- `lag_10__CT_place_LADDER`: coefficient `-0.001540`, |coef| `0.001540`
- `lag_15__CT5__duck_amount`: coefficient `0.001532`, |coef| `0.001532`
- `lag_00__CT3__duck_amount`: coefficient `0.001476`, |coef| `0.001476`
- `lag_05__T1__duck_amount`: coefficient `0.001467`, |coef| `0.001467`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001428`, |coef| `0.001428`
- `lag_00__T_damage_last_5s`: coefficient `-0.001428`, |coef| `0.001428`

## Top 10 utility ridge features

- `lag_05__CT_utility_damage_last_5s`: coefficient `-0.001294` (lowers CT win probability)
- `lag_15__CT_utility_damage_last_5s`: coefficient `0.001292` (raises CT win probability)
- `lag_05__utility_damage_diff_last_5s`: coefficient `-0.001068` (lowers CT win probability)
- `lag_15__utility_damage_diff_last_5s`: coefficient `0.001056` (raises CT win probability)
- `lag_08__T_B_site_active_infernos`: coefficient `0.000809` (raises CT win probability)
- `lag_04__CT1__flash`: coefficient `-0.000729` (lowers CT win probability)
- `lag_03__CT1__flash`: coefficient `-0.000692` (lowers CT win probability)
- `lag_10__CT4__flash_duration`: coefficient `0.000684` (raises CT win probability)
- `lag_15__T2__molly`: coefficient `-0.000681` (lowers CT win probability)
- `lag_12__T_B_site_active_infernos`: coefficient `0.000640` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_05__T_place_TRUCK`: coefficient `0.002875` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002776` (raises CT win probability)
- `lag_00__T_place_TRUCK`: coefficient `-0.002520` (lowers CT win probability)
- `lag_08__T_place_TRUCK`: coefficient `0.002451` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002111` (lowers CT win probability)
- `lag_09__CT_place_BACKALLEY`: coefficient `0.001958` (raises CT win probability)
- `lag_07__T_place_TRUCK`: coefficient `0.001848` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001761` (raises CT win probability)
- `lag_10__CT5__is_scoped`: coefficient `-0.001754` (lowers CT win probability)
- `lag_10__CT_place_LADDER`: coefficient `-0.001540` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `78444`, seconds `98.50`, LSTM delta `-0.2010`

Top all feature movements:
- `lag_08__T_place_TRUCK`: contribution `-0.042568`
- `lag_02__T_duck_amount_mean`: contribution `-0.007881`
- `lag_00__T_kills_last_3s`: contribution `-0.006689`
- `lag_00__kill_diff_last_3s`: contribution `-0.006682`
- `lag_10__CT5__is_scoped`: contribution `-0.006272`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `78348`, seconds `97.00`, LSTM delta `-0.1738`

Top all feature movements:
- `lag_05__T_place_TRUCK`: contribution `-0.049932`
- `lag_00__T_kills_last_3s`: contribution `-0.006689`
- `lag_00__kill_diff_last_3s`: contribution `-0.006682`
- `lag_10__CT5__is_scoped`: contribution `-0.006272`
- `lag_00__CT_place_SHOP`: contribution `-0.006039`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `76204`, seconds `63.50`, LSTM delta `+0.1464`

Top all feature movements:
- `lag_10__CT_place_LADDER`: contribution `+0.016010`
- `lag_09__T2__duck_amount`: contribution `+0.005454`
- `lag_06__CT_place_UNDERPASS`: contribution `+0.004692`
- `lag_12__CT_place_SNIPERSNEST`: contribution `+0.004625`
- `lag_00__T_shots_fired_sum`: contribution `+0.003850`

Top utility-only movements:
- `lag_10__CT4__flash_duration`: contribution `+0.002269`
- `lag_12__T_B_site_active_infernos`: contribution `+0.001809`

### tick `77196`, seconds `79.00`, LSTM delta `+0.1327`

Top all feature movements:
- `lag_09__CT_place_BACKALLEY`: contribution `+0.029357`
- `lag_00__kill_diff_last_3s`: contribution `+0.006682`
- `lag_05__CT_utility_damage_last_5s`: contribution `+0.006412`
- `lag_15__CT_utility_damage_last_5s`: contribution `+0.006398`
- `lag_15__CT5__duck_amount`: contribution `+0.005782`

Top utility-only movements:
- `lag_05__CT_utility_damage_last_5s`: contribution `+0.006412`
- `lag_15__CT_utility_damage_last_5s`: contribution `+0.006398`
- `lag_05__utility_damage_diff_last_5s`: contribution `+0.004337`
- `lag_15__utility_damage_diff_last_5s`: contribution `+0.004291`

### tick `78188`, seconds `94.50`, LSTM delta `+0.1086`

Top all feature movements:
- `lag_00__T_place_TRUCK`: contribution `+0.043766`
- `lag_00__kill_diff_last_3s`: contribution `+0.006682`
- `lag_15__CT5__duck_amount`: contribution `+0.005782`
- `lag_13__CT5__duck_amount`: contribution `+0.004975`
- `lag_00__CT_kills_last_3s`: contribution `+0.004059`

Top utility-only movements:
- No utility movement among the top local contributors.
