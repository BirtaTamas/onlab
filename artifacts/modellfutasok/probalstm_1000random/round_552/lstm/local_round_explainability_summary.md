# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-mouz-vs-falcons-bo3-xBECUqZMcQ8GCwi-GUyz8e/mouz-vs-falcons-m1-train.csv`
- round_num: `16`

## Largest probability jumps

- tick `166824`, seconds `120.50`, LSTM `0.0895`, delta `-0.2943`
- tick `164296`, seconds `81.00`, LSTM `0.5225`, delta `-0.2433`
- tick `166440`, seconds `114.50`, LSTM `0.5117`, delta `-0.1531`
- tick `164584`, seconds `85.50`, LSTM `0.7312`, delta `+0.1419`
- tick `165800`, seconds `104.50`, LSTM `0.7049`, delta `+0.1384`
- tick `164872`, seconds `90.00`, LSTM `0.5570`, delta `-0.1300`
- tick `163816`, seconds `73.50`, LSTM `0.6504`, delta `-0.1136`
- tick `166984`, seconds `123.00`, LSTM `0.1417`, delta `+0.0988`
- tick `164232`, seconds `80.00`, LSTM `0.7814`, delta `+0.0841`
- tick `160264`, seconds `18.00`, LSTM `0.6625`, delta `+0.0820`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003495`, |coef| `0.003495`
- `lag_00__CT4__flash`: coefficient `0.002983`, |coef| `0.002983`
- `lag_00__CT_place_IVY`: coefficient `0.002917`, |coef| `0.002917`
- `lag_00__T_kills_last_3s`: coefficient `-0.002904`, |coef| `0.002904`
- `lag_12__CT_place_CONNECTOR`: coefficient `0.002708`, |coef| `0.002708`
- `lag_00__damage_diff_last_5s`: coefficient `0.002566`, |coef| `0.002566`
- `lag_15__CT_place_IVY`: coefficient `0.002145`, |coef| `0.002145`
- `lag_10__CT_place_LONGDOG`: coefficient `-0.002141`, |coef| `0.002141`
- `lag_00__CT4__alive`: coefficient `0.002110`, |coef| `0.002110`
- `lag_12__CT1__alive`: coefficient `0.002077`, |coef| `0.002077`
- `lag_00__CT_place_TSIDEUPPER`: coefficient `-0.002028`, |coef| `0.002028`
- `lag_12__T_kills_last_3s`: coefficient `-0.001968`, |coef| `0.001968`
- `lag_05__T2__flash_duration`: coefficient `0.001937`, |coef| `0.001937`
- `lag_00__T_damage_last_5s`: coefficient `-0.001897`, |coef| `0.001897`
- `lag_12__CT1__armor`: coefficient `0.001881`, |coef| `0.001881`

## Top 10 utility ridge features

- `lag_00__CT4__flash`: coefficient `0.002983` (raises CT win probability)
- `lag_05__T2__flash_duration`: coefficient `0.001937` (raises CT win probability)
- `lag_00__CT4__utility_total`: coefficient `0.001600` (raises CT win probability)
- `lag_11__T2__flash_duration`: coefficient `0.001591` (raises CT win probability)
- `lag_00__T2__flash_duration`: coefficient `0.001504` (raises CT win probability)
- `lag_12__CT1__flash`: coefficient `0.001490` (raises CT win probability)
- `lag_01__CT4__flash`: coefficient `0.001420` (raises CT win probability)
- `lag_00__CT_flash_inv`: coefficient `0.001316` (raises CT win probability)
- `lag_15__T2__flash_duration`: coefficient `0.001227` (raises CT win probability)
- `lag_12__T2__flash_duration`: coefficient `0.001220` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003495` (raises CT win probability)
- `lag_00__CT_place_IVY`: coefficient `0.002917` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002904` (lowers CT win probability)
- `lag_12__CT_place_CONNECTOR`: coefficient `0.002708` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002566` (raises CT win probability)
- `lag_15__CT_place_IVY`: coefficient `0.002145` (raises CT win probability)
- `lag_10__CT_place_LONGDOG`: coefficient `-0.002141` (lowers CT win probability)
- `lag_00__CT4__alive`: coefficient `0.002110` (raises CT win probability)
- `lag_12__CT1__alive`: coefficient `0.002077` (raises CT win probability)
- `lag_00__CT_place_TSIDEUPPER`: coefficient `-0.002028` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `166824`, seconds `120.50`, LSTM delta `-0.2943`

Top all feature movements:
- `lag_00__CT4__flash`: contribution `-0.010342`
- `lag_12__CT_place_CONNECTOR`: contribution `-0.009683`
- `lag_00__T_kills_last_3s`: contribution `-0.009200`
- `lag_00__kill_diff_last_3s`: contribution `-0.008413`
- `lag_12__T_kills_last_3s`: contribution `-0.006234`

Top utility-only movements:
- `lag_00__CT4__flash`: contribution `-0.010342`

### tick `164296`, seconds `81.00`, LSTM delta `-0.2433`

Top all feature movements:
- `lag_15__CT_place_IVY`: contribution `-0.048961`
- `lag_00__CT_place_DUMPSTER`: contribution `-0.044699`
- `lag_02__CT_place_DUMPSTER`: contribution `-0.035519`
- `lag_00__T_kills_last_3s`: contribution `-0.009200`
- `lag_15__CT_place_ENTRANCE`: contribution `-0.008946`

Top utility-only movements:
- `lag_04__CT3__flash_duration`: contribution `-0.003374`

### tick `166440`, seconds `114.50`, LSTM delta `-0.1531`

Top all feature movements:
- `lag_10__CT_place_LONGDOG`: contribution `-0.013967`
- `lag_05__T2__flash_duration`: contribution `-0.013484`
- `lag_00__T_kills_last_3s`: contribution `-0.009200`
- `lag_00__kill_diff_last_3s`: contribution `-0.008413`
- `lag_10__CT_place_BACKOFB`: contribution `-0.006097`

Top utility-only movements:
- `lag_05__T2__flash_duration`: contribution `-0.013484`

### tick `164584`, seconds `85.50`, LSTM delta `+0.1419`

Top all feature movements:
- `lag_05__CT_place_DUMPSTER`: contribution `+0.055246`
- `lag_11__CT_place_DUMPSTER`: contribution `+0.018308`
- `lag_00__T_place_ELECTRICALBOX`: contribution `+0.016410`
- `lag_09__CT_place_DUMPSTER`: contribution `+0.014955`
- `lag_00__kill_diff_last_3s`: contribution `+0.008413`

Top utility-only movements:
- `lag_13__CT3__flash_duration`: contribution `+0.002427`

### tick `165800`, seconds `104.50`, LSTM delta `+0.1384`

Top all feature movements:
- `lag_00__CT_place_TSIDEUPPER`: contribution `+0.015242`
- `lag_00__CT_place_BACKOFB`: contribution `+0.009544`
- `lag_00__kill_diff_last_3s`: contribution `+0.008413`
- `lag_09__T2__duck_amount`: contribution `+0.005375`
- `lag_00__damage_diff_last_5s`: contribution `+0.004515`

Top utility-only movements:
- No utility movement among the top local contributors.
