# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-falcons-bo5-L7CZVGSHd1AqjKPyYU04lA/furia-vs-falcons-m1-inferno.csv`
- round_num: `10`

## Largest probability jumps

- tick `88128`, seconds `88.00`, LSTM `0.9187`, delta `+0.1405`
- tick `88064`, seconds `87.00`, LSTM `0.7403`, delta `+0.1109`
- tick `83136`, seconds `10.00`, LSTM `0.6632`, delta `+0.1027`
- tick `86304`, seconds `59.50`, LSTM `0.7088`, delta `+0.0888`
- tick `89568`, seconds `110.50`, LSTM `0.8790`, delta `-0.0805`
- tick `86400`, seconds `61.00`, LSTM `0.5823`, delta `-0.0779`
- tick `89120`, seconds `103.50`, LSTM `0.9185`, delta `-0.0517`
- tick `86464`, seconds `62.00`, LSTM `0.6332`, delta `+0.0512`
- tick `87744`, seconds `82.00`, LSTM `0.7115`, delta `-0.0471`
- tick `87776`, seconds `82.50`, LSTM `0.6674`, delta `-0.0441`

## Top 15 local ridge features

- `lag_00__T_place_QUAD`: coefficient `0.002018`, |coef| `0.002018`
- `lag_00__CT4__flash_duration`: coefficient `-0.001687`, |coef| `0.001687`
- `lag_00__T2__duck_amount`: coefficient `-0.001584`, |coef| `0.001584`
- `lag_00__CT5__duck_amount`: coefficient `0.001480`, |coef| `0.001480`
- `lag_00__CT_place_BALCONY`: coefficient `-0.001425`, |coef| `0.001425`
- `lag_00__kill_diff_last_3s`: coefficient `0.001401`, |coef| `0.001401`
- `lag_11__T_place_LOWERMID`: coefficient `0.001334`, |coef| `0.001334`
- `lag_15__CT_place_TRAMP`: coefficient `0.001323`, |coef| `0.001323`
- `lag_00__damage_diff_last_5s`: coefficient `0.001237`, |coef| `0.001237`
- `lag_00__CT_kills_last_3s`: coefficient `0.001171`, |coef| `0.001171`
- `lag_00__CT_flash_duration_sum`: coefficient `-0.001087`, |coef| `0.001087`
- `lag_07__T_place_TRAMP`: coefficient `0.001086`, |coef| `0.001086`
- `lag_07__T_place_LOWERMID`: coefficient `-0.001080`, |coef| `0.001080`
- `lag_02__T_place_QUAD`: coefficient `-0.001062`, |coef| `0.001062`
- `lag_00__CT_duck_amount_mean`: coefficient `0.001030`, |coef| `0.001030`

## Top 10 utility ridge features

- `lag_00__CT4__flash_duration`: coefficient `-0.001687` (lowers CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `-0.001087` (lowers CT win probability)
- `lag_05__T_flash_duration_sum`: coefficient `0.001000` (raises CT win probability)
- `lag_01__CT4__flash_duration`: coefficient `-0.000928` (lowers CT win probability)
- `lag_02__T2__flash_duration`: coefficient `-0.000877` (lowers CT win probability)
- `lag_05__T3__flash_duration`: coefficient `0.000849` (raises CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `-0.000842` (lowers CT win probability)
- `lag_05__T4__flash_duration`: coefficient `0.000830` (raises CT win probability)
- `lag_10__CT3__flash_duration`: coefficient `0.000769` (raises CT win probability)
- `lag_10__CT4__flash_duration`: coefficient `0.000763` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_QUAD`: coefficient `0.002018` (raises CT win probability)
- `lag_00__T2__duck_amount`: coefficient `-0.001584` (lowers CT win probability)
- `lag_00__CT5__duck_amount`: coefficient `0.001480` (raises CT win probability)
- `lag_00__CT_place_BALCONY`: coefficient `-0.001425` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001401` (raises CT win probability)
- `lag_11__T_place_LOWERMID`: coefficient `0.001334` (raises CT win probability)
- `lag_15__CT_place_TRAMP`: coefficient `0.001323` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001237` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001171` (raises CT win probability)
- `lag_07__T_place_TRAMP`: coefficient `0.001086` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `88128`, seconds `88.00`, LSTM delta `+0.1405`

Top all feature movements:
- `lag_00__CT4__flash_duration`: contribution `+0.011079`
- `lag_02__T2__flash_duration`: contribution `+0.006338`
- `lag_12__CT4__flash_duration`: contribution `+0.004721`
- `lag_12__CT3__flash_duration`: contribution `+0.003821`
- `lag_07__T_flash_duration_sum`: contribution `+0.003552`

Top utility-only movements:
- `lag_00__CT4__flash_duration`: contribution `+0.011079`
- `lag_02__T2__flash_duration`: contribution `+0.006338`
- `lag_12__CT4__flash_duration`: contribution `+0.004721`
- `lag_12__CT3__flash_duration`: contribution `+0.003821`
- `lag_07__T_flash_duration_sum`: contribution `+0.003552`

### tick `88064`, seconds `87.00`, LSTM delta `+0.1109`

Top all feature movements:
- `lag_05__T_flash_duration_sum`: contribution `+0.006515`
- `lag_00__T2__duck_amount`: contribution `+0.006055`
- `lag_05__T3__flash_duration`: contribution `+0.005496`
- `lag_05__T4__flash_duration`: contribution `+0.005247`
- `lag_10__CT4__flash_duration`: contribution `+0.005012`

Top utility-only movements:
- `lag_05__T_flash_duration_sum`: contribution `+0.006515`
- `lag_05__T3__flash_duration`: contribution `+0.005496`
- `lag_05__T4__flash_duration`: contribution `+0.005247`
- `lag_10__CT4__flash_duration`: contribution `+0.005012`
- `lag_00__T2__flash_duration`: contribution `+0.004697`

### tick `83136`, seconds `10.00`, LSTM delta `+0.1027`

Top all feature movements:
- `lag_07__T_place_LOWERMID`: contribution `+0.010781`
- `lag_11__T_place_LOWERMID`: contribution `+0.008876`
- `lag_07__T_place_TRAMP`: contribution `+0.006354`
- `lag_00__CT_kills_last_3s`: contribution `+0.003380`
- `lag_00__kill_diff_last_3s`: contribution `+0.003371`

Top utility-only movements:
- `lag_00__CT4__molly`: contribution `+0.001753`

### tick `86304`, seconds `59.50`, LSTM delta `+0.0888`

Top all feature movements:
- `lag_00__T_place_QUAD`: contribution `+0.048600`
- `lag_00__CT_place_BALCONY`: contribution `+0.009146`
- `lag_14__CT_place_LIBRARY`: contribution `+0.004506`
- `lag_15__T4__is_walking`: contribution `+0.001936`
- `lag_13__T2__duck_amount`: contribution `+0.001770`

Top utility-only movements:
- `lag_04__T3__flash_duration`: contribution `+0.001459`

### tick `89568`, seconds `110.50`, LSTM delta `-0.0805`

Top all feature movements:
- `lag_15__CT_place_TRAMP`: contribution `-0.017829`
- `lag_14__T_duck_amount_mean`: contribution `-0.003845`
- `lag_15__T_duck_amount_mean`: contribution `-0.003536`
- `lag_00__kill_diff_last_3s`: contribution `-0.003371`
- `lag_00__damage_diff_last_5s`: contribution `-0.002596`

Top utility-only movements:
- `lag_11__CT1__molly`: contribution `-0.001522`
- `lag_09__CT_B_site_active_infernos`: contribution `-0.001287`
