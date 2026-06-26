# Local Round Explainability

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-spirit-vs-g2-bo3-3aFk7fRwd7iUE0VJycUPHK/spirit-vs-g2-m3-ancient.csv`
- round_num: `5`

## Largest probability jumps

- tick `42854`, seconds `30.50`, LSTM `0.1930`, delta `+0.0993`
- tick `40934`, seconds `0.50`, LSTM `0.1223`, delta `-0.0893`
- tick `42822`, seconds `30.00`, LSTM `0.0937`, delta `-0.0731`
- tick `43430`, seconds `39.50`, LSTM `0.0187`, delta `-0.0634`
- tick `41990`, seconds `17.00`, LSTM `0.2610`, delta `+0.0600`
- tick `42502`, seconds `25.00`, LSTM `0.2569`, delta `-0.0440`
- tick `43046`, seconds `33.50`, LSTM `0.1397`, delta `-0.0432`
- tick `42214`, seconds `20.50`, LSTM `0.2996`, delta `+0.0376`
- tick `43270`, seconds `37.00`, LSTM `0.0831`, delta `-0.0375`
- tick `42790`, seconds `29.50`, LSTM `0.1668`, delta `-0.0304`

## Top 15 local ridge features

- `lag_01__CT_place_UNKNOWN`: coefficient `-0.001502`, |coef| `0.001502`
- `lag_00__CT_place_UNKNOWN`: coefficient `0.001147`, |coef| `0.001147`
- `lag_04__T_shots_fired_sum`: coefficient `-0.000979`, |coef| `0.000979`
- `lag_00__T_shots_fired_sum`: coefficient `-0.000959`, |coef| `0.000959`
- `lag_00__T_place_SIDEENTRANCE`: coefficient `-0.000874`, |coef| `0.000874`
- `lag_04__T2__shots_fired`: coefficient `-0.000749`, |coef| `0.000749`
- `lag_03__T_place_TSIDELOWER`: coefficient `0.000714`, |coef| `0.000714`
- `lag_05__CT_place_MAINHALL`: coefficient `-0.000712`, |coef| `0.000712`
- `lag_05__T_place_TSIDELOWER`: coefficient `0.000667`, |coef| `0.000667`
- `lag_01__CT_place_SIDEENTRANCE`: coefficient `-0.000633`, |coef| `0.000633`
- `lag_14__T4__duck_amount`: coefficient `-0.000618`, |coef| `0.000618`
- `lag_00__kill_diff_last_3s`: coefficient `0.000608`, |coef| `0.000608`
- `lag_06__T_place_WATER`: coefficient `-0.000571`, |coef| `0.000571`
- `lag_11__T5__is_walking`: coefficient `0.000566`, |coef| `0.000566`
- `lag_08__T_place_TUNNEL`: coefficient `-0.000559`, |coef| `0.000559`

## Top 10 utility ridge features

- `lag_00__CT_flashes_last_5s`: coefficient `-0.000534` (lowers CT win probability)
- `lag_15__T5__flash_duration`: coefficient `0.000527` (raises CT win probability)
- `lag_10__T_B_site_active_infernos`: coefficient `0.000487` (raises CT win probability)
- `lag_15__CT_he_last_5s`: coefficient `-0.000426` (lowers CT win probability)
- `lag_10__T_active_infernos`: coefficient `0.000396` (raises CT win probability)
- `lag_14__CT_he_last_5s`: coefficient `-0.000386` (lowers CT win probability)
- `lag_08__CT_he_last_5s`: coefficient `-0.000381` (lowers CT win probability)
- `lag_15__T_active_infernos`: coefficient `0.000370` (raises CT win probability)
- `lag_10__T_utility_damage_last_5s`: coefficient `-0.000350` (lowers CT win probability)
- `lag_14__T_B_site_active_infernos`: coefficient `0.000318` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_UNKNOWN`: coefficient `-0.001502` (lowers CT win probability)
- `lag_00__CT_place_UNKNOWN`: coefficient `0.001147` (raises CT win probability)
- `lag_04__T_shots_fired_sum`: coefficient `-0.000979` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.000959` (lowers CT win probability)
- `lag_00__T_place_SIDEENTRANCE`: coefficient `-0.000874` (lowers CT win probability)
- `lag_04__T2__shots_fired`: coefficient `-0.000749` (lowers CT win probability)
- `lag_03__T_place_TSIDELOWER`: coefficient `0.000714` (raises CT win probability)
- `lag_05__CT_place_MAINHALL`: coefficient `-0.000712` (lowers CT win probability)
- `lag_05__T_place_TSIDELOWER`: coefficient `0.000667` (raises CT win probability)
- `lag_01__CT_place_SIDEENTRANCE`: coefficient `-0.000633` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `42854`, seconds `30.50`, LSTM delta `+0.0993`

Top all feature movements:
- `lag_04__T_shots_fired_sum`: contribution `+0.008810`
- `lag_04__T2__shots_fired`: contribution `+0.005292`
- `lag_00__T_shots_fired_sum`: contribution `+0.004313`
- `lag_00__T_place_SIDEENTRANCE`: contribution `+0.004265`
- `lag_01__CT_place_SIDEENTRANCE`: contribution `+0.002549`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `40934`, seconds `0.50`, LSTM delta `-0.0893`

Top all feature movements:
- `lag_01__CT_place_UNKNOWN`: contribution `-0.052712`
- `lag_00__CT_flashes_last_5s`: contribution `-0.005869`
- `lag_01__T_place_TSPAWN`: contribution `-0.001100`
- `lag_00__CT2__armor`: contribution `-0.000865`
- `lag_01__T_closest_enemy_dist`: contribution `-0.000616`

Top utility-only movements:
- `lag_00__CT_flashes_last_5s`: contribution `-0.005869`
- `lag_01__T3__utility_total`: contribution `-0.000444`
- `lag_01__T3__flash`: contribution `-0.000429`
- `lag_01__utility_inv_diff`: contribution `-0.000367`
- `lag_01__flash_inv_diff`: contribution `-0.000321`

### tick `42822`, seconds `30.00`, LSTM delta `-0.0731`

Top all feature movements:
- `lag_04__T_shots_fired_sum`: contribution `-0.003671`
- `lag_00__T_shots_fired_sum`: contribution `-0.003594`
- `lag_03__T_shots_fired_sum`: contribution `-0.002989`
- `lag_14__T4__duck_amount`: contribution `-0.002214`
- `lag_04__T2__shots_fired`: contribution `-0.002205`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `43430`, seconds `39.50`, LSTM delta `-0.0634`

Top all feature movements:
- `lag_05__CT_place_MAINHALL`: contribution `-0.005893`
- `lag_00__T_shots_fired_sum`: contribution `-0.002875`
- `lag_09__T1__duck_amount`: contribution `-0.001879`
- `lag_10__CT4__duck_amount`: contribution `-0.001725`
- `lag_14__T_place_RAMP`: contribution `-0.001670`

Top utility-only movements:
- `lag_10__T_utility_damage_last_5s`: contribution `-0.001400`
- `lag_10__T_B_site_active_infernos`: contribution `-0.001377`

### tick `41990`, seconds `17.00`, LSTM delta `+0.0600`

Top all feature movements:
- `lag_08__T_place_TUNNEL`: contribution `+0.003398`
- `lag_06__T_place_WATER`: contribution `+0.003257`
- `lag_08__T_place_TSIDELOWER`: contribution `+0.003151`
- `lag_15__T5__flash_duration`: contribution `+0.002964`
- `lag_13__T_place_TUNNEL`: contribution `+0.001934`

Top utility-only movements:
- `lag_15__T5__flash_duration`: contribution `+0.002964`
- `lag_03__T5__flash_duration`: contribution `+0.001786`
- `lag_05__CT1__flash_duration`: contribution `+0.001516`
- `lag_06__CT2__flash_duration`: contribution `+0.001487`
- `lag_10__T_B_site_active_infernos`: contribution `+0.001377`
