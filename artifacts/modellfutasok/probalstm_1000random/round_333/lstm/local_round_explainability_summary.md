# Local Round Explainability

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-wildcard-vs-furia-bo3-u8Kr9GGu18RWnHSjYzEreW/wildcard-vs-furia-m2-inferno.csv`
- round_num: `16`

## Largest probability jumps

- tick `138497`, seconds `55.00`, LSTM `0.0873`, delta `-0.2712`
- tick `138241`, seconds `51.00`, LSTM `0.3975`, delta `-0.1337`
- tick `138273`, seconds `51.50`, LSTM `0.2976`, delta `-0.0999`
- tick `139041`, seconds `63.50`, LSTM `0.0141`, delta `-0.0474`
- tick `138433`, seconds `54.00`, LSTM `0.3574`, delta `+0.0386`
- tick `135297`, seconds `5.00`, LSTM `0.4522`, delta `+0.0328`
- tick `138305`, seconds `52.00`, LSTM `0.3284`, delta `+0.0308`
- tick `138529`, seconds `55.50`, LSTM `0.0575`, delta `-0.0299`
- tick `138977`, seconds `62.50`, LSTM `0.0640`, delta `+0.0232`
- tick `136801`, seconds `28.50`, LSTM `0.4811`, delta `-0.0192`

## Top 15 local ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.001993`, |coef| `0.001993`
- `lag_00__CT4__utility_total`: coefficient `0.001832`, |coef| `0.001832`
- `lag_00__CT4__flash`: coefficient `0.001683`, |coef| `0.001683`
- `lag_10__T3__duck_amount`: coefficient `-0.001586`, |coef| `0.001586`
- `lag_09__CT_place_QUAD`: coefficient `0.001459`, |coef| `0.001459`
- `lag_00__kill_diff_last_3s`: coefficient `0.001373`, |coef| `0.001373`
- `lag_00__T_damage_last_5s`: coefficient `-0.001364`, |coef| `0.001364`
- `lag_14__T_flashed_players`: coefficient `0.001272`, |coef| `0.001272`
- `lag_15__T_place_BANANA`: coefficient `-0.001252`, |coef| `0.001252`
- `lag_00__T_burning_players`: coefficient `-0.001230`, |coef| `0.001230`
- `lag_03__CT4__duck_amount`: coefficient `-0.001224`, |coef| `0.001224`
- `lag_14__T_place_BANANA`: coefficient `-0.001205`, |coef| `0.001205`
- `lag_00__CT4__molly`: coefficient `0.001195`, |coef| `0.001195`
- `lag_00__CT4__alive`: coefficient `0.001190`, |coef| `0.001190`
- `lag_05__T3__duck_amount`: coefficient `0.001183`, |coef| `0.001183`

## Top 10 utility ridge features

- `lag_00__CT4__utility_total`: coefficient `0.001832` (raises CT win probability)
- `lag_00__CT4__flash`: coefficient `0.001683` (raises CT win probability)
- `lag_00__CT4__molly`: coefficient `0.001195` (raises CT win probability)
- `lag_00__CT4__smoke`: coefficient `0.001121` (raises CT win probability)
- `lag_03__CT5__flash_duration`: coefficient `0.000972` (raises CT win probability)
- `lag_08__T1__flash_duration`: coefficient `0.000915` (raises CT win probability)
- `lag_08__CT5__smoke`: coefficient `0.000903` (raises CT win probability)
- `lag_13__T_B_site_active_infernos`: coefficient `-0.000896` (lowers CT win probability)
- `lag_09__T1__flash_duration`: coefficient `0.000894` (raises CT win probability)
- `lag_04__CT5__flash_duration`: coefficient `0.000806` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.001993` (lowers CT win probability)
- `lag_10__T3__duck_amount`: coefficient `-0.001586` (lowers CT win probability)
- `lag_09__CT_place_QUAD`: coefficient `0.001459` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001373` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001364` (lowers CT win probability)
- `lag_14__T_flashed_players`: coefficient `0.001272` (raises CT win probability)
- `lag_15__T_place_BANANA`: coefficient `-0.001252` (lowers CT win probability)
- `lag_00__T_burning_players`: coefficient `-0.001230` (lowers CT win probability)
- `lag_03__CT4__duck_amount`: coefficient `-0.001224` (lowers CT win probability)
- `lag_14__T_place_BANANA`: coefficient `-0.001205` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `138497`, seconds `55.00`, LSTM delta `-0.2712`

Top all feature movements:
- `lag_09__CT_place_QUAD`: contribution `-0.011501`
- `lag_00__CT4__utility_total`: contribution `-0.006816`
- `lag_00__T_kills_last_3s`: contribution `-0.006314`
- `lag_10__T3__duck_amount`: contribution `-0.005934`
- `lag_00__CT4__flash`: contribution `-0.005836`

Top utility-only movements:
- `lag_00__CT4__utility_total`: contribution `-0.006816`
- `lag_00__CT4__flash`: contribution `-0.005836`
- `lag_11__CT5__flash_duration`: contribution `-0.003351`

### tick `138241`, seconds `51.00`, LSTM delta `-0.1337`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.006314`
- `lag_01__CT_place_QUAD`: contribution `-0.005491`
- `lag_03__CT5__flash_duration`: contribution `-0.004786`
- `lag_08__T1__flash_duration`: contribution `-0.004382`
- `lag_03__CT4__duck_amount`: contribution `-0.003573`

Top utility-only movements:
- `lag_03__CT5__flash_duration`: contribution `-0.004786`
- `lag_08__T1__flash_duration`: contribution `-0.004382`
- `lag_11__CT5__flash_duration`: contribution `+0.003351`

### tick `138273`, seconds `51.50`, LSTM delta `-0.0999`

Top all feature movements:
- `lag_10__T3__duck_amount`: contribution `-0.004751`
- `lag_09__T1__flash_duration`: contribution `-0.004282`
- `lag_04__CT5__flash_duration`: contribution `-0.003968`
- `lag_02__CT_place_QUAD`: contribution `-0.003895`
- `lag_00__T_shots_fired_sum`: contribution `-0.003763`

Top utility-only movements:
- `lag_09__T1__flash_duration`: contribution `-0.004282`
- `lag_04__CT5__flash_duration`: contribution `-0.003968`
- `lag_06__T_B_site_active_infernos`: contribution `-0.001521`

### tick `139041`, seconds `63.50`, LSTM delta `-0.0474`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.006314`
- `lag_00__kill_diff_last_3s`: contribution `-0.003304`
- `lag_15__CT_place_TOPOFMID`: contribution `-0.003077`
- `lag_05__T3__duck_amount`: contribution `-0.002741`
- `lag_08__T4__is_walking`: contribution `+0.002229`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `138433`, seconds `54.00`, LSTM delta `+0.0386`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `+0.006314`
- `lag_03__CT4__duck_amount`: contribution `+0.004495`
- `lag_00__CT4__duck_amount`: contribution `-0.003568`
- `lag_07__CT_place_QUAD`: contribution `+0.003451`
- `lag_00__kill_diff_last_3s`: contribution `+0.003304`

Top utility-only movements:
- `lag_14__T1__flash_duration`: contribution `-0.001443`
