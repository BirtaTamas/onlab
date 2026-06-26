# Local Round Explainability

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-lynn-vision-vs-furia-bo3-RhNzrLTGYeGsl1rd1jweWL/lynn-vision-vs-furia-m2-anubis.csv`
- round_num: `10`

## Largest probability jumps

- tick `71096`, seconds `42.50`, LSTM `0.7229`, delta `+0.2966`
- tick `71928`, seconds `55.50`, LSTM `0.9125`, delta `+0.1418`
- tick `70744`, seconds `37.00`, LSTM `0.6429`, delta `+0.1175`
- tick `71064`, seconds `42.00`, LSTM `0.4263`, delta `-0.0896`
- tick `70840`, seconds `38.50`, LSTM `0.5444`, delta `-0.0862`
- tick `71416`, seconds `47.50`, LSTM `0.6478`, delta `-0.0810`
- tick `70712`, seconds `36.50`, LSTM `0.5254`, delta `+0.0730`
- tick `71736`, seconds `52.50`, LSTM `0.7687`, delta `+0.0544`
- tick `72440`, seconds `63.50`, LSTM `0.9451`, delta `+0.0484`
- tick `71512`, seconds `49.00`, LSTM `0.6944`, delta `+0.0409`

## Top 15 local ridge features

- `lag_01__T_place_WALKWAY`: coefficient `0.002646`, |coef| `0.002646`
- `lag_00__T_place_WALKWAY`: coefficient `-0.002059`, |coef| `0.002059`
- `lag_00__kill_diff_last_3s`: coefficient `0.001628`, |coef| `0.001628`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001593`, |coef| `0.001593`
- `lag_00__CT_kills_last_3s`: coefficient `0.001546`, |coef| `0.001546`
- `lag_00__damage_diff_last_5s`: coefficient `0.001391`, |coef| `0.001391`
- `lag_01__T_place_MAIN`: coefficient `0.001361`, |coef| `0.001361`
- `lag_12__CT_place_MAIN`: coefficient `-0.001351`, |coef| `0.001351`
- `lag_10__CT_shots_fired_sum`: coefficient `-0.001259`, |coef| `0.001259`
- `lag_01__CT_shots_fired_sum`: coefficient `0.001225`, |coef| `0.001225`
- `lag_00__CT3__is_walking`: coefficient `-0.001170`, |coef| `0.001170`
- `lag_14__CT_place_MAIN`: coefficient `0.001150`, |coef| `0.001150`
- `lag_12__T_place_BRIDGE`: coefficient `-0.001133`, |coef| `0.001133`
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `-0.001095`, |coef| `0.001095`
- `lag_04__T3__flash_duration`: coefficient `0.001064`, |coef| `0.001064`

## Top 10 utility ridge features

- `lag_04__T3__flash_duration`: coefficient `0.001064` (raises CT win probability)
- `lag_04__T1__flash_duration`: coefficient `0.000950` (raises CT win probability)
- `lag_04__T_flash_duration_sum`: coefficient `0.000942` (raises CT win probability)
- `lag_00__T5__utility_total`: coefficient `-0.000721` (lowers CT win probability)
- `lag_07__T3__flash_duration`: coefficient `0.000703` (raises CT win probability)
- `lag_13__T3__flash_duration`: coefficient `0.000663` (raises CT win probability)
- `lag_11__CT_B_site_active_infernos`: coefficient `0.000637` (raises CT win probability)
- `lag_02__CT_A_site_active_infernos`: coefficient `0.000620` (raises CT win probability)
- `lag_00__T5__flash`: coefficient `-0.000619` (lowers CT win probability)
- `lag_15__CT3__molly`: coefficient `0.000610` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_01__T_place_WALKWAY`: coefficient `0.002646` (raises CT win probability)
- `lag_00__T_place_WALKWAY`: coefficient `-0.002059` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001628` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001593` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001546` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001391` (raises CT win probability)
- `lag_01__T_place_MAIN`: coefficient `0.001361` (raises CT win probability)
- `lag_12__CT_place_MAIN`: coefficient `-0.001351` (lowers CT win probability)
- `lag_10__CT_shots_fired_sum`: coefficient `-0.001259` (lowers CT win probability)
- `lag_01__CT_shots_fired_sum`: coefficient `0.001225` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `71096`, seconds `42.50`, LSTM delta `+0.2966`

Top all feature movements:
- `lag_01__T_place_WALKWAY`: contribution `+0.035983`
- `lag_00__T_place_WALKWAY`: contribution `+0.028003`
- `lag_01__T_place_MAIN`: contribution `+0.008797`
- `lag_10__CT_shots_fired_sum`: contribution `+0.008749`
- `lag_04__T3__flash_duration`: contribution `+0.008184`

Top utility-only movements:
- `lag_04__T3__flash_duration`: contribution `+0.008184`
- `lag_04__T_flash_duration_sum`: contribution `+0.006122`
- `lag_04__T1__flash_duration`: contribution `+0.006092`
- `lag_02__CT_A_site_active_infernos`: contribution `+0.002189`

### tick `71928`, seconds `55.50`, LSTM delta `+0.1418`

Top all feature movements:
- `lag_01__T_place_WALKWAY`: contribution `+0.035983`
- `lag_00__T_place_WALKWAY`: contribution `+0.028003`
- `lag_13__CT_place_OUTSIDELONG`: contribution `+0.009086`
- `lag_06__CT_place_TSPAWN`: contribution `+0.006290`
- `lag_00__CT_shots_fired_sum`: contribution `+0.005533`

Top utility-only movements:
- `lag_13__T3__flash_duration`: contribution `+0.004728`
- `lag_00__T3__flash_duration`: contribution `-0.001752`

### tick `70744`, seconds `37.00`, LSTM delta `+0.1175`

Top all feature movements:
- `lag_12__CT_place_MAIN`: contribution `+0.009098`
- `lag_00__CT_shots_fired_sum`: contribution `+0.006639`
- `lag_03__CT_place_MAIN`: contribution `+0.006353`
- `lag_00__CT_kills_last_3s`: contribution `+0.004464`
- `lag_00__kill_diff_last_3s`: contribution `+0.003918`

Top utility-only movements:
- `lag_11__CT_B_site_active_infernos`: contribution `+0.002189`
- `lag_00__T4__utility_total`: contribution `+0.001343`
- `lag_06__CT5__molly`: contribution `+0.001259`

### tick `71064`, seconds `42.00`, LSTM delta `-0.0896`

Top all feature movements:
- `lag_00__T_place_WALKWAY`: contribution `-0.028003`
- `lag_10__CT_shots_fired_sum`: contribution `-0.005250`
- `lag_03__T3__flash_duration`: contribution `-0.003637`
- `lag_09__CT_shots_fired_sum`: contribution `-0.002587`
- `lag_10__CT1__shots_fired`: contribution `-0.002479`

Top utility-only movements:
- `lag_03__T3__flash_duration`: contribution `-0.003637`
- `lag_03__T_flash_duration_sum`: contribution `-0.001646`

### tick `70840`, seconds `38.50`, LSTM delta `-0.0862`

Top all feature movements:
- `lag_15__CT_place_MAIN`: contribution `-0.007018`
- `lag_02__CT_place_MAIN`: contribution `-0.006365`
- `lag_02__CT_shots_fired_sum`: contribution `-0.006269`
- `lag_00__kill_diff_last_3s`: contribution `-0.003918`
- `lag_02__CT1__shots_fired`: contribution `-0.003194`

Top utility-only movements:
- `lag_00__CT_B_site_active_infernos`: contribution `-0.001832`
- `lag_14__CT_B_site_active_infernos`: contribution `-0.001374`
