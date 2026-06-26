# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-heroic-vs-flyquest-bo3-ErQHzvBcWPHiA-H04IjPMf/heroic-vs-flyquest-m2-anubis.csv`
- round_num: `24`

## Largest probability jumps

- tick `197233`, seconds `48.00`, LSTM `0.3092`, delta `-0.3496`
- tick `197969`, seconds `59.50`, LSTM `0.5507`, delta `+0.2387`
- tick `197137`, seconds `46.50`, LSTM `0.7141`, delta `+0.2165`
- tick `196881`, seconds `42.50`, LSTM `0.4484`, delta `-0.1652`
- tick `198577`, seconds `69.00`, LSTM `0.8790`, delta `+0.1149`
- tick `196849`, seconds `42.00`, LSTM `0.6136`, delta `+0.0829`
- tick `197681`, seconds `55.00`, LSTM `0.3615`, delta `+0.0680`
- tick `198225`, seconds `63.50`, LSTM `0.7790`, delta `+0.0614`
- tick `197745`, seconds `56.00`, LSTM `0.3161`, delta `-0.0601`
- tick `197169`, seconds `47.00`, LSTM `0.6620`, delta `-0.0521`

## Top 15 local ridge features

- `lag_07__T_place_CTSIDEUPPER`: coefficient `0.002153`, |coef| `0.002153`
- `lag_00__CT5__flash_duration`: coefficient `0.001953`, |coef| `0.001953`
- `lag_04__CT4__flash_duration`: coefficient `-0.001812`, |coef| `0.001812`
- `lag_00__kill_diff_last_3s`: coefficient `0.001768`, |coef| `0.001768`
- `lag_04__CT5__flash_duration`: coefficient `-0.001730`, |coef| `0.001730`
- `lag_04__CT_flash_duration_sum`: coefficient `-0.001685`, |coef| `0.001685`
- `lag_06__CT_place_BRICKS`: coefficient `-0.001656`, |coef| `0.001656`
- `lag_08__CT_place_WALKWAY`: coefficient `-0.001620`, |coef| `0.001620`
- `lag_00__damage_diff_last_5s`: coefficient `0.001597`, |coef| `0.001597`
- `lag_10__T_place_LOWERTUNNEL`: coefficient `0.001472`, |coef| `0.001472`
- `lag_04__CT_flashed_players`: coefficient `-0.001472`, |coef| `0.001472`
- `lag_04__CT_place_BRICKS`: coefficient `-0.001441`, |coef| `0.001441`
- `lag_11__CT_place_MAIN`: coefficient `0.001436`, |coef| `0.001436`
- `lag_15__CT_place_BRICKS`: coefficient `-0.001431`, |coef| `0.001431`
- `lag_01__CT5__flash_duration`: coefficient `0.001422`, |coef| `0.001422`

## Top 10 utility ridge features

- `lag_00__CT5__flash_duration`: coefficient `0.001953` (raises CT win probability)
- `lag_04__CT4__flash_duration`: coefficient `-0.001812` (lowers CT win probability)
- `lag_04__CT5__flash_duration`: coefficient `-0.001730` (lowers CT win probability)
- `lag_04__CT_flash_duration_sum`: coefficient `-0.001685` (lowers CT win probability)
- `lag_01__CT5__flash_duration`: coefficient `0.001422` (raises CT win probability)
- `lag_14__CT4__flash_duration`: coefficient `-0.001223` (lowers CT win probability)
- `lag_09__CT_B_site_active_infernos`: coefficient `0.001131` (raises CT win probability)
- `lag_05__CT4__flash_duration`: coefficient `-0.001056` (lowers CT win probability)
- `lag_01__CT_flash_duration_sum`: coefficient `0.000963` (raises CT win probability)
- `lag_13__T_B_site_active_infernos`: coefficient `-0.000929` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_07__T_place_CTSIDEUPPER`: coefficient `0.002153` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001768` (raises CT win probability)
- `lag_06__CT_place_BRICKS`: coefficient `-0.001656` (lowers CT win probability)
- `lag_08__CT_place_WALKWAY`: coefficient `-0.001620` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001597` (raises CT win probability)
- `lag_10__T_place_LOWERTUNNEL`: coefficient `0.001472` (raises CT win probability)
- `lag_04__CT_flashed_players`: coefficient `-0.001472` (lowers CT win probability)
- `lag_04__CT_place_BRICKS`: coefficient `-0.001441` (lowers CT win probability)
- `lag_11__CT_place_MAIN`: coefficient `0.001436` (raises CT win probability)
- `lag_15__CT_place_BRICKS`: coefficient `-0.001431` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `197233`, seconds `48.00`, LSTM delta `-0.3496`

Top all feature movements:
- `lag_00__CT5__flash_duration`: contribution `-0.014674`
- `lag_04__CT4__flash_duration`: contribution `-0.013480`
- `lag_04__CT5__flash_duration`: contribution `-0.012993`
- `lag_04__CT_flash_duration_sum`: contribution `-0.012076`
- `lag_11__CT_place_MAIN`: contribution `-0.009671`

Top utility-only movements:
- `lag_00__CT5__flash_duration`: contribution `-0.014674`
- `lag_04__CT4__flash_duration`: contribution `-0.013480`
- `lag_04__CT5__flash_duration`: contribution `-0.012993`
- `lag_04__CT_flash_duration_sum`: contribution `-0.012076`

### tick `197969`, seconds `59.50`, LSTM delta `+0.2387`

Top all feature movements:
- `lag_07__T_place_CTSIDEUPPER`: contribution `+0.079343`
- `lag_14__CT4__flash_duration`: contribution `+0.009098`
- `lag_12__T_place_ALLEY`: contribution `+0.005456`
- `lag_11__CT_place_PALACEINTERIOR`: contribution `+0.005320`
- `lag_07__T_place_LOWERTUNNEL`: contribution `+0.005243`

Top utility-only movements:
- `lag_14__CT4__flash_duration`: contribution `+0.009098`
- `lag_09__CT_B_site_active_infernos`: contribution `+0.003884`
- `lag_14__CT_flash_duration_sum`: contribution `+0.002460`

### tick `197137`, seconds `46.50`, LSTM delta `+0.2165`

Top all feature movements:
- `lag_15__CT_place_BRICKS`: contribution `+0.027479`
- `lag_01__CT5__flash_duration`: contribution `+0.010685`
- `lag_01__CT_flash_duration_sum`: contribution `+0.006900`
- `lag_01__CT_flashed_players`: contribution `+0.006711`
- `lag_07__T_place_LOWERTUNNEL`: contribution `+0.005243`

Top utility-only movements:
- `lag_01__CT5__flash_duration`: contribution `+0.010685`
- `lag_01__CT_flash_duration_sum`: contribution `+0.006900`
- `lag_01__CT4__flash_duration`: contribution `+0.003965`

### tick `196881`, seconds `42.50`, LSTM delta `-0.1652`

Top all feature movements:
- `lag_15__CT_place_BRICKS`: contribution `-0.027479`
- `lag_09__T_place_CTSIDEUPPER`: contribution `-0.026284`
- `lag_10__CT_place_BRICKS`: contribution `-0.015929`
- `lag_11__CT_place_BRICKS`: contribution `-0.012894`
- `lag_07__CT_place_BRICKS`: contribution `+0.010525`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `198577`, seconds `69.00`, LSTM delta `+0.1149`

Top all feature movements:
- `lag_06__T_place_BRICKS`: contribution `+0.059058`
- `lag_00__T_place_CTSIDEUPPER`: contribution `+0.021870`
- `lag_09__T_place_BRICKS`: contribution `-0.013950`
- `lag_13__T_place_BRICKS`: contribution `+0.013054`
- `lag_00__T_shots_fired_sum`: contribution `+0.004640`

Top utility-only movements:
- `lag_09__CT_B_site_active_infernos`: contribution `+0.003884`
- `lag_11__T_flashes_last_5s`: contribution `+0.003362`
- `lag_01__T_flashes_last_5s`: contribution `+0.003255`
- `lag_09__CT_active_infernos`: contribution `+0.001717`
- `lag_11__CT4__molly`: contribution `+0.001707`
