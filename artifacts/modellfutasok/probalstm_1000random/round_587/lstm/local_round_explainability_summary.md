# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-heroic-vs-flyquest-bo3-ErQHzvBcWPHiA-H04IjPMf/heroic-vs-flyquest-m2-anubis.csv`
- round_num: `22`

## Largest probability jumps

- tick `180938`, seconds `26.50`, LSTM `0.1184`, delta `-0.2991`
- tick `180842`, seconds `25.00`, LSTM `0.4085`, delta `-0.0402`
- tick `180970`, seconds `27.00`, LSTM `0.0791`, delta `-0.0393`
- tick `180714`, seconds `23.00`, LSTM `0.4697`, delta `+0.0389`
- tick `180778`, seconds `24.00`, LSTM `0.4610`, delta `-0.0284`
- tick `181834`, seconds `40.50`, LSTM `0.0722`, delta `+0.0218`
- tick `180746`, seconds `23.50`, LSTM `0.4894`, delta `+0.0197`
- tick `179370`, seconds `2.00`, LSTM `0.4685`, delta `+0.0172`
- tick `180650`, seconds `22.00`, LSTM `0.4313`, delta `-0.0148`
- tick `179530`, seconds `4.50`, LSTM `0.4534`, delta `-0.0127`

## Top 15 local ridge features

- `lag_02__CT_place_BRICKS`: coefficient `0.001394`, |coef| `0.001394`
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `0.001242`, |coef| `0.001242`
- `lag_00__CT_place_LOWERTUNNEL`: coefficient `-0.001209`, |coef| `0.001209`
- `lag_03__CT_place_BRICKS`: coefficient `-0.001179`, |coef| `0.001179`
- `lag_00__CT5__utility_total`: coefficient `0.001075`, |coef| `0.001075`
- `lag_09__CT1__flash_duration`: coefficient `-0.001043`, |coef| `0.001043`
- `lag_05__CT5__is_scoped`: coefficient `-0.001043`, |coef| `0.001043`
- `lag_00__CT5__flash`: coefficient `0.001009`, |coef| `0.001009`
- `lag_05__CT1__flash_duration`: coefficient `0.001000`, |coef| `0.001000`
- `lag_15__CT_place_MAIN`: coefficient `-0.000899`, |coef| `0.000899`
- `lag_09__T_place_MIDDOORS`: coefficient `-0.000846`, |coef| `0.000846`
- `lag_12__T_flashed_players`: coefficient `-0.000787`, |coef| `0.000787`
- `lag_08__CT5__is_scoped`: coefficient `-0.000773`, |coef| `0.000773`
- `lag_01__CT5__is_scoped`: coefficient `-0.000760`, |coef| `0.000760`
- `lag_07__T_place_BRIDGE`: coefficient `0.000756`, |coef| `0.000756`

## Top 10 utility ridge features

- `lag_00__CT5__utility_total`: coefficient `0.001075` (raises CT win probability)
- `lag_09__CT1__flash_duration`: coefficient `-0.001043` (lowers CT win probability)
- `lag_00__CT5__flash`: coefficient `0.001009` (raises CT win probability)
- `lag_05__CT1__flash_duration`: coefficient `0.001000` (raises CT win probability)
- `lag_00__CT5__molly`: coefficient `0.000705` (raises CT win probability)
- `lag_09__CT_flash_duration_sum`: coefficient `-0.000666` (lowers CT win probability)
- `lag_13__CT_A_site_active_infernos`: coefficient `0.000627` (raises CT win probability)
- `lag_00__CT5__smoke`: coefficient `0.000624` (raises CT win probability)
- `lag_08__CT_B_site_active_infernos`: coefficient `0.000557` (raises CT win probability)
- `lag_08__CT1__molly`: coefficient `0.000541` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_02__CT_place_BRICKS`: coefficient `0.001394` (raises CT win probability)
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `0.001242` (raises CT win probability)
- `lag_00__CT_place_LOWERTUNNEL`: coefficient `-0.001209` (lowers CT win probability)
- `lag_03__CT_place_BRICKS`: coefficient `-0.001179` (lowers CT win probability)
- `lag_05__CT5__is_scoped`: coefficient `-0.001043` (lowers CT win probability)
- `lag_15__CT_place_MAIN`: coefficient `-0.000899` (lowers CT win probability)
- `lag_09__T_place_MIDDOORS`: coefficient `-0.000846` (lowers CT win probability)
- `lag_12__T_flashed_players`: coefficient `-0.000787` (lowers CT win probability)
- `lag_08__CT5__is_scoped`: coefficient `-0.000773` (lowers CT win probability)
- `lag_01__CT5__is_scoped`: coefficient `-0.000760` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `180938`, seconds `26.50`, LSTM delta `-0.2991`

Top all feature movements:
- `lag_02__CT_place_BRICKS`: contribution `-0.026770`
- `lag_03__CT_place_BRICKS`: contribution `-0.022630`
- `lag_00__CT_place_LOWERTUNNEL`: contribution `-0.008886`
- `lag_09__CT1__flash_duration`: contribution `-0.006725`
- `lag_05__CT1__flash_duration`: contribution `-0.006447`

Top utility-only movements:
- `lag_09__CT1__flash_duration`: contribution `-0.006725`
- `lag_05__CT1__flash_duration`: contribution `-0.006447`
- `lag_00__CT5__utility_total`: contribution `-0.004061`
- `lag_00__CT5__flash`: contribution `-0.003583`

### tick `180842`, seconds `25.00`, LSTM delta `-0.0402`

Top all feature movements:
- `lag_05__CT5__is_scoped`: contribution `-0.003729`
- `lag_07__T_place_BRIDGE`: contribution `+0.003273`
- `lag_02__CT1__flash_duration`: contribution `-0.003240`
- `lag_07__T_place_STREET`: contribution `-0.003219`
- `lag_09__T_flashed_players`: contribution `+0.002624`

Top utility-only movements:
- `lag_02__CT1__flash_duration`: contribution `-0.003240`
- `lag_06__CT1__flash_duration`: contribution `+0.001238`
- `lag_02__CT_flash_duration_sum`: contribution `-0.000907`

### tick `180970`, seconds `27.00`, LSTM delta `-0.0393`

Top all feature movements:
- `lag_03__CT_place_BRICKS`: contribution `+0.022630`
- `lag_01__CT_place_LOWERTUNNEL`: contribution `-0.003581`
- `lag_04__CT_place_BRICKS`: contribution `-0.003530`
- `lag_08__CT5__is_scoped`: contribution `+0.002764`
- `lag_10__CT1__flash_duration`: contribution `-0.002687`

Top utility-only movements:
- `lag_10__CT1__flash_duration`: contribution `-0.002687`
- `lag_01__CT5__utility_total`: contribution `-0.001752`
- `lag_01__CT5__flash`: contribution `-0.001520`

### tick `180714`, seconds `23.00`, LSTM delta `+0.0389`

Top all feature movements:
- `lag_03__T_place_STREET`: contribution `+0.003439`
- `lag_02__CT1__flash_duration`: contribution `+0.003240`
- `lag_01__CT5__is_scoped`: contribution `-0.002717`
- `lag_00__CT5__is_scoped`: contribution `+0.002398`
- `lag_14__CT_place_MAIN`: contribution `+0.002206`

Top utility-only movements:
- `lag_02__CT1__flash_duration`: contribution `+0.003240`
- `lag_02__CT_flash_duration_sum`: contribution `+0.001301`

### tick `180778`, seconds `24.00`, LSTM delta `-0.0284`

Top all feature movements:
- `lag_15__CT_place_MAIN`: contribution `-0.006050`
- `lag_00__CT5__is_scoped`: contribution `-0.002398`
- `lag_04__CT_flashed_players`: contribution `+0.002040`
- `lag_04__CT1__flash_duration`: contribution `+0.001946`
- `lag_05__CT2__duck_amount`: contribution `-0.001944`

Top utility-only movements:
- `lag_04__CT1__flash_duration`: contribution `+0.001946`
- `lag_04__CT_flash_duration_sum`: contribution `+0.001446`
- `lag_04__CT5__flash_duration`: contribution `+0.001220`
