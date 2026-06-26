# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m1-inferno.csv`
- round_num: `9`

## Largest probability jumps

- tick `83263`, seconds `52.50`, LSTM `0.8521`, delta `+0.1428`
- tick `82047`, seconds `33.50`, LSTM `0.6913`, delta `+0.1057`
- tick `83231`, seconds `52.00`, LSTM `0.7094`, delta `+0.0850`
- tick `82591`, seconds `42.00`, LSTM `0.7022`, delta `+0.0761`
- tick `82655`, seconds `43.00`, LSTM `0.6738`, delta `-0.0636`
- tick `83167`, seconds `51.00`, LSTM `0.6588`, delta `-0.0508`
- tick `83327`, seconds `53.50`, LSTM `0.9422`, delta `+0.0481`
- tick `83135`, seconds `50.50`, LSTM `0.7096`, delta `+0.0464`
- tick `83295`, seconds `53.00`, LSTM `0.8940`, delta `+0.0419`
- tick `82367`, seconds `38.50`, LSTM `0.6425`, delta `-0.0402`

## Top 15 local ridge features

- `lag_00__T_place_QUAD`: coefficient `0.001684`, |coef| `0.001684`
- `lag_03__T5__is_scoped`: coefficient `0.001296`, |coef| `0.001296`
- `lag_03__T_place_ARCH`: coefficient `-0.001290`, |coef| `0.001290`
- `lag_00__kill_diff_last_3s`: coefficient `0.001181`, |coef| `0.001181`
- `lag_00__T_bomb_zone_count`: coefficient `-0.001131`, |coef| `0.001131`
- `lag_00__CT_kills_last_3s`: coefficient `0.001093`, |coef| `0.001093`
- `lag_06__T_flashed_players`: coefficient `0.001068`, |coef| `0.001068`
- `lag_01__T_place_QUAD`: coefficient `0.001009`, |coef| `0.001009`
- `lag_07__CT5__duck_amount`: coefficient `-0.001001`, |coef| `0.001001`
- `lag_15__CT_flashes_last_5s`: coefficient `0.000948`, |coef| `0.000948`
- `lag_06__CT3__is_walking`: coefficient `-0.000943`, |coef| `0.000943`
- `lag_03__CT_place_PIT`: coefficient `-0.000931`, |coef| `0.000931`
- `lag_02__T_place_ARCH`: coefficient `-0.000894`, |coef| `0.000894`
- `lag_07__T2__is_walking`: coefficient `-0.000854`, |coef| `0.000854`
- `lag_00__T_duck_amount_mean`: coefficient `0.000829`, |coef| `0.000829`

## Top 10 utility ridge features

- `lag_15__CT_flashes_last_5s`: coefficient `0.000948` (raises CT win probability)
- `lag_06__T_flash_duration_sum`: coefficient `0.000583` (raises CT win probability)
- `lag_05__CT4__flash_duration`: coefficient `0.000574` (raises CT win probability)
- `lag_08__T2__flash_duration`: coefficient `0.000571` (raises CT win probability)
- `lag_14__CT2__molly`: coefficient `0.000557` (raises CT win probability)
- `lag_02__T_A_site_active_infernos`: coefficient `-0.000536` (lowers CT win probability)
- `lag_06__T1__flash_duration`: coefficient `0.000536` (raises CT win probability)
- `lag_00__T3__smoke`: coefficient `-0.000532` (lowers CT win probability)
- `lag_06__T5__flash_duration`: coefficient `0.000523` (raises CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `-0.000501` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_QUAD`: coefficient `0.001684` (raises CT win probability)
- `lag_03__T5__is_scoped`: coefficient `0.001296` (raises CT win probability)
- `lag_03__T_place_ARCH`: coefficient `-0.001290` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001181` (raises CT win probability)
- `lag_00__T_bomb_zone_count`: coefficient `-0.001131` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001093` (raises CT win probability)
- `lag_06__T_flashed_players`: coefficient `0.001068` (raises CT win probability)
- `lag_01__T_place_QUAD`: coefficient `0.001009` (raises CT win probability)
- `lag_07__CT5__duck_amount`: coefficient `-0.001001` (lowers CT win probability)
- `lag_06__CT3__is_walking`: coefficient `-0.000943` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `83263`, seconds `52.50`, LSTM delta `+0.1428`

Top all feature movements:
- `lag_03__T_place_ARCH`: contribution `+0.012002`
- `lag_00__T_bomb_zone_count`: contribution `+0.006584`
- `lag_00__T_duck_amount_mean`: contribution `+0.004436`
- `lag_02__T_bomb_zone_count`: contribution `+0.004066`
- `lag_03__CT_place_PIT`: contribution `+0.004010`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `82047`, seconds `33.50`, LSTM delta `+0.1057`

Top all feature movements:
- `lag_06__T_flashed_players`: contribution `+0.006184`
- `lag_03__T5__is_scoped`: contribution `+0.006180`
- `lag_00__CT_kills_last_3s`: contribution `+0.003155`
- `lag_00__kill_diff_last_3s`: contribution `+0.002842`
- `lag_03__CT1__duck_amount`: contribution `+0.002812`

Top utility-only movements:
- `lag_06__T1__flash_duration`: contribution `+0.001653`
- `lag_05__CT4__flash_duration`: contribution `+0.001586`
- `lag_06__T_flash_duration_sum`: contribution `+0.001473`
- `lag_00__CT4__flash_duration`: contribution `+0.001385`
- `lag_14__CT2__molly`: contribution `+0.001373`

### tick `83231`, seconds `52.00`, LSTM delta `+0.0850`

Top all feature movements:
- `lag_02__T_place_ARCH`: contribution `+0.008318`
- `lag_03__CT_place_LIBRARY`: contribution `+0.005297`
- `lag_13__T5__is_scoped`: contribution `+0.003850`
- `lag_07__CT2__duck_amount`: contribution `+0.003149`
- `lag_02__CT_place_PIT`: contribution `+0.003054`

Top utility-only movements:
- `lag_01__T_A_site_active_infernos`: contribution `+0.001371`

### tick `82591`, seconds `42.00`, LSTM delta `+0.0761`

Top all feature movements:
- `lag_00__T_place_QUAD`: contribution `+0.040555`
- `lag_03__T5__is_scoped`: contribution `+0.006180`
- `lag_01__CT_place_LIBRARY`: contribution `+0.003668`
- `lag_08__T2__flash_duration`: contribution `+0.003508`
- `lag_02__T5__is_scoped`: contribution `-0.002706`

Top utility-only movements:
- `lag_08__T2__flash_duration`: contribution `+0.003508`
- `lag_08__CT2__flash_duration`: contribution `+0.001209`
- `lag_14__T2__flash_duration`: contribution `+0.000973`

### tick `82655`, seconds `43.00`, LSTM delta `-0.0636`

Top all feature movements:
- `lag_00__T_place_QUAD`: contribution `-0.040555`
- `lag_02__T_place_QUAD`: contribution `-0.007138`
- `lag_03__CT_place_LIBRARY`: contribution `-0.005297`
- `lag_13__T5__is_scoped`: contribution `+0.003850`
- `lag_05__T5__is_scoped`: contribution `-0.002969`

Top utility-only movements:
- `lag_05__CT2__flash_duration`: contribution `+0.001060`
