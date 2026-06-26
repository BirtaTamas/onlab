# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-imperial-vs-legacy-bo3-GRvbnL5Q4zT_JzAd-0AXgo/imperial-vs-legacy-m2-dust2.csv`
- round_num: `5`

## Largest probability jumps

- tick `38489`, seconds `117.50`, LSTM `0.4322`, delta `-0.2897`
- tick `36569`, seconds `87.50`, LSTM `0.5735`, delta `+0.2825`
- tick `36345`, seconds `84.00`, LSTM `0.3928`, delta `-0.1669`
- tick `38457`, seconds `117.00`, LSTM `0.7219`, delta `-0.1576`
- tick `35833`, seconds `76.00`, LSTM `0.5480`, delta `+0.1241`
- tick `37305`, seconds `99.00`, LSTM `0.9200`, delta `+0.1121`
- tick `39641`, seconds `135.50`, LSTM `0.1296`, delta `-0.1107`
- tick `37273`, seconds `98.50`, LSTM `0.8079`, delta `+0.1048`
- tick `36377`, seconds `84.50`, LSTM `0.2959`, delta `-0.0969`
- tick `39801`, seconds `138.00`, LSTM `0.0349`, delta `-0.0852`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003797`, |coef| `0.003797`
- `lag_00__damage_diff_last_5s`: coefficient `0.003356`, |coef| `0.003356`
- `lag_00__CT_place_CATWALK`: coefficient `0.003278`, |coef| `0.003278`
- `lag_01__T_place_BDOORS`: coefficient `-0.003264`, |coef| `0.003264`
- `lag_00__T_kills_last_3s`: coefficient `-0.003169`, |coef| `0.003169`
- `lag_00__T_damage_last_5s`: coefficient `-0.002866`, |coef| `0.002866`
- `lag_11__T_bomb_zone_count`: coefficient `0.002664`, |coef| `0.002664`
- `lag_01__CT_place_CATWALK`: coefficient `0.002481`, |coef| `0.002481`
- `lag_01__T_kills_last_3s`: coefficient `-0.002351`, |coef| `0.002351`
- `lag_12__CT_velocity_mean`: coefficient `-0.002271`, |coef| `0.002271`
- `lag_10__CT_place_HOLE`: coefficient `0.002216`, |coef| `0.002216`
- `lag_00__CT_duck_amount_mean`: coefficient `0.002149`, |coef| `0.002149`
- `lag_11__CT3__is_walking`: coefficient `0.002116`, |coef| `0.002116`
- `lag_10__T_bomb_zone_count`: coefficient `0.002078`, |coef| `0.002078`
- `lag_01__kill_diff_last_3s`: coefficient `0.002029`, |coef| `0.002029`

## Top 10 utility ridge features

- `lag_15__T1__flash_duration`: coefficient `-0.001564` (lowers CT win probability)
- `lag_01__T_A_site_active_infernos`: coefficient `0.001529` (raises CT win probability)
- `lag_15__T_A_site_active_infernos`: coefficient `-0.001435` (lowers CT win probability)
- `lag_01__CT2__molly`: coefficient `0.001368` (raises CT win probability)
- `lag_09__T5__flash_duration`: coefficient `-0.001320` (lowers CT win probability)
- `lag_15__T5__flash_duration`: coefficient `-0.001309` (lowers CT win probability)
- `lag_00__T_flashes_last_5s`: coefficient `-0.001179` (lowers CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `-0.001177` (lowers CT win probability)
- `lag_14__T_A_site_active_infernos`: coefficient `-0.001149` (lowers CT win probability)
- `lag_00__CT2__molly`: coefficient `0.001146` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003797` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003356` (raises CT win probability)
- `lag_00__CT_place_CATWALK`: coefficient `0.003278` (raises CT win probability)
- `lag_01__T_place_BDOORS`: coefficient `-0.003264` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003169` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002866` (lowers CT win probability)
- `lag_11__T_bomb_zone_count`: coefficient `0.002664` (raises CT win probability)
- `lag_01__CT_place_CATWALK`: coefficient `0.002481` (raises CT win probability)
- `lag_01__T_kills_last_3s`: coefficient `-0.002351` (lowers CT win probability)
- `lag_12__CT_velocity_mean`: coefficient `-0.002271` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `38489`, seconds `117.50`, LSTM delta `-0.2897`

Top all feature movements:
- `lag_11__T_bomb_zone_count`: contribution `-0.015511`
- `lag_00__CT_place_CATWALK`: contribution `-0.013058`
- `lag_00__T_kills_last_3s`: contribution `-0.010038`
- `lag_01__CT_place_CATWALK`: contribution `-0.009884`
- `lag_00__kill_diff_last_3s`: contribution `-0.009140`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `36569`, seconds `87.50`, LSTM delta `+0.2825`

Top all feature movements:
- `lag_01__T_place_BDOORS`: contribution `+0.040830`
- `lag_07__T_place_BDOORS`: contribution `+0.015847`
- `lag_06__T_place_BDOORS`: contribution `+0.015614`
- `lag_09__T5__flash_duration`: contribution `+0.009598`
- `lag_00__kill_diff_last_3s`: contribution `+0.009140`

Top utility-only movements:
- `lag_09__T5__flash_duration`: contribution `+0.009598`
- `lag_15__T1__flash_duration`: contribution `+0.006266`

### tick `36345`, seconds `84.00`, LSTM delta `-0.1669`

Top all feature movements:
- `lag_00__T_place_BDOORS`: contribution `-0.022881`
- `lag_03__CT_shots_fired_sum`: contribution `-0.013786`
- `lag_00__T_kills_last_3s`: contribution `-0.010038`
- `lag_15__T5__flash_duration`: contribution `-0.009523`
- `lag_00__kill_diff_last_3s`: contribution `-0.009140`

Top utility-only movements:
- `lag_15__T5__flash_duration`: contribution `-0.009523`
- `lag_02__T5__flash_duration`: contribution `-0.007870`
- `lag_15__T1__flash_duration`: contribution `-0.006266`
- `lag_15__T_flash_duration_sum`: contribution `-0.005070`

### tick `38457`, seconds `117.00`, LSTM delta `-0.1576`

Top all feature movements:
- `lag_00__CT_place_CATWALK`: contribution `-0.013058`
- `lag_10__T_bomb_zone_count`: contribution `-0.012099`
- `lag_00__T_kills_last_3s`: contribution `-0.010038`
- `lag_00__kill_diff_last_3s`: contribution `-0.009140`
- `lag_00__T_damage_last_5s`: contribution `-0.006872`

Top utility-only movements:
- `lag_14__T_A_site_active_infernos`: contribution `-0.003421`

### tick `35833`, seconds `76.00`, LSTM delta `+0.1241`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.009140`
- `lag_00__damage_diff_last_5s`: contribution `+0.007571`
- `lag_01__CT4__duck_amount`: contribution `+0.007229`
- `lag_06__CT5__is_scoped`: contribution `+0.006918`
- `lag_07__CT4__duck_amount`: contribution `+0.006293`

Top utility-only movements:
- No utility movement among the top local contributors.
