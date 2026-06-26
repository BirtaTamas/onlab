# Local Round Explainability

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-faze-vs-aurora-bo3-ZssSxRC3p7Nn5A_BOLQ-lD/faze-vs-aurora-m2-mirage.csv`
- round_num: `9`

## Largest probability jumps

- tick `61877`, seconds `57.50`, LSTM `0.5506`, delta `+0.3557`
- tick `61813`, seconds `56.50`, LSTM `0.2690`, delta `-0.2693`
- tick `60853`, seconds `41.50`, LSTM `0.8255`, delta `+0.2576`
- tick `60821`, seconds `41.00`, LSTM `0.5678`, delta `-0.2204`
- tick `62165`, seconds `62.00`, LSTM `0.8150`, delta `+0.2044`
- tick `61301`, seconds `48.50`, LSTM `0.5304`, delta `-0.1775`
- tick `60757`, seconds `40.00`, LSTM `0.7918`, delta `+0.1402`
- tick `60725`, seconds `39.50`, LSTM `0.6517`, delta `-0.1239`
- tick `60629`, seconds `38.00`, LSTM `0.7586`, delta `+0.0790`
- tick `61845`, seconds `57.00`, LSTM `0.1950`, delta `-0.0740`

## Top 15 local ridge features

- `lag_12__T_place_SCAFFOLDING`: coefficient `-0.006070`, |coef| `0.006070`
- `lag_00__CT_defusing_count`: coefficient `0.004881`, |coef| `0.004881`
- `lag_00__CT_velocity_mean`: coefficient `-0.002501`, |coef| `0.002501`
- `lag_03__T_shots_fired_sum`: coefficient `-0.002378`, |coef| `0.002378`
- `lag_00__kill_diff_last_3s`: coefficient `0.002078`, |coef| `0.002078`
- `lag_00__T_shots_fired_sum`: coefficient `-0.002015`, |coef| `0.002015`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.001785`, |coef| `0.001785`
- `lag_02__CT_duck_amount_mean`: coefficient `-0.001781`, |coef| `0.001781`
- `lag_01__CT5__flash_duration`: coefficient `-0.001778`, |coef| `0.001778`
- `lag_02__CT_shots_fired_sum`: coefficient `-0.001743`, |coef| `0.001743`
- `lag_00__damage_diff_last_5s`: coefficient `0.001675`, |coef| `0.001675`
- `lag_09__CT_place_STAIRS`: coefficient `0.001637`, |coef| `0.001637`
- `lag_00__T_place_PALACEINTERIOR`: coefficient `-0.001617`, |coef| `0.001617`
- `lag_03__CT2__is_scoped`: coefficient `0.001569`, |coef| `0.001569`
- `lag_05__CT_place_STAIRS`: coefficient `-0.001535`, |coef| `0.001535`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.001785` (lowers CT win probability)
- `lag_01__CT5__flash_duration`: coefficient `-0.001778` (lowers CT win probability)
- `lag_11__CT_A_site_active_infernos`: coefficient `-0.001278` (lowers CT win probability)
- `lag_00__CT3__molly`: coefficient `0.001237` (raises CT win probability)
- `lag_02__CT_flash_duration_sum`: coefficient `0.001132` (raises CT win probability)
- `lag_02__CT2__flash_duration`: coefficient `0.001118` (raises CT win probability)
- `lag_01__CT_flash_duration_sum`: coefficient `-0.001117` (lowers CT win probability)
- `lag_10__CT_A_site_active_infernos`: coefficient `-0.001047` (lowers CT win probability)
- `lag_08__CT3__molly`: coefficient `0.001037` (raises CT win probability)
- `lag_07__CT2__flash_duration`: coefficient `0.001005` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_12__T_place_SCAFFOLDING`: coefficient `-0.006070` (lowers CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.004881` (raises CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.002501` (lowers CT win probability)
- `lag_03__T_shots_fired_sum`: coefficient `-0.002378` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002078` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.002015` (lowers CT win probability)
- `lag_02__CT_duck_amount_mean`: coefficient `-0.001781` (lowers CT win probability)
- `lag_02__CT_shots_fired_sum`: coefficient `-0.001743` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001675` (raises CT win probability)
- `lag_09__CT_place_STAIRS`: coefficient `0.001637` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `61877`, seconds `57.50`, LSTM delta `+0.3557`

Top all feature movements:
- `lag_12__T_place_SCAFFOLDING`: contribution `+0.206724`
- `lag_14__T_place_SCAFFOLDING`: contribution `+0.043416`
- `lag_00__CT_place_STAIRS`: contribution `+0.007756`
- `lag_00__CT_velocity_mean`: contribution `+0.007633`
- `lag_01__T_bomb_zone_count`: contribution `+0.005697`

Top utility-only movements:
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.004700`
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.004382`
- `lag_14__T4__flash_duration`: contribution `+0.003066`
- `lag_05__CT_A_site_active_smokes`: contribution `+0.001950`

### tick `61813`, seconds `56.50`, LSTM delta `-0.2693`

Top all feature movements:
- `lag_12__T_place_SCAFFOLDING`: contribution `-0.206724`
- `lag_10__T_place_SCAFFOLDING`: contribution `-0.021681`
- `lag_00__T_shots_fired_sum`: contribution `-0.007552`
- `lag_00__kill_diff_last_3s`: contribution `-0.005001`
- `lag_00__T_kills_last_3s`: contribution `-0.004710`

Top utility-only movements:
- `lag_13__T_A_site_active_infernos`: contribution `+0.001321`

### tick `60853`, seconds `41.50`, LSTM delta `+0.2576`

Top all feature movements:
- `lag_03__T_shots_fired_sum`: contribution `+0.012478`
- `lag_03__CT2__is_scoped`: contribution `+0.009601`
- `lag_01__CT5__flash_duration`: contribution `+0.008751`
- `lag_02__CT_shots_fired_sum`: contribution `+0.008474`
- `lag_08__CT_place_TRUCK`: contribution `+0.007815`

Top utility-only movements:
- `lag_01__CT5__flash_duration`: contribution `+0.008751`
- `lag_02__CT2__flash_duration`: contribution `+0.005521`
- `lag_02__CT_flash_duration_sum`: contribution `+0.005106`
- `lag_02__CT5__flash_duration`: contribution `+0.004220`
- `lag_06__CT1__flash_duration`: contribution `+0.003051`

### tick `60821`, seconds `41.00`, LSTM delta `-0.2204`

Top all feature movements:
- `lag_03__T_shots_fired_sum`: contribution `-0.012478`
- `lag_01__CT5__flash_duration`: contribution `-0.008751`
- `lag_07__CT_place_TRUCK`: contribution `-0.007072`
- `lag_02__CT_shots_fired_sum`: contribution `-0.006053`
- `lag_00__T_shots_fired_sum`: contribution `-0.006042`

Top utility-only movements:
- `lag_01__CT5__flash_duration`: contribution `-0.008751`
- `lag_01__CT_flash_duration_sum`: contribution `-0.005042`
- `lag_00__CT5__flash_duration`: contribution `-0.004455`
- `lag_05__CT1__flash_duration`: contribution `-0.003341`

### tick `62165`, seconds `62.00`, LSTM delta `+0.2044`

Top all feature movements:
- `lag_09__CT_place_STAIRS`: contribution `+0.012744`
- `lag_05__CT_place_STAIRS`: contribution `+0.011948`
- `lag_00__T_flash_alpha_mean`: contribution `+0.010829`
- `lag_02__CT_duck_amount_mean`: contribution `+0.010667`
- `lag_03__CT_duck_amount_mean`: contribution `+0.007920`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.010829`
- `lag_09__CT_utility_damage_last_5s`: contribution `+0.006145`
- `lag_09__utility_damage_diff_last_5s`: contribution `+0.005276`
- `lag_03__CT_A_site_active_infernos`: contribution `+0.002323`
