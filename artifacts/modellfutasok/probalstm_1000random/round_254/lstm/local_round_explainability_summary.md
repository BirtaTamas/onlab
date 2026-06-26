# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-natus-vincere-vs-spirit-bo3-cW0x-KCT4cbPLaZUAvb08Z/natus-vincere-vs-spirit-m2-ancient.csv`
- round_num: `23`

## Largest probability jumps

- tick `179641`, seconds `52.50`, LSTM `0.7756`, delta `+0.2433`
- tick `177369`, seconds `17.00`, LSTM `0.3413`, delta `-0.1768`
- tick `179321`, seconds `47.50`, LSTM `0.6848`, delta `+0.1670`
- tick `177081`, seconds `12.50`, LSTM `0.5001`, delta `-0.1540`
- tick `179577`, seconds `51.50`, LSTM `0.5374`, delta `-0.1245`
- tick `178329`, seconds `32.00`, LSTM `0.5321`, delta `+0.1021`
- tick `177913`, seconds `25.50`, LSTM `0.4057`, delta `+0.1017`
- tick `178233`, seconds `30.50`, LSTM `0.3100`, delta `-0.0992`
- tick `179545`, seconds `51.00`, LSTM `0.6619`, delta `-0.0656`
- tick `178297`, seconds `31.50`, LSTM `0.4300`, delta `+0.0649`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003021`, |coef| `0.003021`
- `lag_00__CT_kills_last_3s`: coefficient `0.002403`, |coef| `0.002403`
- `lag_03__T_utility_damage_last_5s`: coefficient `0.002274`, |coef| `0.002274`
- `lag_00__damage_diff_last_5s`: coefficient `0.001863`, |coef| `0.001863`
- `lag_07__CT5__is_scoped`: coefficient `0.001776`, |coef| `0.001776`
- `lag_01__T_utility_damage_last_5s`: coefficient `-0.001700`, |coef| `0.001700`
- `lag_14__CT4__flash_duration`: coefficient `-0.001677`, |coef| `0.001677`
- `lag_01__T_place_WATER`: coefficient `-0.001604`, |coef| `0.001604`
- `lag_09__T_place_TUNNEL`: coefficient `0.001551`, |coef| `0.001551`
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.001514`, |coef| `0.001514`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001479`, |coef| `0.001479`
- `lag_09__CT5__is_scoped`: coefficient `0.001473`, |coef| `0.001473`
- `lag_13__T4__is_walking`: coefficient `0.001388`, |coef| `0.001388`
- `lag_06__T_place_TSIDELOWER`: coefficient `-0.001375`, |coef| `0.001375`
- `lag_04__T_B_site_active_infernos`: coefficient `0.001352`, |coef| `0.001352`

## Top 10 utility ridge features

- `lag_03__T_utility_damage_last_5s`: coefficient `0.002274` (raises CT win probability)
- `lag_01__T_utility_damage_last_5s`: coefficient `-0.001700` (lowers CT win probability)
- `lag_14__CT4__flash_duration`: coefficient `-0.001677` (lowers CT win probability)
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.001514` (lowers CT win probability)
- `lag_04__T_B_site_active_infernos`: coefficient `0.001352` (raises CT win probability)
- `lag_03__utility_damage_diff_last_5s`: coefficient `-0.001337` (lowers CT win probability)
- `lag_15__CT4__flash_duration`: coefficient `-0.001288` (lowers CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.001220` (raises CT win probability)
- `lag_01__utility_damage_diff_last_5s`: coefficient `0.001157` (raises CT win probability)
- `lag_13__T_B_site_active_infernos`: coefficient `-0.001153` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003021` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002403` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001863` (raises CT win probability)
- `lag_07__CT5__is_scoped`: coefficient `0.001776` (raises CT win probability)
- `lag_01__T_place_WATER`: coefficient `-0.001604` (lowers CT win probability)
- `lag_09__T_place_TUNNEL`: coefficient `0.001551` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001479` (raises CT win probability)
- `lag_09__CT5__is_scoped`: coefficient `0.001473` (raises CT win probability)
- `lag_13__T4__is_walking`: coefficient `0.001388` (raises CT win probability)
- `lag_06__T_place_TSIDELOWER`: coefficient `-0.001375` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `179641`, seconds `52.50`, LSTM delta `+0.2433`

Top all feature movements:
- `lag_03__T_utility_damage_last_5s`: contribution `+0.017205`
- `lag_00__CT_shots_fired_sum`: contribution `+0.008218`
- `lag_00__kill_diff_last_3s`: contribution `+0.007272`
- `lag_00__CT_kills_last_3s`: contribution `+0.006937`
- `lag_03__utility_damage_diff_last_5s`: contribution `+0.006399`

Top utility-only movements:
- `lag_03__T_utility_damage_last_5s`: contribution `+0.017205`
- `lag_03__utility_damage_diff_last_5s`: contribution `+0.006399`
- `lag_04__T_B_site_active_infernos`: contribution `+0.003823`
- `lag_04__T_active_infernos`: contribution `+0.002062`

### tick `177369`, seconds `17.00`, LSTM delta `-0.1768`

Top all feature movements:
- `lag_09__T_place_TUNNEL`: contribution `-0.009419`
- `lag_00__CT_kills_last_3s`: contribution `+0.006937`
- `lag_00__CT_place_TSIDEUPPER`: contribution `-0.006475`
- `lag_07__CT_utility_damage_last_5s`: contribution `-0.006150`
- `lag_04__CT_place_TSIDEUPPER`: contribution `-0.005540`

Top utility-only movements:
- `lag_07__CT_utility_damage_last_5s`: contribution `-0.006150`
- `lag_10__CT3__flash_duration`: contribution `-0.004941`
- `lag_07__utility_damage_diff_last_5s`: contribution `-0.004323`
- `lag_04__T_B_site_active_infernos`: contribution `-0.003823`
- `lag_12__CT2__flash_duration`: contribution `-0.003170`

### tick `179321`, seconds `47.50`, LSTM delta `+0.1670`

Top all feature movements:
- `lag_14__CT4__flash_duration`: contribution `+0.007602`
- `lag_00__kill_diff_last_3s`: contribution `+0.007272`
- `lag_00__CT_kills_last_3s`: contribution `+0.006937`
- `lag_09__CT5__is_scoped`: contribution `+0.005269`
- `lag_06__T_place_TSIDELOWER`: contribution `+0.005155`

Top utility-only movements:
- `lag_14__CT4__flash_duration`: contribution `+0.007602`
- `lag_13__T_B_site_active_infernos`: contribution `+0.003261`
- `lag_05__T5__molly`: contribution `+0.002140`
- `lag_01__T_B_site_active_infernos`: contribution `+0.001949`

### tick `177081`, seconds `12.50`, LSTM delta `-0.1540`

Top all feature movements:
- `lag_12__T_he_last_5s`: contribution `-0.013429`
- `lag_12__CT_flashes_last_5s`: contribution `-0.009527`
- `lag_06__T_place_TUNNEL`: contribution `-0.007801`
- `lag_00__kill_diff_last_3s`: contribution `-0.007272`
- `lag_10__CT3__flash_duration`: contribution `-0.004848`

Top utility-only movements:
- `lag_12__T_he_last_5s`: contribution `-0.013429`
- `lag_12__CT_flashes_last_5s`: contribution `-0.009527`
- `lag_10__CT3__flash_duration`: contribution `-0.004848`
- `lag_08__CT_utility_damage_last_5s`: contribution `-0.002893`
- `lag_10__CT_flash_duration_sum`: contribution `-0.002774`

### tick `179577`, seconds `51.50`, LSTM delta `-0.1245`

Top all feature movements:
- `lag_01__T_utility_damage_last_5s`: contribution `-0.012860`
- `lag_00__kill_diff_last_3s`: contribution `-0.007272`
- `lag_01__utility_damage_diff_last_5s`: contribution `-0.005538`
- `lag_00__T_kills_last_3s`: contribution `-0.004245`
- `lag_08__CT5__is_scoped`: contribution `-0.004222`

Top utility-only movements:
- `lag_01__T_utility_damage_last_5s`: contribution `-0.012860`
- `lag_01__utility_damage_diff_last_5s`: contribution `-0.005538`
- `lag_09__T_B_site_active_infernos`: contribution `-0.002092`
