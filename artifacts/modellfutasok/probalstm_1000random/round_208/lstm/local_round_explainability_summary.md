# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-betboom-vs-nemiga-train-khA7BVyAiKBjWcyTrFzube/betboom-vs-nemiga-train.csv`
- round_num: `20`

## Largest probability jumps

- tick `171372`, seconds `92.50`, LSTM `0.7640`, delta `+0.4597`
- tick `171340`, seconds `92.00`, LSTM `0.3043`, delta `-0.3338`
- tick `168044`, seconds `40.50`, LSTM `0.2261`, delta `-0.2687`
- tick `167564`, seconds `33.00`, LSTM `0.2722`, delta `+0.2546`
- tick `171180`, seconds `89.50`, LSTM `0.3133`, delta `+0.2529`
- tick `171212`, seconds `90.00`, LSTM `0.4936`, delta `+0.1802`
- tick `167020`, seconds `24.50`, LSTM `0.0320`, delta `-0.1661`
- tick `166764`, seconds `20.50`, LSTM `0.3292`, delta `-0.1651`
- tick `166444`, seconds `15.50`, LSTM `0.3044`, delta `-0.1445`
- tick `171084`, seconds `88.00`, LSTM `0.1808`, delta `+0.1150`

## Top 15 local ridge features

- `lag_00__CT_defusing_count`: coefficient `0.008379`, |coef| `0.008379`
- `lag_05__CT_defusing_count`: coefficient `0.005338`, |coef| `0.005338`
- `lag_07__CT_defusing_count`: coefficient `0.003797`, |coef| `0.003797`
- `lag_09__CT_defusing_count`: coefficient `0.003568`, |coef| `0.003568`
- `lag_00__kill_diff_last_3s`: coefficient `0.003463`, |coef| `0.003463`
- `lag_15__T_place_ELECTRICALBOX`: coefficient `0.003450`, |coef| `0.003450`
- `lag_08__CT_defusing_count`: coefficient `-0.003403`, |coef| `0.003403`
- `lag_00__T_place_CONNECTOR`: coefficient `-0.003325`, |coef| `0.003325`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.003298`, |coef| `0.003298`
- `lag_01__T_place_CONNECTOR`: coefficient `-0.003274`, |coef| `0.003274`
- `lag_00__T_place_ELECTRICALBOX`: coefficient `-0.003159`, |coef| `0.003159`
- `lag_00__CT_kills_last_3s`: coefficient `0.003019`, |coef| `0.003019`
- `lag_03__CT_defusing_count`: coefficient `0.002742`, |coef| `0.002742`
- `lag_02__T_place_CONNECTOR`: coefficient `-0.002545`, |coef| `0.002545`
- `lag_00__T_burning_players`: coefficient `-0.002405`, |coef| `0.002405`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.003298` (lowers CT win probability)
- `lag_02__CT_utility_damage_last_5s`: coefficient `0.002137` (raises CT win probability)
- `lag_07__CT_B_site_active_infernos`: coefficient `0.001954` (raises CT win probability)
- `lag_06__CT_B_site_active_infernos`: coefficient `0.001868` (raises CT win probability)
- `lag_11__CT_B_site_active_infernos`: coefficient `0.001794` (raises CT win probability)
- `lag_05__CT_B_site_active_infernos`: coefficient `0.001788` (raises CT win probability)
- `lag_06__CT_A_site_active_infernos`: coefficient `0.001753` (raises CT win probability)
- `lag_02__utility_damage_diff_last_5s`: coefficient `0.001737` (raises CT win probability)
- `lag_11__CT_A_site_active_infernos`: coefficient `0.001718` (raises CT win probability)
- `lag_07__CT_A_site_active_infernos`: coefficient `0.001712` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_defusing_count`: coefficient `0.008379` (raises CT win probability)
- `lag_05__CT_defusing_count`: coefficient `0.005338` (raises CT win probability)
- `lag_07__CT_defusing_count`: coefficient `0.003797` (raises CT win probability)
- `lag_09__CT_defusing_count`: coefficient `0.003568` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003463` (raises CT win probability)
- `lag_15__T_place_ELECTRICALBOX`: coefficient `0.003450` (raises CT win probability)
- `lag_08__CT_defusing_count`: coefficient `-0.003403` (lowers CT win probability)
- `lag_00__T_place_CONNECTOR`: coefficient `-0.003325` (lowers CT win probability)
- `lag_01__T_place_CONNECTOR`: coefficient `-0.003274` (lowers CT win probability)
- `lag_00__T_place_ELECTRICALBOX`: coefficient `-0.003159` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `171372`, seconds `92.50`, LSTM delta `+0.4597`

Top all feature movements:
- `lag_05__CT_defusing_count`: contribution `+0.051741`
- `lag_09__CT_defusing_count`: contribution `+0.034591`
- `lag_08__CT_defusing_count`: contribution `+0.032984`
- `lag_00__T_flash_alpha_mean`: contribution `+0.020011`
- `lag_00__T_place_CONNECTOR`: contribution `+0.016104`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.020011`
- `lag_11__CT_B_site_active_infernos`: contribution `+0.006163`
- `lag_11__CT_A_site_active_infernos`: contribution `+0.006064`
- `lag_02__CT_utility_damage_last_5s`: contribution `+0.003764`

### tick `171340`, seconds `92.00`, LSTM delta `-0.3338`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `-0.081223`
- `lag_07__CT_defusing_count`: contribution `-0.036812`
- `lag_08__CT_defusing_count`: contribution `-0.032984`
- `lag_00__CT_duck_amount_mean`: contribution `-0.009201`
- `lag_04__CT_defusing_count`: contribution `+0.007516`

Top utility-only movements:
- `lag_10__CT_A_site_active_infernos`: contribution `-0.003060`
- `lag_10__CT_B_site_active_infernos`: contribution `-0.002758`

### tick `168044`, seconds `40.50`, LSTM delta `-0.2687`

Top all feature movements:
- `lag_15__T_place_ELECTRICALBOX`: contribution `-0.090557`
- `lag_14__CT_shots_fired_sum`: contribution `-0.016374`
- `lag_05__T1__flash_duration`: contribution `-0.010853`
- `lag_15__T_shots_fired_sum`: contribution `-0.009781`
- `lag_09__CT_kills_last_3s`: contribution `-0.009741`

Top utility-only movements:
- `lag_05__T1__flash_duration`: contribution `-0.010853`
- `lag_13__CT3__flash_duration`: contribution `-0.003703`

### tick `167564`, seconds `33.00`, LSTM delta `+0.2546`

Top all feature movements:
- `lag_00__T_place_ELECTRICALBOX`: contribution `+0.082923`
- `lag_00__CT_kills_last_3s`: contribution `+0.017434`
- `lag_00__kill_diff_last_3s`: contribution `+0.016671`
- `lag_00__CT_shots_fired_sum`: contribution `+0.015583`
- `lag_08__T_place_DUMPSTER`: contribution `+0.006845`

Top utility-only movements:
- `lag_10__CT_A_site_active_infernos`: contribution `+0.003060`
- `lag_12__CT3__flash_duration`: contribution `+0.003007`

### tick `171180`, seconds `89.50`, LSTM delta `+0.2529`

Top all feature movements:
- `lag_03__CT_defusing_count`: contribution `+0.026585`
- `lag_01__T_place_CONNECTOR`: contribution `+0.015855`
- `lag_09__T3__is_scoped`: contribution `+0.014114`
- `lag_00__CT_kills_last_3s`: contribution `+0.008717`
- `lag_00__kill_diff_last_3s`: contribution `+0.008336`

Top utility-only movements:
- `lag_05__CT_B_site_active_infernos`: contribution `+0.006144`
- `lag_05__CT_A_site_active_infernos`: contribution `+0.005769`
- `lag_02__CT_utility_damage_last_5s`: contribution `+0.004234`
- `lag_08__CT4__molly`: contribution `+0.003475`
