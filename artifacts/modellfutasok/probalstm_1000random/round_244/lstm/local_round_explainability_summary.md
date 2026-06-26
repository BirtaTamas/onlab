# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-lynn-vision-bo3-6KVULP2-Gxo12lI67V9ZfV/chinggis-warriors-vs-lynn-vision-m3-ancient.csv`
- round_num: `20`

## Largest probability jumps

- tick `176255`, seconds `95.50`, LSTM `0.8112`, delta `+0.3059`
- tick `176191`, seconds `94.50`, LSTM `0.4727`, delta `-0.2839`
- tick `176799`, seconds `104.00`, LSTM `0.6147`, delta `+0.2766`
- tick `176351`, seconds `97.00`, LSTM `0.5286`, delta `-0.2651`
- tick `176095`, seconds `93.00`, LSTM `0.9081`, delta `+0.1260`
- tick `176159`, seconds `94.00`, LSTM `0.7566`, delta `-0.1257`
- tick `177375`, seconds `113.00`, LSTM `0.8742`, delta `+0.1246`
- tick `175775`, seconds `88.00`, LSTM `0.7211`, delta `-0.0989`
- tick `175455`, seconds `83.00`, LSTM `0.6193`, delta `+0.0984`
- tick `174143`, seconds `62.50`, LSTM `0.6553`, delta `+0.0816`

## Top 15 local ridge features

- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.004274`, |coef| `0.004274`
- `lag_00__kill_diff_last_3s`: coefficient `0.003656`, |coef| `0.003656`
- `lag_07__CT_flashes_last_5s`: coefficient `0.003412`, |coef| `0.003412`
- `lag_01__T_shots_fired_sum`: coefficient `-0.003211`, |coef| `0.003211`
- `lag_02__T_duck_amount_mean`: coefficient `-0.003106`, |coef| `0.003106`
- `lag_00__CT_defusing_count`: coefficient `0.003006`, |coef| `0.003006`
- `lag_02__T5__duck_amount`: coefficient `-0.002854`, |coef| `0.002854`
- `lag_00__CT_kills_last_3s`: coefficient `0.002707`, |coef| `0.002707`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.002365`, |coef| `0.002365`
- `lag_04__T_place_CTSPAWN`: coefficient `0.002308`, |coef| `0.002308`
- `lag_00__damage_diff_last_5s`: coefficient `0.002260`, |coef| `0.002260`
- `lag_02__T_bomb_zone_count`: coefficient `-0.002250`, |coef| `0.002250`
- `lag_10__CT_place_TSIDEUPPER`: coefficient `-0.002190`, |coef| `0.002190`
- `lag_11__T_bomb_zone_count`: coefficient `0.002129`, |coef| `0.002129`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001910`, |coef| `0.001910`

## Top 10 utility ridge features

- `lag_07__CT_flashes_last_5s`: coefficient `0.003412` (raises CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.002365` (lowers CT win probability)
- `lag_00__CT_flashes_last_5s`: coefficient `0.001688` (raises CT win probability)
- `lag_08__CT_flashes_last_5s`: coefficient `0.001457` (raises CT win probability)
- `lag_14__CT4__flash`: coefficient `-0.001409` (lowers CT win probability)
- `lag_11__T_flash_alpha_mean`: coefficient `-0.001123` (lowers CT win probability)
- `lag_00__CT4__flash`: coefficient `0.000945` (raises CT win probability)
- `lag_09__T_A_site_active_infernos`: coefficient `-0.000860` (lowers CT win probability)
- `lag_09__T_B_site_active_infernos`: coefficient `-0.000841` (lowers CT win probability)
- `lag_11__T_A_site_active_infernos`: coefficient `0.000830` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.004274` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003656` (raises CT win probability)
- `lag_01__T_shots_fired_sum`: coefficient `-0.003211` (lowers CT win probability)
- `lag_02__T_duck_amount_mean`: coefficient `-0.003106` (lowers CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.003006` (raises CT win probability)
- `lag_02__T5__duck_amount`: coefficient `-0.002854` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002707` (raises CT win probability)
- `lag_04__T_place_CTSPAWN`: coefficient `0.002308` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002260` (raises CT win probability)
- `lag_02__T_bomb_zone_count`: coefficient `-0.002250` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `176255`, seconds `95.50`, LSTM delta `+0.3059`

Top all feature movements:
- `lag_01__T_shots_fired_sum`: contribution `+0.016853`
- `lag_02__T5__duck_amount`: contribution `+0.010838`
- `lag_03__T_place_SIDEHALL`: contribution `+0.010433`
- `lag_07__CT_place_TSIDEUPPER`: contribution `+0.009821`
- `lag_02__T5__shots_fired`: contribution `+0.009725`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `176191`, seconds `94.50`, LSTM delta `-0.2839`

Top all feature movements:
- `lag_00__CT_place_TSIDEUPPER`: contribution `-0.032126`
- `lag_01__T_shots_fired_sum`: contribution `-0.014445`
- `lag_02__T5__duck_amount`: contribution `-0.010838`
- `lag_13__CT_place_TSIDEUPPER`: contribution `-0.009107`
- `lag_02__T_duck_amount_mean`: contribution `-0.009031`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `176799`, seconds `104.00`, LSTM delta `+0.2766`

Top all feature movements:
- `lag_02__T_duck_amount_mean`: contribution `+0.018062`
- `lag_00__T_flash_alpha_mean`: contribution `+0.014348`
- `lag_02__T_bomb_zone_count`: contribution `+0.013100`
- `lag_11__T_bomb_zone_count`: contribution `+0.012391`
- `lag_02__T5__duck_amount`: contribution `+0.010838`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.014348`
- `lag_14__CT4__flash`: contribution `+0.004884`

### tick `176351`, seconds `97.00`, LSTM delta `-0.2651`

Top all feature movements:
- `lag_10__CT_place_TSIDEUPPER`: contribution `-0.016460`
- `lag_04__T_place_CTSPAWN`: contribution `-0.011010`
- `lag_06__T_place_SIDEHALL`: contribution `-0.009904`
- `lag_04__T_shots_fired_sum`: contribution `-0.009839`
- `lag_05__T5__shots_fired`: contribution `-0.007662`

Top utility-only movements:
- `lag_00__CT4__flash`: contribution `-0.003278`

### tick `176095`, seconds `93.00`, LSTM delta `+0.1260`

Top all feature movements:
- `lag_10__CT_place_TSIDEUPPER`: contribution `+0.016460`
- `lag_00__kill_diff_last_3s`: contribution `+0.008801`
- `lag_00__CT_kills_last_3s`: contribution `+0.007815`
- `lag_00__CT_shots_fired_sum`: contribution `+0.005308`
- `lag_10__CT_place_SIDEENTRANCE`: contribution `+0.004093`

Top utility-only movements:
- `lag_06__T_A_site_active_infernos`: contribution `+0.001678`
