# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-big-vs-furia-bo3-8LyYppfzx0M6KmNUlhRuUi/big-vs-furia-m1-inferno.csv`
- round_num: `10`

## Largest probability jumps

- tick `77342`, seconds `112.00`, LSTM `0.4583`, delta `+0.2077`
- tick `77694`, seconds `117.50`, LSTM `0.7212`, delta `+0.1825`
- tick `76382`, seconds `97.00`, LSTM `0.6565`, delta `-0.1507`
- tick `74334`, seconds `65.00`, LSTM `0.7689`, delta `+0.1497`
- tick `77982`, seconds `122.00`, LSTM `0.9348`, delta `+0.1440`
- tick `76926`, seconds `105.50`, LSTM `0.4979`, delta `-0.1323`
- tick `77150`, seconds `109.00`, LSTM `0.2396`, delta `-0.1024`
- tick `76254`, seconds `95.00`, LSTM `0.6749`, delta `-0.0887`
- tick `76990`, seconds `106.50`, LSTM `0.3715`, delta `-0.0858`
- tick `76318`, seconds `96.00`, LSTM `0.7580`, delta `+0.0566`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003289`, |coef| `0.003289`
- `lag_00__CT_kills_last_3s`: coefficient `0.002792`, |coef| `0.002792`
- `lag_11__CT2__flash_duration`: coefficient `-0.002097`, |coef| `0.002097`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.001976`, |coef| `0.001976`
- `lag_00__T3__is_scoped`: coefficient `0.001942`, |coef| `0.001942`
- `lag_00__CT_place_BANANA`: coefficient `0.001904`, |coef| `0.001904`
- `lag_12__T3__is_scoped`: coefficient `-0.001858`, |coef| `0.001858`
- `lag_00__damage_diff_last_5s`: coefficient `0.001681`, |coef| `0.001681`
- `lag_02__CT_place_RUINS`: coefficient `0.001620`, |coef| `0.001620`
- `lag_06__T_bomb_zone_count`: coefficient `-0.001506`, |coef| `0.001506`
- `lag_00__T1__duck_amount`: coefficient `-0.001469`, |coef| `0.001469`
- `lag_07__T_kills_last_3s`: coefficient `-0.001439`, |coef| `0.001439`
- `lag_00__T_duck_amount_mean`: coefficient `-0.001405`, |coef| `0.001405`
- `lag_14__T3__is_scoped`: coefficient `-0.001400`, |coef| `0.001400`
- `lag_00__alive_diff`: coefficient `0.001337`, |coef| `0.001337`

## Top 10 utility ridge features

- `lag_11__CT2__flash_duration`: coefficient `-0.002097` (lowers CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.001976` (lowers CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `-0.001080` (lowers CT win probability)
- `lag_13__CT2__flash_duration`: coefficient `-0.001056` (lowers CT win probability)
- `lag_09__T_B_site_active_smokes`: coefficient `-0.000993` (lowers CT win probability)
- `lag_14__CT2__flash_duration`: coefficient `-0.000991` (lowers CT win probability)
- `lag_00__T2__flash`: coefficient `-0.000984` (lowers CT win probability)
- `lag_11__CT_flash_duration_sum`: coefficient `-0.000953` (lowers CT win probability)
- `lag_07__T_B_site_active_infernos`: coefficient `0.000950` (raises CT win probability)
- `lag_15__CT2__flash_duration`: coefficient `-0.000937` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003289` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002792` (raises CT win probability)
- `lag_00__T3__is_scoped`: coefficient `0.001942` (raises CT win probability)
- `lag_00__CT_place_BANANA`: coefficient `0.001904` (raises CT win probability)
- `lag_12__T3__is_scoped`: coefficient `-0.001858` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001681` (raises CT win probability)
- `lag_02__CT_place_RUINS`: coefficient `0.001620` (raises CT win probability)
- `lag_06__T_bomb_zone_count`: coefficient `-0.001506` (lowers CT win probability)
- `lag_00__T1__duck_amount`: coefficient `-0.001469` (lowers CT win probability)
- `lag_07__T_kills_last_3s`: coefficient `-0.001439` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `77342`, seconds `112.00`, LSTM delta `+0.2077`

Top all feature movements:
- `lag_11__CT2__flash_duration`: contribution `+0.016818`
- `lag_00__T3__is_scoped`: contribution `+0.012456`
- `lag_06__T_bomb_zone_count`: contribution `+0.008770`
- `lag_13__T3__is_scoped`: contribution `+0.008330`
- `lag_00__CT_kills_last_3s`: contribution `+0.008062`

Top utility-only movements:
- `lag_11__CT2__flash_duration`: contribution `+0.016818`
- `lag_11__CT_flash_duration_sum`: contribution `+0.003469`

### tick `77694`, seconds `117.50`, LSTM delta `+0.1825`

Top all feature movements:
- `lag_14__T3__is_scoped`: contribution `+0.008980`
- `lag_00__CT_kills_last_3s`: contribution `+0.008062`
- `lag_00__kill_diff_last_3s`: contribution `+0.007918`
- `lag_10__T3__is_scoped`: contribution `+0.006319`
- `lag_01__T1__duck_amount`: contribution `+0.004358`

Top utility-only movements:
- `lag_11__CT2__flash_duration`: contribution `+0.003049`

### tick `76382`, seconds `97.00`, LSTM delta `-0.1507`

Top all feature movements:
- `lag_02__CT_place_QUAD`: contribution `-0.009326`
- `lag_00__kill_diff_last_3s`: contribution `-0.007918`
- `lag_00__T1__duck_amount`: contribution `-0.005750`
- `lag_02__CT_place_RUINS`: contribution `-0.005660`
- `lag_03__CT_place_BALCONY`: contribution `-0.005376`

Top utility-only movements:
- `lag_04__T_B_site_active_infernos`: contribution `-0.002795`
- `lag_00__CT3__flash_duration`: contribution `-0.002604`

### tick `74334`, seconds `65.00`, LSTM delta `+0.1497`

Top all feature movements:
- `lag_12__T_place_BALCONY`: contribution `+0.014870`
- `lag_00__CT_kills_last_3s`: contribution `+0.008062`
- `lag_00__kill_diff_last_3s`: contribution `+0.007918`
- `lag_02__CT_place_RUINS`: contribution `+0.005660`
- `lag_00__damage_diff_last_5s`: contribution `+0.003793`

Top utility-only movements:
- `lag_00__T2__flash`: contribution `+0.002896`
- `lag_00__T2__utility_total`: contribution `+0.002050`

### tick `77982`, seconds `122.00`, LSTM delta `+0.1440`

Top all feature movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.011987`
- `lag_00__CT_place_BANANA`: contribution `+0.011273`
- `lag_14__T3__is_scoped`: contribution `+0.008980`
- `lag_00__T_duck_amount_mean`: contribution `+0.008171`
- `lag_00__CT_kills_last_3s`: contribution `+0.008062`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.011987`
- `lag_01__CT_utility_damage_last_5s`: contribution `+0.002069`
