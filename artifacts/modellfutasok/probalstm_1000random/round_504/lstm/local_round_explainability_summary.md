# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-passion-ua-vs-spirit-bo3-WimU0hRkNcqhh3KAjCozBx/passion-ua-vs-spirit-m3-ancient.csv`
- round_num: `9`

## Largest probability jumps

- tick `72674`, seconds `38.00`, LSTM `0.2110`, delta `-0.2767`
- tick `76194`, seconds `93.00`, LSTM `0.2602`, delta `-0.2654`
- tick `75810`, seconds `87.00`, LSTM `0.6707`, delta `+0.2032`
- tick `74306`, seconds `63.50`, LSTM `0.2713`, delta `+0.1981`
- tick `71362`, seconds `17.50`, LSTM `0.4977`, delta `-0.1289`
- tick `72706`, seconds `38.50`, LSTM `0.1239`, delta `-0.0871`
- tick `75746`, seconds `86.00`, LSTM `0.4610`, delta `+0.0719`
- tick `75970`, seconds `89.50`, LSTM `0.5986`, delta `-0.0661`
- tick `76226`, seconds `93.50`, LSTM `0.1959`, delta `-0.0643`
- tick `75362`, seconds `80.00`, LSTM `0.2432`, delta `-0.0600`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004206`, |coef| `0.004206`
- `lag_00__CT_place_UNKNOWN`: coefficient `0.003451`, |coef| `0.003451`
- `lag_00__closest_enemy_dist_diff`: coefficient `0.003322`, |coef| `0.003322`
- `lag_03__CT_place_WATER`: coefficient `-0.003271`, |coef| `0.003271`
- `lag_00__CT_place_WATER`: coefficient `0.003256`, |coef| `0.003256`
- `lag_11__CT_shots_fired_sum`: coefficient `0.003186`, |coef| `0.003186`
- `lag_12__T_bomb_zone_count`: coefficient `0.003121`, |coef| `0.003121`
- `lag_00__T_kills_last_3s`: coefficient `-0.003040`, |coef| `0.003040`
- `lag_14__T_bomb_zone_count`: coefficient `0.003028`, |coef| `0.003028`
- `lag_00__T_bomb_zone_count`: coefficient `-0.002772`, |coef| `0.002772`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002768`, |coef| `0.002768`
- `lag_11__CT4__shots_fired`: coefficient `0.002592`, |coef| `0.002592`
- `lag_13__T2__duck_amount`: coefficient `-0.002426`, |coef| `0.002426`
- `lag_03__CT_place_RUINS`: coefficient `0.002356`, |coef| `0.002356`
- `lag_00__CT_kills_last_3s`: coefficient `0.002275`, |coef| `0.002275`

## Top 10 utility ridge features

- `lag_13__T_B_site_active_infernos`: coefficient `-0.001616` (lowers CT win probability)
- `lag_04__CT_B_site_active_infernos`: coefficient `0.001572` (raises CT win probability)
- `lag_07__CT_B_site_active_infernos`: coefficient `0.001535` (raises CT win probability)
- `lag_13__T_active_infernos`: coefficient `-0.001472` (lowers CT win probability)
- `lag_00__CT5__flash`: coefficient `0.001466` (raises CT win probability)
- `lag_14__T_B_site_active_smokes`: coefficient `0.001392` (raises CT win probability)
- `lag_00__CT4__smoke`: coefficient `0.001321` (raises CT win probability)
- `lag_06__CT_B_site_active_infernos`: coefficient `0.001310` (raises CT win probability)
- `lag_02__CT_B_site_active_smokes`: coefficient `0.001252` (raises CT win probability)
- `lag_00__CT_flash_inv`: coefficient `0.001210` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004206` (raises CT win probability)
- `lag_00__CT_place_UNKNOWN`: coefficient `0.003451` (raises CT win probability)
- `lag_00__closest_enemy_dist_diff`: coefficient `0.003322` (raises CT win probability)
- `lag_03__CT_place_WATER`: coefficient `-0.003271` (lowers CT win probability)
- `lag_00__CT_place_WATER`: coefficient `0.003256` (raises CT win probability)
- `lag_11__CT_shots_fired_sum`: coefficient `0.003186` (raises CT win probability)
- `lag_12__T_bomb_zone_count`: coefficient `0.003121` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003040` (lowers CT win probability)
- `lag_14__T_bomb_zone_count`: coefficient `0.003028` (raises CT win probability)
- `lag_00__T_bomb_zone_count`: coefficient `-0.002772` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `72674`, seconds `38.00`, LSTM delta `-0.2767`

Top all feature movements:
- `lag_03__CT_place_WATER`: contribution `-0.019877`
- `lag_00__CT_place_WATER`: contribution `-0.019785`
- `lag_00__T_shots_fired_sum`: contribution `-0.011452`
- `lag_00__kill_diff_last_3s`: contribution `-0.010124`
- `lag_00__T_kills_last_3s`: contribution `-0.009631`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `76194`, seconds `93.00`, LSTM delta `-0.2654`

Top all feature movements:
- `lag_11__CT_shots_fired_sum`: contribution `-0.033201`
- `lag_11__CT4__shots_fired`: contribution `-0.020946`
- `lag_12__T_bomb_zone_count`: contribution `-0.018166`
- `lag_00__kill_diff_last_3s`: contribution `-0.010124`
- `lag_00__T_kills_last_3s`: contribution `-0.009631`

Top utility-only movements:
- `lag_07__CT_B_site_active_infernos`: contribution `-0.005273`

### tick `75810`, seconds `87.00`, LSTM delta `+0.2032`

Top all feature movements:
- `lag_14__T_bomb_zone_count`: contribution `+0.017628`
- `lag_00__T_bomb_zone_count`: contribution `+0.016139`
- `lag_00__kill_diff_last_3s`: contribution `+0.010124`
- `lag_00__CT_shots_fired_sum`: contribution `+0.009614`
- `lag_05__T3__duck_amount`: contribution `+0.007713`

Top utility-only movements:
- `lag_13__T_B_site_active_infernos`: contribution `+0.004570`
- `lag_06__CT_B_site_active_infernos`: contribution `+0.004499`

### tick `74306`, seconds `63.50`, LSTM delta `+0.1981`

Top all feature movements:
- `lag_03__T_place_TUNNEL`: contribution `+0.012407`
- `lag_00__kill_diff_last_3s`: contribution `+0.010124`
- `lag_01__T_place_WATER`: contribution `+0.010116`
- `lag_05__T_place_TUNNEL`: contribution `+0.008432`
- `lag_05__T3__duck_amount`: contribution `+0.007713`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `71362`, seconds `17.50`, LSTM delta `-0.1289`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.010124`
- `lag_00__T_kills_last_3s`: contribution `-0.009631`
- `lag_01__CT_shots_fired_sum`: contribution `-0.009222`
- `lag_04__CT_B_site_active_infernos`: contribution `-0.005400`
- `lag_06__CT_B_site_active_infernos`: contribution `+0.004499`

Top utility-only movements:
- `lag_04__CT_B_site_active_infernos`: contribution `-0.005400`
- `lag_06__CT_B_site_active_infernos`: contribution `+0.004499`
- `lag_04__T4__flash_duration`: contribution `-0.003344`
- `lag_00__CT5__flash_duration`: contribution `-0.003199`
- `lag_13__T_active_infernos`: contribution `-0.003065`
