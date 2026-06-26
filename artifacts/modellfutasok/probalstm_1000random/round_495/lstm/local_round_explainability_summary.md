# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-mouz-bo3-Ko5VJMvyF1OsCx2TbVU9pb/vitality-vs-mouz-m1-inferno.csv`
- round_num: `10`

## Largest probability jumps

- tick `66366`, seconds `21.00`, LSTM `0.3242`, delta `-0.4404`
- tick `68894`, seconds `60.50`, LSTM `0.4548`, delta `+0.3387`
- tick `70558`, seconds `86.50`, LSTM `0.5591`, delta `+0.3269`
- tick `71966`, seconds `108.50`, LSTM `0.9179`, delta `+0.3003`
- tick `68638`, seconds `56.50`, LSTM `0.1772`, delta `-0.2412`
- tick `66398`, seconds `21.50`, LSTM `0.2131`, delta `-0.1110`
- tick `70334`, seconds `83.00`, LSTM `0.2523`, delta `-0.0774`
- tick `69918`, seconds `76.50`, LSTM `0.4055`, delta `-0.0709`
- tick `66142`, seconds `17.50`, LSTM `0.7116`, delta `+0.0661`
- tick `72190`, seconds `112.00`, LSTM `0.9636`, delta `+0.0632`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.007588`, |coef| `0.007588`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.007280`, |coef| `0.007280`
- `lag_00__CT_kills_last_3s`: coefficient `0.005636`, |coef| `0.005636`
- `lag_07__T_bomb_zone_count`: coefficient `-0.004878`, |coef| `0.004878`
- `lag_00__T_shots_fired_sum`: coefficient `-0.004683`, |coef| `0.004683`
- `lag_00__T_place_BALCONY`: coefficient `-0.004353`, |coef| `0.004353`
- `lag_08__T4__duck_amount`: coefficient `-0.004141`, |coef| `0.004141`
- `lag_09__T4__duck_amount`: coefficient `-0.004016`, |coef| `0.004016`
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.003863`, |coef| `0.003863`
- `lag_00__T_macro_B`: coefficient `-0.003863`, |coef| `0.003863`
- `lag_00__T_kills_last_3s`: coefficient `-0.003802`, |coef| `0.003802`
- `lag_01__T_duck_amount_mean`: coefficient `0.003784`, |coef| `0.003784`
- `lag_07__CT_place_PIT`: coefficient `-0.003779`, |coef| `0.003779`
- `lag_01__T_place_BALCONY`: coefficient `0.003712`, |coef| `0.003712`
- `lag_00__centroid_distance_xy`: coefficient `0.003451`, |coef| `0.003451`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.007280` (lowers CT win probability)
- `lag_15__CT_A_site_active_infernos`: coefficient `-0.002643` (lowers CT win probability)
- `lag_12__CT3__smoke`: coefficient `-0.002321` (lowers CT win probability)
- `lag_04__CT_A_site_active_infernos`: coefficient `0.002263` (raises CT win probability)
- `lag_00__T3__flash_duration`: coefficient `0.002151` (raises CT win probability)
- `lag_06__T5__flash_duration`: coefficient `-0.001950` (lowers CT win probability)
- `lag_10__CT3__flash_duration`: coefficient `0.001875` (raises CT win probability)
- `lag_07__T_utility_damage_last_5s`: coefficient `-0.001834` (lowers CT win probability)
- `lag_09__T3__flash_duration`: coefficient `-0.001788` (lowers CT win probability)
- `lag_02__T_flash_alpha_mean`: coefficient `-0.001664` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.007588` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.005636` (raises CT win probability)
- `lag_07__T_bomb_zone_count`: coefficient `-0.004878` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.004683` (lowers CT win probability)
- `lag_00__T_place_BALCONY`: coefficient `-0.004353` (lowers CT win probability)
- `lag_08__T4__duck_amount`: coefficient `-0.004141` (lowers CT win probability)
- `lag_09__T4__duck_amount`: coefficient `-0.004016` (lowers CT win probability)
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.003863` (lowers CT win probability)
- `lag_00__T_macro_B`: coefficient `-0.003863` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003802` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `66366`, seconds `21.00`, LSTM delta `-0.4404`

Top all feature movements:
- `lag_08__T_place_BALCONY`: contribution `-0.026266`
- `lag_00__T_kills_last_3s`: contribution `-0.024093`
- `lag_00__kill_diff_last_3s`: contribution `-0.018263`
- `lag_00__CT_place_TOPOFMID`: contribution `-0.017591`
- `lag_00__CT_kills_last_3s`: contribution `+0.016272`

Top utility-only movements:
- `lag_00__T3__flash_duration`: contribution `-0.014972`
- `lag_07__T_utility_damage_last_5s`: contribution `-0.013614`
- `lag_09__T3__flash_duration`: contribution `-0.012446`
- `lag_10__CT3__flash_duration`: contribution `-0.010151`
- `lag_09__T1__flash_duration`: contribution `-0.009517`

### tick `68894`, seconds `60.50`, LSTM delta `+0.3387`

Top all feature movements:
- `lag_00__T_place_BALCONY`: contribution `+0.059863`
- `lag_01__T_place_BALCONY`: contribution `+0.051046`
- `lag_00__kill_diff_last_3s`: contribution `+0.018263`
- `lag_00__CT_kills_last_3s`: contribution `+0.016272`
- `lag_08__T4__duck_amount`: contribution `+0.014229`

Top utility-only movements:
- `lag_12__CT_A_site_active_infernos`: contribution `+0.004737`

### tick `70558`, seconds `86.50`, LSTM delta `+0.3269`

Top all feature movements:
- `lag_07__T_bomb_zone_count`: contribution `+0.028395`
- `lag_00__kill_diff_last_3s`: contribution `+0.018263`
- `lag_00__CT_kills_last_3s`: contribution `+0.016272`
- `lag_03__CT_shots_fired_sum`: contribution `+0.012530`
- `lag_03__CT2__shots_fired`: contribution `+0.011139`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `71966`, seconds `108.50`, LSTM delta `+0.3003`

Top all feature movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.044170`
- `lag_01__T_duck_amount_mean`: contribution `+0.022007`
- `lag_00__kill_diff_last_3s`: contribution `+0.018263`
- `lag_00__CT_kills_last_3s`: contribution `+0.016272`
- `lag_00__T_duck_amount_mean`: contribution `+0.014722`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.044170`
- `lag_12__CT3__smoke`: contribution `+0.005134`

### tick `68638`, seconds `56.50`, LSTM delta `-0.2412`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.018263`
- `lag_00__T_shots_fired_sum`: contribution `-0.017555`
- `lag_07__CT_place_PIT`: contribution `-0.016268`
- `lag_09__T4__duck_amount`: contribution `-0.014851`
- `lag_00__T_kills_last_3s`: contribution `-0.012047`

Top utility-only movements:
- `lag_15__CT_A_site_active_infernos`: contribution `-0.009327`
- `lag_04__CT_A_site_active_infernos`: contribution `-0.007986`
- `lag_15__CT_active_infernos`: contribution `-0.003669`
