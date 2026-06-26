# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-gamerlegion-vs-tyloo-bo3-0g9mXt3FIxC8XzjXNUjRL7/gamerlegion-vs-tyloo-m1-ancient-p3.csv`
- round_num: `4`

## Largest probability jumps

- tick `35289`, seconds `63.50`, LSTM `0.7216`, delta `+0.4286`
- tick `35417`, seconds `65.50`, LSTM `0.2024`, delta `-0.3805`
- tick `34489`, seconds `51.00`, LSTM `0.3490`, delta `-0.3372`
- tick `35193`, seconds `62.00`, LSTM `0.3949`, delta `+0.2648`
- tick `35321`, seconds `64.00`, LSTM `0.6047`, delta `-0.1169`
- tick `35257`, seconds `63.00`, LSTM `0.2930`, delta `-0.0909`
- tick `34329`, seconds `48.50`, LSTM `0.7327`, delta `+0.0866`
- tick `34585`, seconds `52.50`, LSTM `0.4361`, delta `+0.0687`
- tick `34841`, seconds `56.50`, LSTM `0.2586`, delta `-0.0634`
- tick `34521`, seconds `51.50`, LSTM `0.4061`, delta `+0.0570`

## Top 15 local ridge features

- `lag_15__T5__flash_duration`: coefficient `0.003508`, |coef| `0.003508`
- `lag_00__kill_diff_last_3s`: coefficient `0.002649`, |coef| `0.002649`
- `lag_00__damage_diff_last_5s`: coefficient `0.002506`, |coef| `0.002506`
- `lag_02__T_shots_fired_sum`: coefficient `-0.002288`, |coef| `0.002288`
- `lag_00__T_shots_fired_sum`: coefficient `-0.002258`, |coef| `0.002258`
- `lag_13__T_place_SIDEHALL`: coefficient `0.002199`, |coef| `0.002199`
- `lag_08__T_place_SIDEHALL`: coefficient `-0.001956`, |coef| `0.001956`
- `lag_03__CT3__is_scoped`: coefficient `0.001956`, |coef| `0.001956`
- `lag_00__T_kills_last_3s`: coefficient `-0.001935`, |coef| `0.001935`
- `lag_12__CT4__flash_duration`: coefficient `-0.001931`, |coef| `0.001931`
- `lag_08__T1__flash_duration`: coefficient `0.001919`, |coef| `0.001919`
- `lag_11__T_place_SIDEHALL`: coefficient `-0.001885`, |coef| `0.001885`
- `lag_11__T5__flash_duration`: coefficient `-0.001859`, |coef| `0.001859`
- `lag_12__CT5__flash_duration`: coefficient `-0.001809`, |coef| `0.001809`
- `lag_08__CT4__flash_duration`: coefficient `0.001775`, |coef| `0.001775`

## Top 10 utility ridge features

- `lag_15__T5__flash_duration`: coefficient `0.003508` (raises CT win probability)
- `lag_12__CT4__flash_duration`: coefficient `-0.001931` (lowers CT win probability)
- `lag_08__T1__flash_duration`: coefficient `0.001919` (raises CT win probability)
- `lag_11__T5__flash_duration`: coefficient `-0.001859` (lowers CT win probability)
- `lag_12__CT5__flash_duration`: coefficient `-0.001809` (lowers CT win probability)
- `lag_08__CT4__flash_duration`: coefficient `0.001775` (raises CT win probability)
- `lag_05__CT4__flash_duration`: coefficient `0.001712` (raises CT win probability)
- `lag_05__T1__flash_duration`: coefficient `0.001660` (raises CT win probability)
- `lag_12__T1__flash_duration`: coefficient `-0.001634` (lowers CT win probability)
- `lag_03__T1__flash_duration`: coefficient `-0.001590` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002649` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002506` (raises CT win probability)
- `lag_02__T_shots_fired_sum`: coefficient `-0.002288` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.002258` (lowers CT win probability)
- `lag_13__T_place_SIDEHALL`: coefficient `0.002199` (raises CT win probability)
- `lag_08__T_place_SIDEHALL`: coefficient `-0.001956` (lowers CT win probability)
- `lag_03__CT3__is_scoped`: coefficient `0.001956` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001935` (lowers CT win probability)
- `lag_11__T_place_SIDEHALL`: coefficient `-0.001885` (lowers CT win probability)
- `lag_07__T1__has_bomb`: coefficient `0.001675` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `35289`, seconds `63.50`, LSTM delta `+0.4286`

Top all feature movements:
- `lag_15__T5__flash_duration`: contribution `+0.021369`
- `lag_08__CT4__flash_duration`: contribution `+0.014660`
- `lag_13__T_place_SIDEHALL`: contribution `+0.014255`
- `lag_11__T_place_SIDEHALL`: contribution `+0.012217`
- `lag_08__T1__flash_duration`: contribution `+0.011442`

Top utility-only movements:
- `lag_15__T5__flash_duration`: contribution `+0.021369`
- `lag_08__CT4__flash_duration`: contribution `+0.014660`
- `lag_08__T1__flash_duration`: contribution `+0.011442`
- `lag_11__T5__flash_duration`: contribution `+0.011323`
- `lag_03__T1__flash_duration`: contribution `+0.009476`

### tick `35417`, seconds `65.50`, LSTM delta `-0.3805`

Top all feature movements:
- `lag_15__T5__flash_duration`: contribution `-0.021369`
- `lag_12__CT4__flash_duration`: contribution `-0.015949`
- `lag_15__T_place_SIDEHALL`: contribution `-0.010760`
- `lag_12__T1__flash_duration`: contribution `-0.009741`
- `lag_02__T_duck_amount_mean`: contribution `-0.009505`

Top utility-only movements:
- `lag_15__T5__flash_duration`: contribution `-0.021369`
- `lag_12__CT4__flash_duration`: contribution `-0.015949`
- `lag_12__T1__flash_duration`: contribution `-0.009741`
- `lag_12__CT5__flash_duration`: contribution `-0.009307`
- `lag_07__T1__flash_duration`: contribution `-0.007374`

### tick `34489`, seconds `51.00`, LSTM delta `-0.3372`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.018621`
- `lag_02__T_shots_fired_sum`: contribution `-0.013722`
- `lag_01__T_shots_fired_sum`: contribution `-0.008722`
- `lag_02__CT5__flash_duration`: contribution `-0.007126`
- `lag_00__kill_diff_last_3s`: contribution `-0.006377`

Top utility-only movements:
- `lag_02__CT5__flash_duration`: contribution `-0.007126`
- `lag_05__CT_A_site_active_infernos`: contribution `-0.004484`

### tick `35193`, seconds `62.00`, LSTM delta `+0.2648`

Top all feature movements:
- `lag_05__CT4__flash_duration`: contribution `+0.014143`
- `lag_08__T_place_SIDEHALL`: contribution `+0.012676`
- `lag_05__T1__flash_duration`: contribution `+0.009894`
- `lag_10__T_place_SIDEHALL`: contribution `+0.008760`
- `lag_14__CT5__flash_duration`: contribution `+0.008259`

Top utility-only movements:
- `lag_05__CT4__flash_duration`: contribution `+0.014143`
- `lag_05__T1__flash_duration`: contribution `+0.009894`
- `lag_14__CT5__flash_duration`: contribution `+0.008259`
- `lag_00__T1__flash_duration`: contribution `+0.007555`
- `lag_05__T5__flash_duration`: contribution `+0.005895`

### tick `35321`, seconds `64.00`, LSTM delta `-0.1169`

Top all feature movements:
- `lag_00__CT3__is_scoped`: contribution `-0.005196`
- `lag_12__CT4__flash_duration`: contribution `+0.005164`
- `lag_04__T1__flash_duration`: contribution `-0.005138`
- `lag_09__CT4__flash_duration`: contribution `-0.004385`
- `lag_09__CT5__flash_duration`: contribution `-0.004317`

Top utility-only movements:
- `lag_12__CT4__flash_duration`: contribution `+0.005164`
- `lag_04__T1__flash_duration`: contribution `-0.005138`
- `lag_09__CT4__flash_duration`: contribution `-0.004385`
- `lag_09__CT5__flash_duration`: contribution `-0.004317`
- `lag_09__T5__flash_duration`: contribution `-0.002827`
