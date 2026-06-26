# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-vitality-bo5-RwgqrXEuhDJTxQHhSIn72X/mouz-vs-vitality-m2-nuke.csv`
- round_num: `2`

## Largest probability jumps

- tick `9928`, seconds `86.50`, LSTM `0.9489`, delta `+0.0456`
- tick `8136`, seconds `58.50`, LSTM `0.9157`, delta `+0.0358`
- tick `8552`, seconds `65.00`, LSTM `0.9585`, delta `+0.0262`
- tick `8520`, seconds `64.50`, LSTM `0.9323`, delta `+0.0243`
- tick `8200`, seconds `59.50`, LSTM `0.9224`, delta `+0.0233`
- tick `8328`, seconds `61.50`, LSTM `0.9252`, delta `-0.0213`
- tick `8456`, seconds `63.50`, LSTM `0.9052`, delta `-0.0202`
- tick `7432`, seconds `47.50`, LSTM `0.8892`, delta `-0.0202`
- tick `9352`, seconds `77.50`, LSTM `0.9412`, delta `+0.0196`
- tick `9896`, seconds `86.00`, LSTM `0.9034`, delta `-0.0195`

## Top 15 local ridge features

- `lag_00__CT3__is_walking`: coefficient `-0.000790`, |coef| `0.000790`
- `lag_00__CT5__is_walking`: coefficient `-0.000718`, |coef| `0.000718`
- `lag_00__CT_place_ADMIN`: coefficient `0.000717`, |coef| `0.000717`
- `lag_00__CT_walking_count`: coefficient `-0.000660`, |coef| `0.000660`
- `lag_00__T_place_VENDING`: coefficient `-0.000590`, |coef| `0.000590`
- `lag_04__T_place_CONTROL`: coefficient `-0.000583`, |coef| `0.000583`
- `lag_00__CT_shots_fired_sum`: coefficient `0.000544`, |coef| `0.000544`
- `lag_01__CT_shots_fired_sum`: coefficient `0.000496`, |coef| `0.000496`
- `lag_00__CT_place_HELL`: coefficient `-0.000492`, |coef| `0.000492`
- `lag_00__CT_damage_last_5s`: coefficient `0.000486`, |coef| `0.000486`
- `lag_00__T4__is_walking`: coefficient `-0.000453`, |coef| `0.000453`
- `lag_05__T4__duck_amount`: coefficient `0.000434`, |coef| `0.000434`
- `lag_00__damage_diff_last_5s`: coefficient `0.000432`, |coef| `0.000432`
- `lag_00__CT_place_MINI`: coefficient `0.000421`, |coef| `0.000421`
- `lag_00__CT_duck_amount_mean`: coefficient `0.000395`, |coef| `0.000395`

## Top 10 utility ridge features

- `lag_02__T_flash_alpha_mean`: coefficient `-0.000079` (lowers CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.000066` (lowers CT win probability)
- `lag_12__CT_A_site_active_infernos`: coefficient `0.000063` (raises CT win probability)
- `lag_05__T_flash_alpha_mean`: coefficient `-0.000063` (lowers CT win probability)
- `lag_12__CT_B_site_active_infernos`: coefficient `0.000061` (raises CT win probability)
- `lag_01__CT3__utility_total`: coefficient `0.000060` (raises CT win probability)
- `lag_06__T_flash_alpha_mean`: coefficient `-0.000060` (lowers CT win probability)
- `lag_03__CT_A_site_active_smokes`: coefficient `-0.000057` (lowers CT win probability)
- `lag_01__CT3__smoke`: coefficient `0.000057` (raises CT win probability)
- `lag_01__CT1__smoke`: coefficient `0.000056` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT3__is_walking`: coefficient `-0.000790` (lowers CT win probability)
- `lag_00__CT5__is_walking`: coefficient `-0.000718` (lowers CT win probability)
- `lag_00__CT_place_ADMIN`: coefficient `0.000717` (raises CT win probability)
- `lag_00__CT_walking_count`: coefficient `-0.000660` (lowers CT win probability)
- `lag_00__T_place_VENDING`: coefficient `-0.000590` (lowers CT win probability)
- `lag_04__T_place_CONTROL`: coefficient `-0.000583` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.000544` (raises CT win probability)
- `lag_01__CT_shots_fired_sum`: coefficient `0.000496` (raises CT win probability)
- `lag_00__CT_place_HELL`: coefficient `-0.000492` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.000486` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `9928`, seconds `86.50`, LSTM delta `+0.0456`

Top all feature movements:
- `lag_00__CT_place_ADMIN`: contribution `+0.004981`
- `lag_00__CT_place_HELL`: contribution `+0.002666`
- `lag_00__CT_shots_fired_sum`: contribution `+0.001889`
- `lag_00__CT3__is_walking`: contribution `+0.001887`
- `lag_00__CT_damage_last_5s`: contribution `+0.000954`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `8136`, seconds `58.50`, LSTM delta `+0.0358`

Top all feature movements:
- `lag_15__T_place_VENDING`: contribution `+0.001915`
- `lag_00__CT3__is_walking`: contribution `+0.001887`
- `lag_08__T_place_VENDING`: contribution `+0.001506`
- `lag_05__T_place_TROPHY`: contribution `+0.001477`
- `lag_13__T_place_VENDING`: contribution `+0.001268`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `8552`, seconds `65.00`, LSTM delta `+0.0262`

Top all feature movements:
- `lag_04__T_place_CONTROL`: contribution `+0.004140`
- `lag_09__T_place_TROPHY`: contribution `+0.003834`
- `lag_09__T_place_CONTROL`: contribution `+0.002191`
- `lag_02__T_place_CONTROL`: contribution `-0.001753`
- `lag_07__T_place_CONTROL`: contribution `+0.001750`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `8520`, seconds `64.50`, LSTM delta `+0.0243`

Top all feature movements:
- `lag_04__T_place_CONTROL`: contribution `+0.008279`
- `lag_08__T_place_CONTROL`: contribution `+0.003519`
- `lag_08__T_place_TROPHY`: contribution `-0.002429`
- `lag_00__CT_place_VENTS`: contribution `+0.001560`
- `lag_10__CT_place_MINI`: contribution `+0.001332`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `8200`, seconds `59.50`, LSTM delta `+0.0233`

Top all feature movements:
- `lag_00__T_place_VENDING`: contribution `+0.002990`
- `lag_00__CT_place_MINI`: contribution `+0.002583`
- `lag_15__T_place_VENDING`: contribution `+0.001915`
- `lag_01__T_place_VENDING`: contribution `+0.001841`
- `lag_07__T_place_VENDING`: contribution `+0.001758`

Top utility-only movements:
- No utility movement among the top local contributors.
