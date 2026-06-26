# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-eternal-fire-vs-flyquest-bo3-bOv4otMGdpLsO1VdhzI_AV/eternal-fire-vs-flyquest-m2-nuke.csv`
- round_num: `7`

## Largest probability jumps

- tick `44209`, seconds `57.00`, LSTM `0.2595`, delta `-0.1899`
- tick `43889`, seconds `52.00`, LSTM `0.5057`, delta `+0.1616`
- tick `42161`, seconds `25.00`, LSTM `0.3926`, delta `-0.0939`
- tick `43857`, seconds `51.50`, LSTM `0.3441`, delta `+0.0803`
- tick `43825`, seconds `51.00`, LSTM `0.2637`, delta `+0.0764`
- tick `44273`, seconds `58.00`, LSTM `0.1457`, delta `-0.0759`
- tick `42193`, seconds `25.50`, LSTM `0.3267`, delta `-0.0659`
- tick `44369`, seconds `59.50`, LSTM `0.0472`, delta `-0.0650`
- tick `43505`, seconds `46.00`, LSTM `0.2647`, delta `-0.0621`
- tick `44017`, seconds `54.00`, LSTM `0.4586`, delta `-0.0447`

## Top 15 local ridge features

- `lag_15__T_place_MINI`: coefficient `-0.002228`, |coef| `0.002228`
- `lag_00__CT_place_CONTROL`: coefficient `-0.002053`, |coef| `0.002053`
- `lag_02__CT_place_TROPHY`: coefficient `0.001940`, |coef| `0.001940`
- `lag_12__CT_place_CONTROL`: coefficient `0.001809`, |coef| `0.001809`
- `lag_02__CT_place_CONTROL`: coefficient `-0.001689`, |coef| `0.001689`
- `lag_15__CT_place_CONTROL`: coefficient `0.001447`, |coef| `0.001447`
- `lag_14__T_place_MINI`: coefficient `-0.001298`, |coef| `0.001298`
- `lag_00__damage_diff_last_5s`: coefficient `0.001262`, |coef| `0.001262`
- `lag_12__CT_place_TROPHY`: coefficient `-0.001255`, |coef| `0.001255`
- `lag_02__CT_utility_damage_last_5s`: coefficient `0.001210`, |coef| `0.001210`
- `lag_06__CT3__flash_duration`: coefficient `0.001190`, |coef| `0.001190`
- `lag_00__kill_diff_last_3s`: coefficient `0.001161`, |coef| `0.001161`
- `lag_01__CT_place_TROPHY`: coefficient `0.001119`, |coef| `0.001119`
- `lag_08__CT_place_VENDING`: coefficient `-0.001088`, |coef| `0.001088`
- `lag_07__CT_place_CONTROL`: coefficient `-0.001071`, |coef| `0.001071`

## Top 10 utility ridge features

- `lag_02__CT_utility_damage_last_5s`: coefficient `0.001210` (raises CT win probability)
- `lag_06__CT3__flash_duration`: coefficient `0.001190` (raises CT win probability)
- `lag_02__utility_damage_diff_last_5s`: coefficient `0.000969` (raises CT win probability)
- `lag_14__T_A_site_active_smokes`: coefficient `-0.000839` (lowers CT win probability)
- `lag_00__CT4__utility_total`: coefficient `0.000756` (raises CT win probability)
- `lag_00__CT4__smoke`: coefficient `0.000756` (raises CT win probability)
- `lag_13__T_A_site_active_smokes`: coefficient `-0.000750` (lowers CT win probability)
- `lag_07__CT4__flash_duration`: coefficient `-0.000684` (lowers CT win probability)
- `lag_04__CT_B_site_active_smokes`: coefficient `-0.000677` (lowers CT win probability)
- `lag_06__CT4__flash_duration`: coefficient `-0.000665` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_15__T_place_MINI`: coefficient `-0.002228` (lowers CT win probability)
- `lag_00__CT_place_CONTROL`: coefficient `-0.002053` (lowers CT win probability)
- `lag_02__CT_place_TROPHY`: coefficient `0.001940` (raises CT win probability)
- `lag_12__CT_place_CONTROL`: coefficient `0.001809` (raises CT win probability)
- `lag_02__CT_place_CONTROL`: coefficient `-0.001689` (lowers CT win probability)
- `lag_15__CT_place_CONTROL`: coefficient `0.001447` (raises CT win probability)
- `lag_14__T_place_MINI`: coefficient `-0.001298` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001262` (raises CT win probability)
- `lag_12__CT_place_TROPHY`: coefficient `-0.001255` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001161` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `44209`, seconds `57.00`, LSTM delta `-0.1899`

Top all feature movements:
- `lag_15__T_place_MINI`: contribution `-0.030998`
- `lag_12__CT_place_CONTROL`: contribution `-0.018772`
- `lag_08__CT_place_VENDING`: contribution `-0.018645`
- `lag_12__CT_place_TROPHY`: contribution `-0.018537`
- `lag_05__T_place_MINI`: contribution `+0.008765`

Top utility-only movements:
- `lag_02__CT_utility_damage_last_5s`: contribution `-0.005993`
- `lag_02__utility_damage_diff_last_5s`: contribution `-0.003935`
- `lag_06__CT3__flash_duration`: contribution `-0.003811`
- `lag_09__T_A_site_active_infernos`: contribution `-0.003236`
- `lag_09__T_B_site_active_infernos`: contribution `-0.002902`

### tick `43889`, seconds `52.00`, LSTM delta `+0.1616`

Top all feature movements:
- `lag_02__CT_place_TROPHY`: contribution `+0.028656`
- `lag_12__CT_place_CONTROL`: contribution `+0.018772`
- `lag_02__CT_place_CONTROL`: contribution `+0.017530`
- `lag_15__CT_place_CONTROL`: contribution `+0.015022`
- `lag_05__T_place_MINI`: contribution `+0.008765`

Top utility-only movements:
- `lag_06__CT3__flash_duration`: contribution `+0.006557`
- `lag_02__CT_utility_damage_last_5s`: contribution `+0.005993`
- `lag_02__utility_damage_diff_last_5s`: contribution `+0.003935`

### tick `42161`, seconds `25.00`, LSTM delta `-0.0939`

Top all feature movements:
- `lag_08__CT_place_HEAVEN`: contribution `-0.004062`
- `lag_05__CT_place_HELL`: contribution `-0.004051`
- `lag_12__CT_place_HEAVEN`: contribution `-0.003873`
- `lag_00__T_kills_last_3s`: contribution `-0.003270`
- `lag_01__CT2__duck_amount`: contribution `-0.003134`

Top utility-only movements:
- `lag_06__CT4__flash_duration`: contribution `-0.002132`
- `lag_01__CT4__flash_duration`: contribution `-0.002131`
- `lag_00__CT4__smoke`: contribution `-0.001650`

### tick `43857`, seconds `51.50`, LSTM delta `+0.0803`

Top all feature movements:
- `lag_01__CT_place_TROPHY`: contribution `+0.016526`
- `lag_04__T_place_MINI`: contribution `+0.011255`
- `lag_14__CT_place_CONTROL`: contribution `+0.009150`
- `lag_01__CT_place_CONTROL`: contribution `+0.008872`
- `lag_01__T_place_SQUEAKY`: contribution `+0.003607`

Top utility-only movements:
- `lag_05__CT3__flash_duration`: contribution `+0.002515`
- `lag_01__CT_utility_damage_last_5s`: contribution `+0.001834`
- `lag_01__utility_damage_diff_last_5s`: contribution `+0.001517`

### tick `43825`, seconds `51.00`, LSTM delta `+0.0764`

Top all feature movements:
- `lag_00__CT_place_CONTROL`: contribution `+0.021314`
- `lag_03__T_place_MINI`: contribution `+0.010716`
- `lag_00__CT_place_TROPHY`: contribution `+0.010493`
- `lag_13__CT_place_CONTROL`: contribution `+0.005971`
- `lag_00__T_place_SQUEAKY`: contribution `+0.002905`

Top utility-only movements:
- `lag_04__CT3__flash_duration`: contribution `+0.002489`
- `lag_01__CT3__flash_duration`: contribution `+0.001482`
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.001439`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.001411`
