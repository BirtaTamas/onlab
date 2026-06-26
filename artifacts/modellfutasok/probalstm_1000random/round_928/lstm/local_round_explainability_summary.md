# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-saw-bo3-tIR5RlOpBrnlpEe6MBVyNd/heroic-vs-saw-m2-train.csv`
- round_num: `17`

## Largest probability jumps

- tick `142761`, seconds `92.00`, LSTM `0.0908`, delta `-0.3101`
- tick `138217`, seconds `21.00`, LSTM `0.4040`, delta `+0.3000`
- tick `139465`, seconds `40.50`, LSTM `0.4974`, delta `-0.2488`
- tick `139401`, seconds `39.50`, LSTM `0.7060`, delta `+0.1989`
- tick `138089`, seconds `19.00`, LSTM `0.1243`, delta `-0.1714`
- tick `136905`, seconds `0.50`, LSTM `0.1887`, delta `-0.0761`
- tick `138697`, seconds `28.50`, LSTM `0.3977`, delta `-0.0578`
- tick `138409`, seconds `24.00`, LSTM `0.4597`, delta `+0.0484`
- tick `142729`, seconds `91.50`, LSTM `0.4009`, delta `-0.0452`
- tick `142793`, seconds `92.50`, LSTM `0.0483`, delta `-0.0425`

## Top 15 local ridge features

- `lag_00__CT_place_LONGDOG`: coefficient `0.006052`, |coef| `0.006052`
- `lag_03__T_place_TMAIN`: coefficient `0.004281`, |coef| `0.004281`
- `lag_02__T_place_TMAIN`: coefficient `0.004258`, |coef| `0.004258`
- `lag_00__kill_diff_last_3s`: coefficient `0.003857`, |coef| `0.003857`
- `lag_00__T_kills_last_3s`: coefficient `-0.003283`, |coef| `0.003283`
- `lag_00__damage_diff_last_5s`: coefficient `0.003175`, |coef| `0.003175`
- `lag_00__T_damage_last_5s`: coefficient `-0.002841`, |coef| `0.002841`
- `lag_15__T5__is_scoped`: coefficient `-0.002419`, |coef| `0.002419`
- `lag_03__CT2__is_scoped`: coefficient `0.002278`, |coef| `0.002278`
- `lag_01__CT_place_LONGDOG`: coefficient `0.002272`, |coef| `0.002272`
- `lag_00__T_place_LONGDOG`: coefficient `-0.002116`, |coef| `0.002116`
- `lag_00__T_shots_fired_sum`: coefficient `-0.002087`, |coef| `0.002087`
- `lag_01__T_place_TMAIN`: coefficient `0.002044`, |coef| `0.002044`
- `lag_02__T_place_BOMBSITEA`: coefficient `-0.001976`, |coef| `0.001976`
- `lag_02__T_macro_A`: coefficient `-0.001976`, |coef| `0.001976`

## Top 10 utility ridge features

- `lag_12__T_A_site_active_infernos`: coefficient `-0.001125` (lowers CT win probability)
- `lag_04__CT4__flash_duration`: coefficient `-0.001101` (lowers CT win probability)
- `lag_14__T_A_site_active_infernos`: coefficient `0.001052` (raises CT win probability)
- `lag_08__CT2__flash_duration`: coefficient `0.001027` (raises CT win probability)
- `lag_11__CT3__flash_duration`: coefficient `0.000980` (raises CT win probability)
- `lag_02__T5__flash_duration`: coefficient `-0.000963` (lowers CT win probability)
- `lag_08__CT_flash_duration_sum`: coefficient `0.000923` (raises CT win probability)
- `lag_04__T_A_site_active_smokes`: coefficient `-0.000905` (lowers CT win probability)
- `lag_08__T3__flash_duration`: coefficient `-0.000899` (lowers CT win probability)
- `lag_00__T3__smoke`: coefficient `-0.000814` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_LONGDOG`: coefficient `0.006052` (raises CT win probability)
- `lag_03__T_place_TMAIN`: coefficient `0.004281` (raises CT win probability)
- `lag_02__T_place_TMAIN`: coefficient `0.004258` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003857` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003283` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003175` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002841` (lowers CT win probability)
- `lag_15__T5__is_scoped`: coefficient `-0.002419` (lowers CT win probability)
- `lag_03__CT2__is_scoped`: coefficient `0.002278` (raises CT win probability)
- `lag_01__CT_place_LONGDOG`: coefficient `0.002272` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `142761`, seconds `92.00`, LSTM delta `-0.3101`

Top all feature movements:
- `lag_00__CT_place_LONGDOG`: contribution `-0.039475`
- `lag_03__T_place_TMAIN`: contribution `-0.016603`
- `lag_02__T_place_TMAIN`: contribution `-0.016512`
- `lag_03__CT2__is_scoped`: contribution `-0.013945`
- `lag_00__T_kills_last_3s`: contribution `-0.010400`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `138217`, seconds `21.00`, LSTM delta `+0.3000`

Top all feature movements:
- `lag_12__CT_place_ELECTRICALBOX`: contribution `+0.022787`
- `lag_03__T_place_TMAIN`: contribution `+0.016603`
- `lag_04__CT_place_ELECTRICALBOX`: contribution `+0.013023`
- `lag_05__CT_place_ELECTRICALBOX`: contribution `+0.011510`
- `lag_03__T_place_DUMPSTER`: contribution `+0.011474`

Top utility-only movements:
- `lag_11__CT3__flash_duration`: contribution `+0.006984`
- `lag_08__T3__flash_duration`: contribution `+0.005911`
- `lag_08__CT2__flash_duration`: contribution `+0.005469`
- `lag_02__T5__flash_duration`: contribution `+0.005105`

### tick `139465`, seconds `40.50`, LSTM delta `-0.2488`

Top all feature movements:
- `lag_00__CT_place_LONGDOG`: contribution `-0.039475`
- `lag_00__T_kills_last_3s`: contribution `-0.010400`
- `lag_00__kill_diff_last_3s`: contribution `-0.009284`
- `lag_00__T_shots_fired_sum`: contribution `-0.007824`
- `lag_02__T_place_LONGDOG`: contribution `-0.007749`

Top utility-only movements:
- `lag_14__T_A_site_active_infernos`: contribution `-0.003130`

### tick `139401`, seconds `39.50`, LSTM delta `+0.1989`

Top all feature movements:
- `lag_15__T5__is_scoped`: contribution `+0.011536`
- `lag_00__T_place_LONGDOG`: contribution `+0.009845`
- `lag_00__kill_diff_last_3s`: contribution `+0.009284`
- `lag_11__CT2__is_scoped`: contribution `+0.008142`
- `lag_10__CT_place_CONNECTOR`: contribution `+0.005712`

Top utility-only movements:
- `lag_12__T_A_site_active_infernos`: contribution `+0.003350`

### tick `138089`, seconds `19.00`, LSTM delta `-0.1714`

Top all feature movements:
- `lag_13__CT_place_ELECTRICALBOX`: contribution `-0.013140`
- `lag_00__T_kills_last_3s`: contribution `-0.010400`
- `lag_00__kill_diff_last_3s`: contribution `-0.009284`
- `lag_07__CT_place_ELECTRICALBOX`: contribution `-0.008870`
- `lag_00__T_shots_fired_sum`: contribution `-0.007824`

Top utility-only movements:
- `lag_04__CT4__flash_duration`: contribution `-0.005563`
- `lag_04__T3__flash_duration`: contribution `-0.005019`
- `lag_04__CT_flash_duration_sum`: contribution `-0.003689`
- `lag_12__T_A_site_active_infernos`: contribution `-0.003350`
- `lag_07__CT3__flash_duration`: contribution `-0.002728`
