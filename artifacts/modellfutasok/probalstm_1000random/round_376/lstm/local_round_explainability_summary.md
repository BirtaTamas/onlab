# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m1-inferno.csv`
- round_num: `4`

## Largest probability jumps

- tick `35686`, seconds `139.50`, LSTM `0.8626`, delta `+0.2075`
- tick `35622`, seconds `138.50`, LSTM `0.6303`, delta `-0.1651`
- tick `34374`, seconds `119.00`, LSTM `0.7532`, delta `+0.1208`
- tick `34054`, seconds `114.00`, LSTM `0.5232`, delta `+0.1196`
- tick `36262`, seconds `148.50`, LSTM `0.8077`, delta `-0.1174`
- tick `36326`, seconds `149.50`, LSTM `0.6556`, delta `-0.1068`
- tick `33926`, seconds `112.00`, LSTM `0.4988`, delta `+0.0915`
- tick `36070`, seconds `145.50`, LSTM `0.9124`, delta `+0.0899`
- tick `34022`, seconds `113.50`, LSTM `0.4036`, delta `-0.0853`
- tick `33702`, seconds `108.50`, LSTM `0.4245`, delta `-0.0764`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003103`, |coef| `0.003103`
- `lag_00__T_shots_fired_sum`: coefficient `-0.003010`, |coef| `0.003010`
- `lag_00__CT_defusing_count`: coefficient `0.002906`, |coef| `0.002906`
- `lag_00__T_kills_last_3s`: coefficient `-0.002879`, |coef| `0.002879`
- `lag_13__CT4__duck_amount`: coefficient `-0.002629`, |coef| `0.002629`
- `lag_06__CT_defusing_count`: coefficient `-0.002293`, |coef| `0.002293`
- `lag_15__CT_defusing_count`: coefficient `-0.001813`, |coef| `0.001813`
- `lag_00__damage_diff_last_5s`: coefficient `0.001803`, |coef| `0.001803`
- `lag_00__T_flash_alpha_mean`: coefficient `0.001778`, |coef| `0.001778`
- `lag_00__CT_velocity_mean`: coefficient `-0.001701`, |coef| `0.001701`
- `lag_01__T5__shots_fired`: coefficient `0.001590`, |coef| `0.001590`
- `lag_10__T5__duck_amount`: coefficient `-0.001588`, |coef| `0.001588`
- `lag_01__CT_defusing_count`: coefficient `0.001562`, |coef| `0.001562`
- `lag_14__T5__is_walking`: coefficient `0.001506`, |coef| `0.001506`
- `lag_14__CT_defusing_count`: coefficient `-0.001501`, |coef| `0.001501`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `0.001778` (raises CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `0.001332` (raises CT win probability)
- `lag_07__T_flash_alpha_mean`: coefficient `0.001139` (raises CT win probability)
- `lag_09__T_flash_alpha_mean`: coefficient `0.001050` (raises CT win probability)
- `lag_08__T_flash_alpha_mean`: coefficient `0.001007` (raises CT win probability)
- `lag_06__T_flash_alpha_mean`: coefficient `0.000942` (raises CT win probability)
- `lag_02__T_flash_alpha_mean`: coefficient `0.000731` (raises CT win probability)
- `lag_05__T_flash_alpha_mean`: coefficient `0.000646` (raises CT win probability)
- `lag_07__T_B_site_active_smokes`: coefficient `0.000613` (raises CT win probability)
- `lag_09__T_B_site_active_smokes`: coefficient `0.000542` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003103` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.003010` (lowers CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.002906` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002879` (lowers CT win probability)
- `lag_13__CT4__duck_amount`: coefficient `-0.002629` (lowers CT win probability)
- `lag_06__CT_defusing_count`: coefficient `-0.002293` (lowers CT win probability)
- `lag_15__CT_defusing_count`: coefficient `-0.001813` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001803` (raises CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.001701` (lowers CT win probability)
- `lag_01__T5__shots_fired`: coefficient `0.001590` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `35686`, seconds `139.50`, LSTM delta `+0.2075`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.018051`
- `lag_13__CT4__duck_amount`: contribution `+0.009654`
- `lag_00__kill_diff_last_3s`: contribution `+0.007468`
- `lag_12__CT_place_RUINS`: contribution `+0.004894`
- `lag_01__T5__shots_fired`: contribution `+0.004887`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `35622`, seconds `138.50`, LSTM delta `-0.1651`

Top all feature movements:
- `lag_13__CT4__duck_amount`: contribution `-0.009654`
- `lag_00__T_kills_last_3s`: contribution `-0.009122`
- `lag_00__kill_diff_last_3s`: contribution `-0.007468`
- `lag_00__T_shots_fired_sum`: contribution `-0.006769`
- `lag_10__T5__duck_amount`: contribution `-0.004991`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `34374`, seconds `119.00`, LSTM delta `+0.1208`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.007468`
- `lag_09__CT_shots_fired_sum`: contribution `+0.006993`
- `lag_12__CT_place_RUINS`: contribution `+0.004894`
- `lag_13__CT_shots_fired_sum`: contribution `+0.004324`
- `lag_11__T_bomb_zone_count`: contribution `+0.004314`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `34054`, seconds `114.00`, LSTM delta `+0.1196`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.011282`
- `lag_02__CT_shots_fired_sum`: contribution `+0.008524`
- `lag_00__kill_diff_last_3s`: contribution `+0.007468`
- `lag_10__T5__duck_amount`: contribution `+0.006029`
- `lag_01__T_shots_fired_sum`: contribution `+0.004157`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `36262`, seconds `148.50`, LSTM delta `-0.1174`

Top all feature movements:
- `lag_06__CT_defusing_count`: contribution `-0.022229`
- `lag_00__T_kills_last_3s`: contribution `-0.009122`
- `lag_00__kill_diff_last_3s`: contribution `-0.007468`
- `lag_09__CT4__is_scoped`: contribution `-0.003455`
- `lag_12__CT_kills_last_3s`: contribution `-0.003275`

Top utility-only movements:
- No utility movement among the top local contributors.
