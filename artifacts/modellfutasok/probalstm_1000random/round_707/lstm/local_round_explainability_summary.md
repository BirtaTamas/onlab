# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-aurora-vs-falcons-bo3-5oHSxtVT-5F3Op7ZcgBMjW/aurora-vs-falcons-m1-inferno.csv`
- round_num: `2`

## Largest probability jumps

- tick `14639`, seconds `69.50`, LSTM `0.0919`, delta `-0.1921`
- tick `10223`, seconds `0.50`, LSTM `0.2433`, delta `-0.0613`
- tick `14607`, seconds `69.00`, LSTM `0.2839`, delta `-0.0570`
- tick `11887`, seconds `26.50`, LSTM `0.2924`, delta `-0.0529`
- tick `10479`, seconds `4.50`, LSTM `0.3370`, delta `+0.0502`
- tick `14415`, seconds `66.00`, LSTM `0.3048`, delta `+0.0424`
- tick `11567`, seconds `21.50`, LSTM `0.2829`, delta `+0.0416`
- tick `10511`, seconds `5.00`, LSTM `0.3784`, delta `+0.0414`
- tick `10671`, seconds `7.50`, LSTM `0.3843`, delta `-0.0397`
- tick `14703`, seconds `70.50`, LSTM `0.0349`, delta `-0.0371`

## Top 15 local ridge features

- `lag_09__T_place_ARCH`: coefficient `-0.002438`, |coef| `0.002438`
- `lag_12__T_place_ARCH`: coefficient `-0.001946`, |coef| `0.001946`
- `lag_00__CT_place_ARCH`: coefficient `0.001785`, |coef| `0.001785`
- `lag_02__T_place_ARCH`: coefficient `0.001705`, |coef| `0.001705`
- `lag_10__T_place_ARCH`: coefficient `-0.001463`, |coef| `0.001463`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001457`, |coef| `0.001457`
- `lag_01__CT_flash_alpha_mean`: coefficient `0.001294`, |coef| `0.001294`
- `lag_04__CT3__is_walking`: coefficient `-0.001294`, |coef| `0.001294`
- `lag_00__T3__shots_fired`: coefficient `-0.001225`, |coef| `0.001225`
- `lag_00__CT5__is_walking`: coefficient `-0.001187`, |coef| `0.001187`
- `lag_03__T_place_ARCH`: coefficient `0.001160`, |coef| `0.001160`
- `lag_00__T1__duck_amount`: coefficient `-0.001034`, |coef| `0.001034`
- `lag_08__CT2__is_walking`: coefficient `0.001010`, |coef| `0.001010`
- `lag_01__CT_place_ARCH`: coefficient `0.001001`, |coef| `0.001001`
- `lag_14__CT3__is_walking`: coefficient `-0.000991`, |coef| `0.000991`

## Top 10 utility ridge features

- `lag_01__CT_flash_alpha_mean`: coefficient `0.001294` (raises CT win probability)
- `lag_12__CT2__smoke`: coefficient `0.000753` (raises CT win probability)
- `lag_01__CT_active_smokes`: coefficient `0.000712` (raises CT win probability)
- `lag_00__CT_active_smokes`: coefficient `0.000708` (raises CT win probability)
- `lag_15__T_utility_damage_last_5s`: coefficient `0.000700` (raises CT win probability)
- `lag_15__CT_utility_damage_last_5s`: coefficient `0.000693` (raises CT win probability)
- `lag_00__CT_A_site_active_smokes`: coefficient `0.000683` (raises CT win probability)
- `lag_11__T3__flash_duration`: coefficient `0.000671` (raises CT win probability)
- `lag_15__CT_flash_alpha_mean`: coefficient `0.000665` (raises CT win probability)
- `lag_02__CT_flash_alpha_mean`: coefficient `0.000664` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_09__T_place_ARCH`: coefficient `-0.002438` (lowers CT win probability)
- `lag_12__T_place_ARCH`: coefficient `-0.001946` (lowers CT win probability)
- `lag_00__CT_place_ARCH`: coefficient `0.001785` (raises CT win probability)
- `lag_02__T_place_ARCH`: coefficient `0.001705` (raises CT win probability)
- `lag_10__T_place_ARCH`: coefficient `-0.001463` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001457` (lowers CT win probability)
- `lag_04__CT3__is_walking`: coefficient `-0.001294` (lowers CT win probability)
- `lag_00__T3__shots_fired`: coefficient `-0.001225` (lowers CT win probability)
- `lag_00__CT5__is_walking`: coefficient `-0.001187` (lowers CT win probability)
- `lag_03__T_place_ARCH`: coefficient `0.001160` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `14639`, seconds `69.50`, LSTM delta `-0.1921`

Top all feature movements:
- `lag_09__T_place_ARCH`: contribution `-0.022680`
- `lag_12__T_place_ARCH`: contribution `-0.018105`
- `lag_02__T_place_ARCH`: contribution `-0.015858`
- `lag_10__T_place_ARCH`: contribution `-0.013611`
- `lag_00__T_shots_fired_sum`: contribution `-0.005463`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `10223`, seconds `0.50`, LSTM delta `-0.0613`

Top all feature movements:
- `lag_01__CT_flash_alpha_mean`: contribution `-0.007650`
- `lag_01__CT_place_CTSPAWN`: contribution `-0.004083`
- `lag_01__T_place_TSPAWN`: contribution `-0.003804`
- `lag_01__T_closest_enemy_dist`: contribution `-0.003615`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.003354`

Top utility-only movements:
- `lag_01__CT_flash_alpha_mean`: contribution `-0.007650`
- `lag_01__T_flash_alpha_mean`: contribution `-0.002911`
- `lag_01__molly_inv_diff`: contribution `-0.000864`
- `lag_01__CT_active_smokes`: contribution `-0.000578`
- `lag_01__T1__molly`: contribution `-0.000563`

### tick `14607`, seconds `69.00`, LSTM delta `-0.0570`

Top all feature movements:
- `lag_09__T_place_ARCH`: contribution `-0.022680`
- `lag_11__T_place_ARCH`: contribution `-0.008477`
- `lag_08__T_place_ARCH`: contribution `-0.004444`
- `lag_01__T_place_ARCH`: contribution `-0.003595`
- `lag_00__CT_place_RUINS`: contribution `+0.002888`

Top utility-only movements:
- `lag_00__CT_active_smokes`: contribution `-0.001635`

### tick `11887`, seconds `26.50`, LSTM delta `-0.0529`

Top all feature movements:
- `lag_00__CT_place_ARCH`: contribution `-0.014567`
- `lag_00__T_shots_fired_sum`: contribution `-0.005463`
- `lag_00__T3__shots_fired`: contribution `-0.003708`
- `lag_15__T3__duck_amount`: contribution `-0.002458`
- `lag_14__CT3__is_walking`: contribution `+0.002367`

Top utility-only movements:
- `lag_15__CT_utility_damage_last_5s`: contribution `-0.002060`

### tick `10479`, seconds `4.50`, LSTM delta `+0.0502`

Top all feature movements:
- `lag_01__CT_place_ARCH`: contribution `+0.008166`
- `lag_05__CT_place_LIBRARY`: contribution `+0.005498`
- `lag_00__CT_place_LIBRARY`: contribution `+0.003946`
- `lag_00__T_place_LOWERMID`: contribution `+0.003240`
- `lag_00__CT_place_RUINS`: contribution `+0.002888`

Top utility-only movements:
- `lag_09__T_flash_alpha_mean`: contribution `+0.001099`
- `lag_09__CT_flash_alpha_mean`: contribution `+0.000789`
- `lag_09__CT3__smoke`: contribution `+0.000575`
- `lag_09__CT_smoke_inv`: contribution `+0.000524`
