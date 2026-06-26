# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-pain-bo3-6mWraId8pA69o5etX6dmBT/falcons-vs-pain-m1-inferno.csv`
- round_num: `3`

## Largest probability jumps

- tick `14741`, seconds `27.00`, LSTM `0.8486`, delta `+0.1288`
- tick `15413`, seconds `37.50`, LSTM `0.9118`, delta `+0.0719`
- tick `16853`, seconds `60.00`, LSTM `0.9729`, delta `+0.0319`
- tick `13365`, seconds `5.50`, LSTM `0.7647`, delta `+0.0238`
- tick `13045`, seconds `0.50`, LSTM `0.7261`, delta `+0.0228`
- tick `14581`, seconds `24.50`, LSTM `0.6897`, delta `-0.0212`
- tick `16437`, seconds `53.50`, LSTM `0.9497`, delta `+0.0211`
- tick `14709`, seconds `26.50`, LSTM `0.7197`, delta `+0.0198`
- tick `14357`, seconds `21.00`, LSTM `0.7447`, delta `-0.0192`
- tick `15381`, seconds `37.00`, LSTM `0.8399`, delta `+0.0171`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001399`, |coef| `0.001399`
- `lag_10__T_place_BALCONY`: coefficient `0.001184`, |coef| `0.001184`
- `lag_00__damage_diff_last_5s`: coefficient `0.001141`, |coef| `0.001141`
- `lag_00__CT_damage_last_5s`: coefficient `0.001129`, |coef| `0.001129`
- `lag_00__kill_diff_last_3s`: coefficient `0.001096`, |coef| `0.001096`
- `lag_08__T_place_BALCONY`: coefficient `-0.001096`, |coef| `0.001096`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001048`, |coef| `0.001048`
- `lag_12__CT_shots_fired_sum`: coefficient `-0.000984`, |coef| `0.000984`
- `lag_00__T3__alive`: coefficient `-0.000864`, |coef| `0.000864`
- `lag_00__CT2__duck_amount`: coefficient `0.000831`, |coef| `0.000831`
- `lag_00__T3__armor`: coefficient `-0.000812`, |coef| `0.000812`
- `lag_00__T3__hp`: coefficient `-0.000777`, |coef| `0.000777`
- `lag_00__T3__smoke`: coefficient `-0.000776`, |coef| `0.000776`
- `lag_13__CT3__smoke`: coefficient `-0.000724`, |coef| `0.000724`
- `lag_01__CT_shots_fired_sum`: coefficient `0.000700`, |coef| `0.000700`

## Top 10 utility ridge features

- `lag_00__T3__smoke`: coefficient `-0.000776` (lowers CT win probability)
- `lag_13__CT3__smoke`: coefficient `-0.000724` (lowers CT win probability)
- `lag_09__CT_A_site_active_infernos`: coefficient `-0.000576` (lowers CT win probability)
- `lag_08__CT4__smoke`: coefficient `0.000562` (raises CT win probability)
- `lag_00__T4__smoke`: coefficient `-0.000453` (lowers CT win probability)
- `lag_00__T_smoke_inv`: coefficient `-0.000419` (lowers CT win probability)
- `lag_08__CT4__flash_duration`: coefficient `-0.000415` (lowers CT win probability)
- `lag_12__CT3__smoke`: coefficient `-0.000390` (lowers CT win probability)
- `lag_13__CT4__flash_duration`: coefficient `0.000373` (raises CT win probability)
- `lag_08__CT_A_site_active_infernos`: coefficient `-0.000367` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001399` (raises CT win probability)
- `lag_10__T_place_BALCONY`: coefficient `0.001184` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001141` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001129` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001096` (raises CT win probability)
- `lag_08__T_place_BALCONY`: coefficient `-0.001096` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001048` (raises CT win probability)
- `lag_12__CT_shots_fired_sum`: coefficient `-0.000984` (lowers CT win probability)
- `lag_00__T3__alive`: coefficient `-0.000864` (lowers CT win probability)
- `lag_00__CT2__duck_amount`: coefficient `0.000831` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `14741`, seconds `27.00`, LSTM delta `+0.1288`

Top all feature movements:
- `lag_10__T_place_BALCONY`: contribution `+0.016275`
- `lag_08__T_place_BALCONY`: contribution `+0.015070`
- `lag_12__CT_shots_fired_sum`: contribution `+0.010252`
- `lag_12__CT3__shots_fired`: contribution `+0.004976`
- `lag_00__CT_shots_fired_sum`: contribution `+0.004367`

Top utility-only movements:
- `lag_09__CT_A_site_active_infernos`: contribution `+0.002033`
- `lag_08__CT4__flash_duration`: contribution `+0.001298`

### tick `15413`, seconds `37.50`, LSTM delta `+0.0719`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.004040`
- `lag_00__kill_diff_last_3s`: contribution `+0.002638`
- `lag_05__CT_place_RUINS`: contribution `+0.002354`
- `lag_02__CT_place_ARCH`: contribution `+0.002345`
- `lag_00__T3__alive`: contribution `+0.002091`

Top utility-only movements:
- `lag_00__T3__smoke`: contribution `+0.001687`
- `lag_13__CT3__smoke`: contribution `+0.001601`

### tick `16853`, seconds `60.00`, LSTM delta `+0.0319`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.004040`
- `lag_00__kill_diff_last_3s`: contribution `+0.002638`
- `lag_00__damage_diff_last_5s`: contribution `+0.002573`
- `lag_00__CT_damage_last_5s`: contribution `+0.002462`
- `lag_12__CT_shots_fired_sum`: contribution `+0.001367`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `13365`, seconds `5.50`, LSTM delta `+0.0238`

Top all feature movements:
- `lag_00__T_place_LOWERMID`: contribution `+0.003903`
- `lag_04__CT_place_ARCH`: contribution `+0.001142`
- `lag_00__CT_place_LIBRARY`: contribution `+0.001065`
- `lag_05__CT_place_LIBRARY`: contribution `+0.001012`
- `lag_11__CT_place_CTSPAWN`: contribution `+0.000916`

Top utility-only movements:
- `lag_11__flash_inv_diff`: contribution `+0.000382`
- `lag_11__molly_inv_diff`: contribution `+0.000353`

### tick `13045`, seconds `0.50`, LSTM delta `+0.0228`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `+0.001608`
- `lag_01__CT_closest_enemy_dist`: contribution `+0.000969`
- `lag_00__T_velocity_mean`: contribution `+0.000891`
- `lag_01__T_place_TSPAWN`: contribution `+0.000824`
- `lag_01__molly_inv_diff`: contribution `+0.000758`

Top utility-only movements:
- `lag_01__molly_inv_diff`: contribution `+0.000758`
- `lag_01__utility_inv_diff`: contribution `+0.000665`
- `lag_01__flash_inv_diff`: contribution `+0.000515`
- `lag_01__CT_molly_inv`: contribution `+0.000500`
- `lag_01__CT5__molly`: contribution `+0.000428`
