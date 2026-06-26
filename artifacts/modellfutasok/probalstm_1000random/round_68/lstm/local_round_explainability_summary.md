# Local Round Explainability

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-falcons-vs-g2-bo3-Xf_UKx9fB2btv0Vy_VBVUC/falcons-vs-g2-m3-mirage.csv`
- round_num: `8`

## Largest probability jumps

- tick `54786`, seconds `29.00`, LSTM `0.1748`, delta `-0.2921`
- tick `54530`, seconds `25.00`, LSTM `0.4503`, delta `-0.1066`
- tick `54818`, seconds `29.50`, LSTM `0.1104`, delta `-0.0644`
- tick `54850`, seconds `30.00`, LSTM `0.0528`, delta `-0.0576`
- tick `54690`, seconds `27.50`, LSTM `0.3989`, delta `-0.0430`
- tick `54562`, seconds `25.50`, LSTM `0.4933`, delta `+0.0429`
- tick `54722`, seconds `28.00`, LSTM `0.4363`, delta `+0.0374`
- tick `54882`, seconds `30.50`, LSTM `0.0189`, delta `-0.0340`
- tick `54754`, seconds `28.50`, LSTM `0.4669`, delta `+0.0306`
- tick `53666`, seconds `11.50`, LSTM `0.6266`, delta `-0.0298`

## Top 15 local ridge features

- `lag_01__CT5__shots_fired`: coefficient `-0.001691`, |coef| `0.001691`
- `lag_02__CT5__shots_fired`: coefficient `-0.001603`, |coef| `0.001603`
- `lag_00__CT5__shots_fired`: coefficient `-0.001567`, |coef| `0.001567`
- `lag_03__T_place_CATWALK`: coefficient `-0.001437`, |coef| `0.001437`
- `lag_09__CT_place_TRUCK`: coefficient `0.001404`, |coef| `0.001404`
- `lag_00__T4__flash_duration`: coefficient `-0.001329`, |coef| `0.001329`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001296`, |coef| `0.001296`
- `lag_03__CT5__shots_fired`: coefficient `-0.001205`, |coef| `0.001205`
- `lag_12__T_place_CATWALK`: coefficient `-0.001196`, |coef| `0.001196`
- `lag_04__CT5__duck_amount`: coefficient `-0.001187`, |coef| `0.001187`
- `lag_04__CT_place_SHOP`: coefficient `-0.001134`, |coef| `0.001134`
- `lag_11__CT_place_TRUCK`: coefficient `-0.001060`, |coef| `0.001060`
- `lag_00__T_kills_last_3s`: coefficient `-0.001056`, |coef| `0.001056`
- `lag_08__CT3__shots_fired`: coefficient `-0.001037`, |coef| `0.001037`
- `lag_00__T_flash_duration_sum`: coefficient `-0.001032`, |coef| `0.001032`

## Top 10 utility ridge features

- `lag_00__T4__flash_duration`: coefficient `-0.001329` (lowers CT win probability)
- `lag_00__T_flash_duration_sum`: coefficient `-0.001032` (lowers CT win probability)
- `lag_10__CT_B_site_active_infernos`: coefficient `-0.000927` (lowers CT win probability)
- `lag_00__T1__flash_duration`: coefficient `-0.000808` (lowers CT win probability)
- `lag_00__CT5__smoke`: coefficient `0.000792` (raises CT win probability)
- `lag_00__CT_utility_inv`: coefficient `0.000666` (raises CT win probability)
- `lag_02__T4__flash_duration`: coefficient `-0.000661` (lowers CT win probability)
- `lag_14__CT3__molly`: coefficient `0.000642` (raises CT win probability)
- `lag_00__CT_smoke_inv`: coefficient `0.000639` (raises CT win probability)
- `lag_15__T_A_site_active_smokes`: coefficient `-0.000633` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT5__shots_fired`: coefficient `-0.001691` (lowers CT win probability)
- `lag_02__CT5__shots_fired`: coefficient `-0.001603` (lowers CT win probability)
- `lag_00__CT5__shots_fired`: coefficient `-0.001567` (lowers CT win probability)
- `lag_03__T_place_CATWALK`: coefficient `-0.001437` (lowers CT win probability)
- `lag_09__CT_place_TRUCK`: coefficient `0.001404` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001296` (raises CT win probability)
- `lag_03__CT5__shots_fired`: coefficient `-0.001205` (lowers CT win probability)
- `lag_12__T_place_CATWALK`: coefficient `-0.001196` (lowers CT win probability)
- `lag_04__CT5__duck_amount`: coefficient `-0.001187` (lowers CT win probability)
- `lag_04__CT_place_SHOP`: coefficient `-0.001134` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `54786`, seconds `29.00`, LSTM delta `-0.2921`

Top all feature movements:
- `lag_09__CT_place_TRUCK`: contribution `-0.009057`
- `lag_00__T4__flash_duration`: contribution `-0.007303`
- `lag_00__CT_shots_fired_sum`: contribution `-0.007203`
- `lag_11__CT_place_TRUCK`: contribution `-0.006834`
- `lag_07__T_shots_fired_sum`: contribution `-0.006625`

Top utility-only movements:
- `lag_00__T4__flash_duration`: contribution `-0.007303`
- `lag_00__T_flash_duration_sum`: contribution `-0.005113`
- `lag_00__T1__flash_duration`: contribution `-0.003344`

### tick `54530`, seconds `25.00`, LSTM delta `-0.1066`

Top all feature movements:
- `lag_03__T_place_CATWALK`: contribution `-0.004136`
- `lag_00__T_shots_fired_sum`: contribution `-0.003868`
- `lag_00__T_flashed_players`: contribution `+0.003857`
- `lag_01__CT_place_TRUCK`: contribution `-0.003797`
- `lag_00__T_kills_last_3s`: contribution `-0.003346`

Top utility-only movements:
- `lag_00__T_flash_duration_sum`: contribution `+0.001621`
- `lag_00__T1__flash_duration`: contribution `+0.001482`

### tick `54818`, seconds `29.50`, LSTM delta `-0.0644`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.004974`
- `lag_01__CT5__shots_fired`: contribution `-0.004471`
- `lag_02__CT5__shots_fired`: contribution `-0.004237`
- `lag_08__T_shots_fired_sum`: contribution `+0.003945`
- `lag_00__T3__shots_fired`: contribution `+0.003513`

Top utility-only movements:
- `lag_01__T4__flash_duration`: contribution `-0.002614`
- `lag_01__T_flash_duration_sum`: contribution `-0.002377`

### tick `54850`, seconds `30.00`, LSTM delta `-0.0576`

Top all feature movements:
- `lag_11__CT_place_TRUCK`: contribution `+0.006834`
- `lag_02__CT5__shots_fired`: contribution `-0.004237`
- `lag_02__T4__flash_duration`: contribution `-0.003635`
- `lag_03__CT5__shots_fired`: contribution `-0.003186`
- `lag_06__CT_place_SHOP`: contribution `-0.003010`

Top utility-only movements:
- `lag_02__T4__flash_duration`: contribution `-0.003635`
- `lag_02__T_flash_duration_sum`: contribution `-0.002258`

### tick `54690`, seconds `27.50`, LSTM delta `-0.0430`

Top all feature movements:
- `lag_08__T_flashed_players`: contribution `+0.004978`
- `lag_08__CT3__duck_amount`: contribution `-0.002492`
- `lag_01__CT_place_SHOP`: contribution `-0.002203`
- `lag_01__CT5__duck_amount`: contribution `-0.001886`
- `lag_08__T_place_CATWALK`: contribution `-0.001804`

Top utility-only movements:
- `lag_08__T_flash_duration_sum`: contribution `+0.001008`
