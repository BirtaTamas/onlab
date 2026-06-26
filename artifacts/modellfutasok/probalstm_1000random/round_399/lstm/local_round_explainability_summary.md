# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-3dmax-vs-faze-bo3-oPmZd_fSjJ93cG46OkNLxM/3dmax-vs-faze-m2-ancient.csv`
- round_num: `5`

## Largest probability jumps

- tick `29412`, seconds `17.00`, LSTM `0.2748`, delta `-0.1690`
- tick `29540`, seconds `19.00`, LSTM `0.0741`, delta `-0.1582`
- tick `29956`, seconds `25.50`, LSTM `0.0748`, delta `-0.0649`
- tick `29444`, seconds `17.50`, LSTM `0.2295`, delta `-0.0452`
- tick `29988`, seconds `26.00`, LSTM `0.0350`, delta `-0.0398`
- tick `29316`, seconds `15.50`, LSTM `0.4575`, delta `-0.0366`
- tick `28868`, seconds `8.50`, LSTM `0.5327`, delta `-0.0232`
- tick `29028`, seconds `11.00`, LSTM `0.5298`, delta `+0.0202`
- tick `29572`, seconds `19.50`, LSTM `0.0550`, delta `-0.0192`
- tick `29188`, seconds `13.50`, LSTM `0.5162`, delta `-0.0184`

## Top 15 local ridge features

- `lag_00__CT_place_UNKNOWN`: coefficient `0.003256`, |coef| `0.003256`
- `lag_10__CT3__flash_duration`: coefficient `-0.001368`, |coef| `0.001368`
- `lag_06__CT3__flash_duration`: coefficient `-0.001337`, |coef| `0.001337`
- `lag_07__CT_shots_fired_sum`: coefficient `0.001325`, |coef| `0.001325`
- `lag_03__CT3__flash_duration`: coefficient `-0.001088`, |coef| `0.001088`
- `lag_02__CT_place_UNKNOWN`: coefficient `0.001072`, |coef| `0.001072`
- `lag_00__T_kills_last_3s`: coefficient `-0.001024`, |coef| `0.001024`
- `lag_14__CT3__flash_duration`: coefficient `-0.000983`, |coef| `0.000983`
- `lag_00__kill_diff_last_3s`: coefficient `0.000899`, |coef| `0.000899`
- `lag_04__T_utility_damage_last_5s`: coefficient `-0.000895`, |coef| `0.000895`
- `lag_04__CT_place_TSIDEUPPER`: coefficient `-0.000849`, |coef| `0.000849`
- `lag_09__CT3__flash_duration`: coefficient `-0.000824`, |coef| `0.000824`
- `lag_13__CT3__flash_duration`: coefficient `-0.000818`, |coef| `0.000818`
- `lag_15__CT3__flash_duration`: coefficient `-0.000779`, |coef| `0.000779`
- `lag_09__CT_shots_fired_sum`: coefficient `-0.000778`, |coef| `0.000778`

## Top 10 utility ridge features

- `lag_10__CT3__flash_duration`: coefficient `-0.001368` (lowers CT win probability)
- `lag_06__CT3__flash_duration`: coefficient `-0.001337` (lowers CT win probability)
- `lag_03__CT3__flash_duration`: coefficient `-0.001088` (lowers CT win probability)
- `lag_14__CT3__flash_duration`: coefficient `-0.000983` (lowers CT win probability)
- `lag_04__T_utility_damage_last_5s`: coefficient `-0.000895` (lowers CT win probability)
- `lag_09__CT3__flash_duration`: coefficient `-0.000824` (lowers CT win probability)
- `lag_13__CT3__flash_duration`: coefficient `-0.000818` (lowers CT win probability)
- `lag_15__CT3__flash_duration`: coefficient `-0.000779` (lowers CT win probability)
- `lag_08__T_utility_damage_last_5s`: coefficient `-0.000767` (lowers CT win probability)
- `lag_03__CT_B_site_active_infernos`: coefficient `0.000710` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_UNKNOWN`: coefficient `0.003256` (raises CT win probability)
- `lag_07__CT_shots_fired_sum`: coefficient `0.001325` (raises CT win probability)
- `lag_02__CT_place_UNKNOWN`: coefficient `0.001072` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001024` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000899` (raises CT win probability)
- `lag_04__CT_place_TSIDEUPPER`: coefficient `-0.000849` (lowers CT win probability)
- `lag_09__CT_shots_fired_sum`: coefficient `-0.000778` (lowers CT win probability)
- `lag_15__CT_place_ALLEY`: coefficient `0.000768` (raises CT win probability)
- `lag_06__T_flashed_players`: coefficient `-0.000765` (lowers CT win probability)
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.000705` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `29412`, seconds `17.00`, LSTM delta `-0.1690`

Top all feature movements:
- `lag_07__CT_shots_fired_sum`: contribution `-0.009206`
- `lag_06__CT3__flash_duration`: contribution `-0.007348`
- `lag_04__CT_place_TSIDEUPPER`: contribution `-0.006382`
- `lag_04__T_utility_damage_last_5s`: contribution `-0.005498`
- `lag_06__T_flashed_players`: contribution `-0.004428`

Top utility-only movements:
- `lag_06__CT3__flash_duration`: contribution `-0.007348`
- `lag_04__T_utility_damage_last_5s`: contribution `-0.005498`
- `lag_07__CT_flash_duration_sum`: contribution `-0.002479`
- `lag_03__CT_B_site_active_infernos`: contribution `-0.002440`
- `lag_03__CT3__flash_duration`: contribution `+0.002332`

### tick `29540`, seconds `19.00`, LSTM delta `-0.1582`

Top all feature movements:
- `lag_10__CT3__flash_duration`: contribution `-0.007520`
- `lag_07__CT_shots_fired_sum`: contribution `-0.007365`
- `lag_03__CT3__flash_duration`: contribution `-0.006890`
- `lag_00__CT_place_TSIDEUPPER`: contribution `-0.005302`
- `lag_08__T_utility_damage_last_5s`: contribution `-0.004707`

Top utility-only movements:
- `lag_10__CT3__flash_duration`: contribution `-0.007520`
- `lag_03__CT3__flash_duration`: contribution `-0.006890`
- `lag_08__T_utility_damage_last_5s`: contribution `-0.004707`
- `lag_11__CT3__flash_duration`: contribution `+0.002010`
- `lag_08__utility_damage_diff_last_5s`: contribution `-0.001945`

### tick `29956`, seconds `25.50`, LSTM delta `-0.0649`

Top all feature movements:
- `lag_13__CT_place_TSIDEUPPER`: contribution `-0.003593`
- `lag_12__T_shots_fired_sum`: contribution `-0.003260`
- `lag_02__T_place_HOUSE`: contribution `-0.002991`
- `lag_12__T5__shots_fired`: contribution `-0.002475`
- `lag_11__T_utility_damage_last_5s`: contribution `-0.002374`

Top utility-only movements:
- `lag_11__T_utility_damage_last_5s`: contribution `-0.002374`
- `lag_11__T1__flash_duration`: contribution `-0.001837`
- `lag_12__CT5__flash_duration`: contribution `+0.001608`
- `lag_13__T_utility_damage_last_5s`: contribution `-0.001226`
- `lag_07__CT_active_infernos`: contribution `-0.001162`

### tick `29444`, seconds `17.50`, LSTM delta `-0.0452`

Top all feature movements:
- `lag_07__T_flashed_players`: contribution `-0.003959`
- `lag_08__CT3__flash_duration`: contribution `+0.003545`
- `lag_05__T_utility_damage_last_5s`: contribution `-0.003363`
- `lag_08__CT_shots_fired_sum`: contribution `+0.003285`
- `lag_07__CT_flash_duration_sum`: contribution `+0.002744`

Top utility-only movements:
- `lag_08__CT3__flash_duration`: contribution `+0.003545`
- `lag_05__T_utility_damage_last_5s`: contribution `-0.003363`
- `lag_07__CT_flash_duration_sum`: contribution `+0.002744`
- `lag_01__CT3__flash_duration`: contribution `+0.002700`
- `lag_03__CT_B_site_active_infernos`: contribution `-0.002440`

### tick `29988`, seconds `26.00`, LSTM delta `-0.0398`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.003244`
- `lag_00__T_shots_fired_sum`: contribution `+0.002804`
- `lag_12__T_utility_damage_last_5s`: contribution `-0.002468`
- `lag_01__T_shots_fired_sum`: contribution `-0.002443`
- `lag_03__T_place_HOUSE`: contribution `-0.002239`

Top utility-only movements:
- `lag_12__T_utility_damage_last_5s`: contribution `-0.002468`
- `lag_08__CT_active_infernos`: contribution `-0.001050`
- `lag_12__utility_damage_diff_last_5s`: contribution `-0.000923`
- `lag_12__T1__flash_duration`: contribution `-0.000821`
- `lag_13__T_utility_damage_last_5s`: contribution `-0.000818`
