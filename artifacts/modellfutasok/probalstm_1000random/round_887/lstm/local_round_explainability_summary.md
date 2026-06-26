# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-furia-vs-g2-bo3-_Q5oHsnKyowoqEbJh5o3f8/furia-vs-g2-m1-train.csv`
- round_num: `5`

## Largest probability jumps

- tick `33703`, seconds `38.50`, LSTM `0.0543`, delta `-0.3370`
- tick `32039`, seconds `12.50`, LSTM `0.4657`, delta `-0.1166`
- tick `32231`, seconds `15.50`, LSTM `0.4514`, delta `+0.1066`
- tick `33543`, seconds `36.00`, LSTM `0.5199`, delta `-0.0559`
- tick `32711`, seconds `23.00`, LSTM `0.5431`, delta `+0.0493`
- tick `32135`, seconds `14.00`, LSTM `0.3833`, delta `-0.0412`
- tick `32263`, seconds `16.00`, LSTM `0.4897`, delta `+0.0384`
- tick `33575`, seconds `36.50`, LSTM `0.4821`, delta `-0.0378`
- tick `33607`, seconds `37.00`, LSTM `0.4447`, delta `-0.0374`
- tick `32679`, seconds `22.50`, LSTM `0.4938`, delta `+0.0373`

## Top 15 local ridge features

- `lag_00__T_place_IVY`: coefficient `0.002941`, |coef| `0.002941`
- `lag_05__CT_place_TUNNELS`: coefficient `0.002787`, |coef| `0.002787`
- `lag_00__CT3__alive`: coefficient `0.002539`, |coef| `0.002539`
- `lag_00__CT3__hp`: coefficient `0.002503`, |coef| `0.002503`
- `lag_05__CT2__alive`: coefficient `0.002465`, |coef| `0.002465`
- `lag_05__CT2__hp`: coefficient `0.002422`, |coef| `0.002422`
- `lag_00__CT3__armor`: coefficient `0.002401`, |coef| `0.002401`
- `lag_05__CT2__armor`: coefficient `0.002281`, |coef| `0.002281`
- `lag_00__T_kills_last_3s`: coefficient `-0.002261`, |coef| `0.002261`
- `lag_00__damage_diff_last_5s`: coefficient `0.002192`, |coef| `0.002192`
- `lag_00__T_damage_last_5s`: coefficient `-0.002117`, |coef| `0.002117`
- `lag_00__CT3__has_helmet`: coefficient `0.002113`, |coef| `0.002113`
- `lag_00__T3__is_scoped`: coefficient `-0.002108`, |coef| `0.002108`
- `lag_05__CT2__smoke`: coefficient `0.002069`, |coef| `0.002069`
- `lag_09__T4__duck_amount`: coefficient `-0.001957`, |coef| `0.001957`

## Top 10 utility ridge features

- `lag_05__CT2__smoke`: coefficient `0.002069` (raises CT win probability)
- `lag_05__CT2__utility_total`: coefficient `0.001789` (raises CT win probability)
- `lag_05__CT2__flash`: coefficient `0.001744` (raises CT win probability)
- `lag_13__T_active_smokes`: coefficient `0.001561` (raises CT win probability)
- `lag_13__T_B_site_active_smokes`: coefficient `0.001521` (raises CT win probability)
- `lag_15__T_A_site_active_smokes`: coefficient `0.001332` (raises CT win probability)
- `lag_15__T_active_smokes`: coefficient `0.001203` (raises CT win probability)
- `lag_04__CT2__smoke`: coefficient `0.001149` (raises CT win probability)
- `lag_13__active_smokes_total`: coefficient `0.001134` (raises CT win probability)
- `lag_15__active_smokes_total`: coefficient `0.001112` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_IVY`: coefficient `0.002941` (raises CT win probability)
- `lag_05__CT_place_TUNNELS`: coefficient `0.002787` (raises CT win probability)
- `lag_00__CT3__alive`: coefficient `0.002539` (raises CT win probability)
- `lag_00__CT3__hp`: coefficient `0.002503` (raises CT win probability)
- `lag_05__CT2__alive`: coefficient `0.002465` (raises CT win probability)
- `lag_05__CT2__hp`: coefficient `0.002422` (raises CT win probability)
- `lag_00__CT3__armor`: coefficient `0.002401` (raises CT win probability)
- `lag_05__CT2__armor`: coefficient `0.002281` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002261` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002192` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `33703`, seconds `38.50`, LSTM delta `-0.3370`

Top all feature movements:
- `lag_00__T_place_IVY`: contribution `-0.015715`
- `lag_00__T3__is_scoped`: contribution `-0.013521`
- `lag_05__T3__is_scoped`: contribution `-0.011015`
- `lag_05__CT_place_TUNNELS`: contribution `-0.008531`
- `lag_09__T4__duck_amount`: contribution `-0.007238`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `32039`, seconds `12.50`, LSTM delta `-0.1166`

Top all feature movements:
- `lag_13__T_place_DUMPSTER`: contribution `-0.009709`
- `lag_00__T_kills_last_3s`: contribution `-0.007163`
- `lag_08__T3__is_scoped`: contribution `-0.006499`
- `lag_09__T_place_DUMPSTER`: contribution `-0.005425`
- `lag_00__T_damage_last_5s`: contribution `-0.005077`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `32231`, seconds `15.50`, LSTM delta `+0.1066`

Top all feature movements:
- `lag_08__CT_place_ELECTRICALBOX`: contribution `+0.016039`
- `lag_10__CT_place_ELECTRICALBOX`: contribution `+0.013250`
- `lag_15__T_place_DUMPSTER`: contribution `+0.010258`
- `lag_00__T_kills_last_3s`: contribution `+0.007163`
- `lag_06__T_place_DUMPSTER`: contribution `+0.006528`

Top utility-only movements:
- `lag_08__CT2__flash_duration`: contribution `+0.004302`
- `lag_15__CT_B_site_active_smokes`: contribution `+0.001833`

### tick `33543`, seconds `36.00`, LSTM delta `-0.0559`

Top all feature movements:
- `lag_00__T3__is_scoped`: contribution `+0.013521`
- `lag_00__T_kills_last_3s`: contribution `-0.007163`
- `lag_15__T3__is_scoped`: contribution `-0.005309`
- `lag_00__T_damage_last_5s`: contribution `-0.005077`
- `lag_00__damage_diff_last_5s`: contribution `-0.004945`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `32711`, seconds `23.00`, LSTM delta `+0.0493`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.004577`
- `lag_01__T_place_TSTAIRS`: contribution `+0.003895`
- `lag_05__CT3__duck_amount`: contribution `+0.003272`
- `lag_03__T1__flash_duration`: contribution `+0.003135`
- `lag_00__CT_shots_fired_sum`: contribution `+0.002995`

Top utility-only movements:
- `lag_03__T1__flash_duration`: contribution `+0.003135`
