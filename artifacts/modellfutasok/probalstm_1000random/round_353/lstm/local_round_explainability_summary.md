# Local Round Explainability

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-mouz-vs-falcons-bo3-ET1FlQ7LAGQtcSrRzzPcv6/mouz-vs-falcons-m1-dust2.csv`
- round_num: `7`

## Largest probability jumps

- tick `46829`, seconds `84.50`, LSTM `0.0135`, delta `-0.2873`
- tick `43085`, seconds `26.00`, LSTM `0.2247`, delta `-0.2083`
- tick `46765`, seconds `83.50`, LSTM `0.2765`, delta `+0.1754`
- tick `41901`, seconds `7.50`, LSTM `0.3720`, delta `-0.0618`
- tick `43245`, seconds `28.50`, LSTM `0.1892`, delta `+0.0402`
- tick `43629`, seconds `34.50`, LSTM `0.1352`, delta `-0.0380`
- tick `41965`, seconds `8.50`, LSTM `0.4150`, delta `+0.0361`
- tick `43213`, seconds `28.00`, LSTM `0.1490`, delta `-0.0360`
- tick `44077`, seconds `41.50`, LSTM `0.1314`, delta `+0.0324`
- tick `44845`, seconds `53.50`, LSTM `0.1168`, delta `-0.0322`

## Top 15 local ridge features

- `lag_00__damage_diff_last_5s`: coefficient `0.002662`, |coef| `0.002662`
- `lag_06__CT_place_SHORTSTAIRS`: coefficient `0.002520`, |coef| `0.002520`
- `lag_00__kill_diff_last_3s`: coefficient `0.002502`, |coef| `0.002502`
- `lag_00__T_kills_last_3s`: coefficient `-0.002329`, |coef| `0.002329`
- `lag_02__CT4__is_scoped`: coefficient `-0.002325`, |coef| `0.002325`
- `lag_08__T5__is_scoped`: coefficient `-0.002161`, |coef| `0.002161`
- `lag_00__CT_place_EXTENDEDA`: coefficient `0.002151`, |coef| `0.002151`
- `lag_00__T_damage_last_5s`: coefficient `-0.001913`, |coef| `0.001913`
- `lag_07__T1__has_bomb`: coefficient `-0.001850`, |coef| `0.001850`
- `lag_00__CT4__alive`: coefficient `0.001797`, |coef| `0.001797`
- `lag_00__CT4__hp`: coefficient `0.001770`, |coef| `0.001770`
- `lag_00__CT5__duck_amount`: coefficient `-0.001759`, |coef| `0.001759`
- `lag_00__CT_place_ARAMP`: coefficient `-0.001667`, |coef| `0.001667`
- `lag_00__CT4__armor`: coefficient `0.001659`, |coef| `0.001659`
- `lag_08__T5__duck_amount`: coefficient `-0.001569`, |coef| `0.001569`

## Top 10 utility ridge features

- `lag_13__T5__molly`: coefficient `0.001416` (raises CT win probability)
- `lag_12__T4__molly`: coefficient `-0.001085` (lowers CT win probability)
- `lag_13__CT_active_infernos`: coefficient `0.000978` (raises CT win probability)
- `lag_01__CT_utility_damage_last_5s`: coefficient `-0.000766` (lowers CT win probability)
- `lag_13__active_infernos_total`: coefficient `0.000728` (raises CT win probability)
- `lag_11__CT3__flash_duration`: coefficient `0.000703` (raises CT win probability)
- `lag_04__CT1__flash_duration`: coefficient `0.000672` (raises CT win probability)
- `lag_01__utility_damage_diff_last_5s`: coefficient `-0.000644` (lowers CT win probability)
- `lag_08__T2__molly`: coefficient `-0.000631` (lowers CT win probability)
- `lag_00__CT2__molly`: coefficient `0.000578` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__damage_diff_last_5s`: coefficient `0.002662` (raises CT win probability)
- `lag_06__CT_place_SHORTSTAIRS`: coefficient `0.002520` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002502` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002329` (lowers CT win probability)
- `lag_02__CT4__is_scoped`: coefficient `-0.002325` (lowers CT win probability)
- `lag_08__T5__is_scoped`: coefficient `-0.002161` (lowers CT win probability)
- `lag_00__CT_place_EXTENDEDA`: coefficient `0.002151` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001913` (lowers CT win probability)
- `lag_07__T1__has_bomb`: coefficient `-0.001850` (lowers CT win probability)
- `lag_00__CT4__alive`: coefficient `0.001797` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `46829`, seconds `84.50`, LSTM delta `-0.2873`

Top all feature movements:
- `lag_06__CT_place_SHORTSTAIRS`: contribution `-0.028094`
- `lag_00__CT_place_EXTENDEDA`: contribution `-0.024156`
- `lag_00__damage_diff_last_5s`: contribution `-0.016638`
- `lag_00__T_kills_last_3s`: contribution `-0.014756`
- `lag_00__kill_diff_last_3s`: contribution `-0.012046`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `43085`, seconds `26.00`, LSTM delta `-0.2083`

Top all feature movements:
- `lag_08__T5__is_scoped`: contribution `-0.010308`
- `lag_02__CT4__is_scoped`: contribution `-0.007925`
- `lag_00__T_kills_last_3s`: contribution `-0.007378`
- `lag_09__T_place_OUTSIDELONG`: contribution `-0.007293`
- `lag_00__T5__is_scoped`: contribution `-0.006909`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `46765`, seconds `83.50`, LSTM delta `+0.1754`

Top all feature movements:
- `lag_03__T_place_TUNNELSTAIRS`: contribution `+0.010329`
- `lag_08__T_place_ARAMP`: contribution `+0.009722`
- `lag_13__CT3__is_scoped`: contribution `+0.006168`
- `lag_00__T_place_LONGA`: contribution `+0.006093`
- `lag_00__kill_diff_last_3s`: contribution `+0.006023`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `41901`, seconds `7.50`, LSTM delta `-0.0618`

Top all feature movements:
- `lag_00__CT5__duck_amount`: contribution `-0.006641`
- `lag_08__CT_place_MIDDOORS`: contribution `-0.004111`
- `lag_05__CT4__is_walking`: contribution `+0.002629`
- `lag_02__CT_place_BDOORS`: contribution `-0.002579`
- `lag_06__CT4__is_scoped`: contribution `-0.002130`

Top utility-only movements:
- `lag_00__CT1__flash_duration`: contribution `-0.001928`
- `lag_00__CT2__molly`: contribution `-0.001424`

### tick `43245`, seconds `28.50`, LSTM delta `+0.0402`

Top all feature movements:
- `lag_02__CT_place_EXTENDEDA`: contribution `+0.005350`
- `lag_15__CT4__is_scoped`: contribution `+0.004148`
- `lag_05__T4__duck_amount`: contribution `+0.003532`
- `lag_00__T3__is_walking`: contribution `+0.003069`
- `lag_05__CT4__is_walking`: contribution `-0.002629`

Top utility-only movements:
- No utility movement among the top local contributors.
