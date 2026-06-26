# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-g2-vs-falcons-bo3-VnJ8NRf6cDNnH9OuqiscGr/g2-vs-falcons-m1-ancient.csv`
- round_num: `1`

## Largest probability jumps

- tick `3873`, seconds `41.00`, LSTM `0.0670`, delta `-0.1049`
- tick `4929`, seconds `57.50`, LSTM `0.1207`, delta `+0.0952`
- tick `3681`, seconds `38.00`, LSTM `0.3410`, delta `-0.0876`
- tick `3841`, seconds `40.50`, LSTM `0.1719`, delta `-0.0753`
- tick `3649`, seconds `37.50`, LSTM `0.4286`, delta `-0.0670`
- tick `5185`, seconds `61.50`, LSTM `0.0303`, delta `-0.0666`
- tick `4481`, seconds `50.50`, LSTM `0.0500`, delta `-0.0533`
- tick `3713`, seconds `38.50`, LSTM `0.2989`, delta `-0.0421`
- tick `3745`, seconds `39.00`, LSTM `0.2599`, delta `-0.0390`
- tick `4257`, seconds `47.00`, LSTM `0.0769`, delta `-0.0353`

## Top 15 local ridge features

- `lag_01__T_place_SIDEHALL`: coefficient `-0.002515`, |coef| `0.002515`
- `lag_07__T_place_SIDEHALL`: coefficient `-0.002085`, |coef| `0.002085`
- `lag_00__T_place_SIDEHALL`: coefficient `-0.001934`, |coef| `0.001934`
- `lag_02__T_place_SIDEHALL`: coefficient `-0.001921`, |coef| `0.001921`
- `lag_15__T_place_SIDEHALL`: coefficient `-0.001772`, |coef| `0.001772`
- `lag_03__T_place_SIDEHALL`: coefficient `-0.001637`, |coef| `0.001637`
- `lag_06__T_place_SIDEHALL`: coefficient `-0.001600`, |coef| `0.001600`
- `lag_06__CT2__flash_duration`: coefficient `0.001531`, |coef| `0.001531`
- `lag_00__kill_diff_last_3s`: coefficient `0.001394`, |coef| `0.001394`
- `lag_00__T_kills_last_3s`: coefficient `-0.001266`, |coef| `0.001266`
- `lag_02__CT_place_RAMP`: coefficient `-0.001137`, |coef| `0.001137`
- `lag_00__damage_diff_last_5s`: coefficient `0.001130`, |coef| `0.001130`
- `lag_11__CT5__duck_amount`: coefficient `0.001121`, |coef| `0.001121`
- `lag_11__T_place_SIDEHALL`: coefficient `-0.001104`, |coef| `0.001104`
- `lag_15__CT_place_TSIDELOWER`: coefficient `-0.001099`, |coef| `0.001099`

## Top 10 utility ridge features

- `lag_06__CT2__flash_duration`: coefficient `0.001531` (raises CT win probability)
- `lag_05__CT2__flash_duration`: coefficient `0.001007` (raises CT win probability)
- `lag_04__CT2__flash_duration`: coefficient `0.000823` (raises CT win probability)
- `lag_06__CT_flash_duration_sum`: coefficient `0.000701` (raises CT win probability)
- `lag_11__CT2__flash_duration`: coefficient `0.000645` (raises CT win probability)
- `lag_11__T4__smoke`: coefficient `0.000626` (raises CT win probability)
- `lag_13__CT1__smoke`: coefficient `0.000570` (raises CT win probability)
- `lag_10__T3__molly`: coefficient `0.000555` (raises CT win probability)
- `lag_06__T_A_site_active_infernos`: coefficient `-0.000553` (lowers CT win probability)
- `lag_12__T_A_site_active_infernos`: coefficient `-0.000549` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__T_place_SIDEHALL`: coefficient `-0.002515` (lowers CT win probability)
- `lag_07__T_place_SIDEHALL`: coefficient `-0.002085` (lowers CT win probability)
- `lag_00__T_place_SIDEHALL`: coefficient `-0.001934` (lowers CT win probability)
- `lag_02__T_place_SIDEHALL`: coefficient `-0.001921` (lowers CT win probability)
- `lag_15__T_place_SIDEHALL`: coefficient `-0.001772` (lowers CT win probability)
- `lag_03__T_place_SIDEHALL`: coefficient `-0.001637` (lowers CT win probability)
- `lag_06__T_place_SIDEHALL`: coefficient `-0.001600` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001394` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001266` (lowers CT win probability)
- `lag_02__CT_place_RAMP`: coefficient `-0.001137` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `3873`, seconds `41.00`, LSTM delta `-0.1049`

Top all feature movements:
- `lag_07__T_place_SIDEHALL`: contribution `-0.027027`
- `lag_00__T_kills_last_3s`: contribution `-0.004011`
- `lag_01__T_place_MAINHALL`: contribution `-0.003726`
- `lag_08__T_place_SIDEHALL`: contribution `-0.003678`
- `lag_02__CT_place_RAMP`: contribution `-0.003397`

Top utility-only movements:
- `lag_12__T_A_site_active_infernos`: contribution `-0.001634`
- `lag_11__T4__smoke`: contribution `-0.001360`

### tick `4929`, seconds `57.50`, LSTM delta `+0.0952`

Top all feature movements:
- `lag_00__T_place_SIDEHALL`: contribution `+0.012533`
- `lag_14__CT_place_TSIDEUPPER`: contribution `+0.007493`
- `lag_15__CT_place_TSIDEUPPER`: contribution `+0.006555`
- `lag_11__CT2__flash_duration`: contribution `+0.004897`
- `lag_11__CT5__duck_amount`: contribution `+0.004233`

Top utility-only movements:
- `lag_11__CT2__flash_duration`: contribution `+0.004897`

### tick `3681`, seconds `38.00`, LSTM delta `-0.0876`

Top all feature movements:
- `lag_01__T_place_SIDEHALL`: contribution `-0.032598`
- `lag_02__T_place_SIDEHALL`: contribution `-0.012451`
- `lag_15__T_place_MAINHALL`: contribution `-0.002671`
- `lag_12__CT5__is_walking`: contribution `-0.002328`
- `lag_01__T_place_MIDDLE`: contribution `-0.002187`

Top utility-only movements:
- `lag_06__T_A_site_active_infernos`: contribution `-0.001647`
- `lag_10__T3__molly`: contribution `-0.001233`
- `lag_07__CT1__smoke`: contribution `-0.001158`
- `lag_05__T4__smoke`: contribution `-0.001133`
- `lag_06__CT4__smoke`: contribution `-0.001059`

### tick `3841`, seconds `40.50`, LSTM delta `-0.0753`

Top all feature movements:
- `lag_06__T_place_SIDEHALL`: contribution `-0.020737`
- `lag_07__T_place_SIDEHALL`: contribution `-0.013514`
- `lag_11__CT5__duck_amount`: contribution `-0.004233`
- `lag_02__CT_place_RAMP`: contribution `-0.003397`
- `lag_00__damage_diff_last_5s`: contribution `-0.002472`

Top utility-only movements:
- `lag_11__T_A_site_active_infernos`: contribution `-0.001435`

### tick `3649`, seconds `37.50`, LSTM delta `-0.0670`

Top all feature movements:
- `lag_00__T_place_SIDEHALL`: contribution `-0.025065`
- `lag_01__T_place_SIDEHALL`: contribution `-0.016299`
- `lag_09__T1__duck_amount`: contribution `-0.001868`
- `lag_14__T_place_MAINHALL`: contribution `-0.001697`
- `lag_05__T_A_site_active_infernos`: contribution `-0.001404`

Top utility-only movements:
- `lag_05__T_A_site_active_infernos`: contribution `-0.001404`
- `lag_04__T4__smoke`: contribution `-0.000957`
