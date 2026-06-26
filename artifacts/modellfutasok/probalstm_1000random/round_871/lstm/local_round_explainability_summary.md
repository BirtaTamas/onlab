# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-tyloo-vs-vitality-bo3-aF98ikh3PjdqKlkdIJn9tC/tyloo-vs-vitality-m1-inferno.csv`
- round_num: `8`

## Largest probability jumps

- tick `55726`, seconds `80.00`, LSTM `0.8415`, delta `+0.1395`
- tick `55822`, seconds `81.50`, LSTM `0.9314`, delta `+0.0700`
- tick `54830`, seconds `66.00`, LSTM `0.6660`, delta `+0.0447`
- tick `56014`, seconds `84.50`, LSTM `0.9630`, delta `+0.0383`
- tick `55214`, seconds `72.00`, LSTM `0.6687`, delta `-0.0339`
- tick `51278`, seconds `10.50`, LSTM `0.6140`, delta `+0.0324`
- tick `55758`, seconds `80.50`, LSTM `0.8699`, delta `+0.0284`
- tick `54958`, seconds `68.00`, LSTM `0.6989`, delta `+0.0253`
- tick `55086`, seconds `70.00`, LSTM `0.7458`, delta `+0.0238`
- tick `51598`, seconds `15.50`, LSTM `0.6063`, delta `-0.0230`

## Top 15 local ridge features

- `lag_15__T_place_ARCH`: coefficient `0.002533`, |coef| `0.002533`
- `lag_00__T_place_ARCH`: coefficient `-0.002221`, |coef| `0.002221`
- `lag_11__CT_place_LIBRARY`: coefficient `-0.001308`, |coef| `0.001308`
- `lag_03__T_place_ARCH`: coefficient `-0.001283`, |coef| `0.001283`
- `lag_00__CT_kills_last_3s`: coefficient `0.001194`, |coef| `0.001194`
- `lag_00__kill_diff_last_3s`: coefficient `0.001025`, |coef| `0.001025`
- `lag_00__CT_damage_last_5s`: coefficient `0.000988`, |coef| `0.000988`
- `lag_00__T4__is_walking`: coefficient `-0.000979`, |coef| `0.000979`
- `lag_00__damage_diff_last_5s`: coefficient `0.000934`, |coef| `0.000934`
- `lag_13__T_place_ARCH`: coefficient `0.000890`, |coef| `0.000890`
- `lag_00__T4__alive`: coefficient `-0.000886`, |coef| `0.000886`
- `lag_12__T_place_ARCH`: coefficient `0.000822`, |coef| `0.000822`
- `lag_00__T_burning_players`: coefficient `-0.000819`, |coef| `0.000819`
- `lag_00__T4__armor`: coefficient `-0.000809`, |coef| `0.000809`
- `lag_00__T4__hp`: coefficient `-0.000790`, |coef| `0.000790`

## Top 10 utility ridge features

- `lag_03__T3__molly`: coefficient `-0.000697` (lowers CT win probability)
- `lag_00__T3__flash`: coefficient `-0.000594` (lowers CT win probability)
- `lag_08__T3__flash`: coefficient `0.000543` (raises CT win probability)
- `lag_15__CT4__flash`: coefficient `-0.000538` (lowers CT win probability)
- `lag_14__CT_flashes_last_5s`: coefficient `0.000523` (raises CT win probability)
- `lag_02__T_A_site_active_infernos`: coefficient `0.000493` (raises CT win probability)
- `lag_00__T3__utility_total`: coefficient `-0.000468` (lowers CT win probability)
- `lag_00__T2__smoke`: coefficient `-0.000463` (lowers CT win probability)
- `lag_06__T3__molly`: coefficient `-0.000377` (lowers CT win probability)
- `lag_00__T_flash_inv`: coefficient `-0.000373` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_15__T_place_ARCH`: coefficient `0.002533` (raises CT win probability)
- `lag_00__T_place_ARCH`: coefficient `-0.002221` (lowers CT win probability)
- `lag_11__CT_place_LIBRARY`: coefficient `-0.001308` (lowers CT win probability)
- `lag_03__T_place_ARCH`: coefficient `-0.001283` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001194` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001025` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.000988` (raises CT win probability)
- `lag_00__T4__is_walking`: coefficient `-0.000979` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000934` (raises CT win probability)
- `lag_13__T_place_ARCH`: coefficient `0.000890` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `55726`, seconds `80.00`, LSTM delta `+0.1395`

Top all feature movements:
- `lag_15__T_place_ARCH`: contribution `+0.023568`
- `lag_00__T_place_ARCH`: contribution `+0.020662`
- `lag_11__CT_place_LIBRARY`: contribution `+0.008388`
- `lag_00__CT_kills_last_3s`: contribution `+0.003448`
- `lag_15__CT1__duck_amount`: contribution `+0.002752`

Top utility-only movements:
- `lag_03__T3__molly`: contribution `+0.001547`

### tick `55822`, seconds `81.50`, LSTM delta `+0.0700`

Top all feature movements:
- `lag_03__T_place_ARCH`: contribution `+0.011937`
- `lag_00__CT_kills_last_3s`: contribution `+0.003448`
- `lag_00__kill_diff_last_3s`: contribution `+0.002467`
- `lag_03__CT1__is_scoped`: contribution `-0.002401`
- `lag_07__T2__duck_amount`: contribution `+0.002360`

Top utility-only movements:
- `lag_02__T_A_site_active_infernos`: contribution `+0.001467`

### tick `54830`, seconds `66.00`, LSTM delta `+0.0447`

Top all feature movements:
- `lag_11__CT_place_LIBRARY`: contribution `+0.008388`
- `lag_00__CT_kills_last_3s`: contribution `+0.003448`
- `lag_00__kill_diff_last_3s`: contribution `+0.002467`
- `lag_11__CT3__duck_amount`: contribution `+0.002188`
- `lag_00__T_burning_players`: contribution `+0.002075`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `56014`, seconds `84.50`, LSTM delta `+0.0383`

Top all feature movements:
- `lag_02__T_place_BALCONY`: contribution `+0.008705`
- `lag_04__T_place_BALCONY`: contribution `+0.004904`
- `lag_09__T_place_ARCH`: contribution `-0.004062`
- `lag_03__CT1__is_scoped`: contribution `+0.002401`
- `lag_00__T_place_BALCONY`: contribution `+0.002357`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `55214`, seconds `72.00`, LSTM delta `-0.0339`

Top all feature movements:
- `lag_02__CT_place_LIBRARY`: contribution `-0.003647`
- `lag_00__T3__is_scoped`: contribution `-0.003046`
- `lag_08__T3__is_scoped`: contribution `-0.002791`
- `lag_04__CT4__duck_amount`: contribution `-0.002147`
- `lag_06__CT2__is_walking`: contribution `-0.001654`

Top utility-only movements:
- `lag_10__CT_A_site_active_infernos`: contribution `-0.000700`
