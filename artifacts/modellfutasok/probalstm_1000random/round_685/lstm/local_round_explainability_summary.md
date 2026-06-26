# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-furia-bo5-6eeTFVdtPEH4qPNc6w4Z3Y/the-mongolz-vs-furia-m5-dust2.csv`
- round_num: `2`

## Largest probability jumps

- tick `26715`, seconds `53.50`, LSTM `0.9565`, delta `+0.0616`
- tick `23899`, seconds `9.50`, LSTM `0.8847`, delta `+0.0282`
- tick `23355`, seconds `1.00`, LSTM `0.8915`, delta `-0.0187`
- tick `25979`, seconds `42.00`, LSTM `0.9340`, delta `+0.0184`
- tick `25563`, seconds `35.50`, LSTM `0.9296`, delta `+0.0163`
- tick `23835`, seconds `8.50`, LSTM `0.8543`, delta `-0.0162`
- tick `23707`, seconds `6.50`, LSTM `0.8963`, delta `+0.0155`
- tick `25531`, seconds `35.00`, LSTM `0.9133`, delta `-0.0150`
- tick `24123`, seconds `13.00`, LSTM `0.9322`, delta `+0.0147`
- tick `23739`, seconds `7.00`, LSTM `0.8816`, delta `-0.0147`

## Top 15 local ridge features

- `lag_00__T_place_MIDDOORS`: coefficient `-0.000791`, |coef| `0.000791`
- `lag_06__T_place_TUNNELSTAIRS`: coefficient `0.000680`, |coef| `0.000680`
- `lag_01__T_place_TUNNELSTAIRS`: coefficient `-0.000604`, |coef| `0.000604`
- `lag_01__T_place_EXTENDEDA`: coefficient `0.000556`, |coef| `0.000556`
- `lag_11__T_place_MIDDOORS`: coefficient `0.000518`, |coef| `0.000518`
- `lag_15__T_place_SHORTSTAIRS`: coefficient `0.000513`, |coef| `0.000513`
- `lag_01__CT_place_HOLE`: coefficient `-0.000485`, |coef| `0.000485`
- `lag_00__CT5__is_walking`: coefficient `-0.000449`, |coef| `0.000449`
- `lag_00__T_bomb_carrier_alive`: coefficient `-0.000437`, |coef| `0.000437`
- `lag_08__CT_place_PIT`: coefficient `-0.000433`, |coef| `0.000433`
- `lag_03__T_place_MIDDOORS`: coefficient `0.000432`, |coef| `0.000432`
- `lag_06__T_place_LOWERTUNNEL`: coefficient `-0.000413`, |coef| `0.000413`
- `lag_00__T1__is_scoped`: coefficient `0.000411`, |coef| `0.000411`
- `lag_14__T5__has_bomb`: coefficient `-0.000402`, |coef| `0.000402`
- `lag_01__T_place_SHORTSTAIRS`: coefficient `-0.000400`, |coef| `0.000400`

## Top 10 utility ridge features

- `lag_03__CT_flashes_last_5s`: coefficient `-0.000355` (lowers CT win probability)
- `lag_10__CT_flashes_last_5s`: coefficient `-0.000267` (lowers CT win probability)
- `lag_07__CT_utility_damage_last_5s`: coefficient `0.000222` (raises CT win probability)
- `lag_00__CT_flashes_last_5s`: coefficient `0.000218` (raises CT win probability)
- `lag_03__CT_utility_damage_last_5s`: coefficient `0.000218` (raises CT win probability)
- `lag_02__CT_flashes_last_5s`: coefficient `-0.000208` (lowers CT win probability)
- `lag_12__CT_flashes_last_5s`: coefficient `-0.000202` (lowers CT win probability)
- `lag_02__CT1__utility_total`: coefficient `-0.000196` (lowers CT win probability)
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.000195` (raises CT win probability)
- `lag_13__CT_flashes_last_5s`: coefficient `0.000195` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_MIDDOORS`: coefficient `-0.000791` (lowers CT win probability)
- `lag_06__T_place_TUNNELSTAIRS`: coefficient `0.000680` (raises CT win probability)
- `lag_01__T_place_TUNNELSTAIRS`: coefficient `-0.000604` (lowers CT win probability)
- `lag_01__T_place_EXTENDEDA`: coefficient `0.000556` (raises CT win probability)
- `lag_11__T_place_MIDDOORS`: coefficient `0.000518` (raises CT win probability)
- `lag_15__T_place_SHORTSTAIRS`: coefficient `0.000513` (raises CT win probability)
- `lag_01__CT_place_HOLE`: coefficient `-0.000485` (lowers CT win probability)
- `lag_00__CT5__is_walking`: coefficient `-0.000449` (lowers CT win probability)
- `lag_00__T_bomb_carrier_alive`: coefficient `-0.000437` (lowers CT win probability)
- `lag_08__CT_place_PIT`: coefficient `-0.000433` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `26715`, seconds `53.50`, LSTM delta `+0.0616`

Top all feature movements:
- `lag_06__T_place_TUNNELSTAIRS`: contribution `+0.004751`
- `lag_01__T_place_TUNNELSTAIRS`: contribution `+0.004214`
- `lag_00__T_place_MIDDOORS`: contribution `+0.003362`
- `lag_01__T_place_EXTENDEDA`: contribution `+0.002755`
- `lag_11__T_place_MIDDOORS`: contribution `+0.002203`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `23899`, seconds `9.50`, LSTM delta `+0.0282`

Top all feature movements:
- `lag_01__CT_place_HOLE`: contribution `+0.005415`
- `lag_03__CT_place_HOLE`: contribution `+0.002442`
- `lag_09__CT_place_LONGA`: contribution `+0.001390`
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.001330`
- `lag_06__CT_place_BDOORS`: contribution `+0.001282`

Top utility-only movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.001330`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.000895`

### tick `23355`, seconds `1.00`, LSTM delta `-0.0187`

Top all feature movements:
- `lag_02__CT_place_CTSPAWN`: contribution `-0.001662`
- `lag_02__T_place_TSPAWN`: contribution `-0.001198`
- `lag_01__T_velocity_mean`: contribution `-0.001136`
- `lag_00__T1__duck_amount`: contribution `-0.000750`
- `lag_01__CT_velocity_mean`: contribution `-0.000737`

Top utility-only movements:
- `lag_02__CT1__utility_total`: contribution `-0.000570`
- `lag_02__CT1__flash`: contribution `-0.000440`
- `lag_02__CT1__molly`: contribution `-0.000338`
- `lag_02__smoke_inv_diff`: contribution `-0.000196`
- `lag_02__CT1__smoke`: contribution `-0.000179`

### tick `25979`, seconds `42.00`, LSTM delta `+0.0184`

Top all feature movements:
- `lag_03__CT_flashes_last_5s`: contribution `+0.003903`
- `lag_13__CT_flashes_last_5s`: contribution `+0.002142`
- `lag_02__T1__is_scoped`: contribution `+0.001627`
- `lag_07__CT2__duck_amount`: contribution `-0.001344`
- `lag_00__CT5__is_walking`: contribution `+0.001075`

Top utility-only movements:
- `lag_03__CT_flashes_last_5s`: contribution `+0.003903`
- `lag_13__CT_flashes_last_5s`: contribution `+0.002142`

### tick `25563`, seconds `35.50`, LSTM delta `+0.0163`

Top all feature movements:
- `lag_00__CT_flashes_last_5s`: contribution `+0.002395`
- `lag_01__T2__duck_amount`: contribution `+0.001162`
- `lag_15__CT1__duck_amount`: contribution `+0.001157`
- `lag_14__CT4__duck_amount`: contribution `+0.000906`
- `lag_04__T5__is_walking`: contribution `+0.000738`

Top utility-only movements:
- `lag_00__CT_flashes_last_5s`: contribution `+0.002395`
- `lag_02__CT1__molly`: contribution `+0.000423`
