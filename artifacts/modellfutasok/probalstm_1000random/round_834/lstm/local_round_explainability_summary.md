# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-heroic-vs-aurora-bo3-QigxwcikBDdlIOkrYDpY7y/heroic-vs-aurora-m2-dust2.csv`
- round_num: `20`

## Largest probability jumps

- tick `153168`, seconds `70.50`, LSTM `0.8818`, delta `+0.3833`
- tick `153616`, seconds `77.50`, LSTM `0.9496`, delta `+0.1158`
- tick `153104`, seconds `69.50`, LSTM `0.4288`, delta `+0.1033`
- tick `153136`, seconds `70.00`, LSTM `0.4985`, delta `+0.0697`
- tick `153456`, seconds `75.00`, LSTM `0.8855`, delta `-0.0643`
- tick `153200`, seconds `71.00`, LSTM `0.9346`, delta `+0.0528`
- tick `152976`, seconds `67.50`, LSTM `0.2969`, delta `-0.0345`
- tick `151408`, seconds `43.00`, LSTM `0.3824`, delta `-0.0339`
- tick `150480`, seconds `28.50`, LSTM `0.3604`, delta `+0.0311`
- tick `153008`, seconds `68.00`, LSTM `0.3279`, delta `+0.0310`

## Top 15 local ridge features

- `lag_00__T_place_MIDDOORS`: coefficient `-0.003149`, |coef| `0.003149`
- `lag_04__CT_place_HOLE`: coefficient `0.002520`, |coef| `0.002520`
- `lag_00__CT_kills_last_3s`: coefficient `0.002511`, |coef| `0.002511`
- `lag_00__kill_diff_last_3s`: coefficient `0.002227`, |coef| `0.002227`
- `lag_00__CT_damage_last_5s`: coefficient `0.001676`, |coef| `0.001676`
- `lag_11__T1__is_scoped`: coefficient `-0.001506`, |coef| `0.001506`
- `lag_09__CT_place_LONGDOORS`: coefficient `0.001462`, |coef| `0.001462`
- `lag_03__CT_place_HOLE`: coefficient `0.001443`, |coef| `0.001443`
- `lag_12__CT_place_UNDERA`: coefficient `0.001421`, |coef| `0.001421`
- `lag_11__CT_place_BDOORS`: coefficient `0.001410`, |coef| `0.001410`
- `lag_00__T2__flash`: coefficient `-0.001364`, |coef| `0.001364`
- `lag_01__CT_damage_last_5s`: coefficient `0.001360`, |coef| `0.001360`
- `lag_00__CT4__duck_amount`: coefficient `0.001357`, |coef| `0.001357`
- `lag_15__CT3__duck_amount`: coefficient `-0.001320`, |coef| `0.001320`
- `lag_02__CT_place_HOLE`: coefficient `0.001276`, |coef| `0.001276`

## Top 10 utility ridge features

- `lag_00__T2__flash`: coefficient `-0.001364` (lowers CT win probability)
- `lag_00__T_flash_inv`: coefficient `-0.000924` (lowers CT win probability)
- `lag_06__T5__molly`: coefficient `-0.000896` (lowers CT win probability)
- `lag_00__T2__utility_total`: coefficient `-0.000884` (lowers CT win probability)
- `lag_00__flash_inv_diff`: coefficient `0.000840` (raises CT win probability)
- `lag_03__T_B_site_active_infernos`: coefficient `0.000822` (raises CT win probability)
- `lag_02__T3__molly`: coefficient `-0.000783` (lowers CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `0.000705` (raises CT win probability)
- `lag_00__T_utility_inv`: coefficient `-0.000699` (lowers CT win probability)
- `lag_03__T5__flash`: coefficient `-0.000665` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_MIDDOORS`: coefficient `-0.003149` (lowers CT win probability)
- `lag_04__CT_place_HOLE`: coefficient `0.002520` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002511` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002227` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001676` (raises CT win probability)
- `lag_11__T1__is_scoped`: coefficient `-0.001506` (lowers CT win probability)
- `lag_09__CT_place_LONGDOORS`: coefficient `0.001462` (raises CT win probability)
- `lag_03__CT_place_HOLE`: coefficient `0.001443` (raises CT win probability)
- `lag_12__CT_place_UNDERA`: coefficient `0.001421` (raises CT win probability)
- `lag_11__CT_place_BDOORS`: coefficient `0.001410` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `153168`, seconds `70.50`, LSTM delta `+0.3833`

Top all feature movements:
- `lag_04__CT_place_HOLE`: contribution `+0.028132`
- `lag_00__T_place_MIDDOORS`: contribution `+0.026767`
- `lag_00__CT_kills_last_3s`: contribution `+0.014499`
- `lag_00__kill_diff_last_3s`: contribution `+0.010719`
- `lag_11__T1__is_scoped`: contribution `+0.008604`

Top utility-only movements:
- `lag_00__T2__flash`: contribution `+0.004017`

### tick `153616`, seconds `77.50`, LSTM delta `+0.1158`

Top all feature movements:
- `lag_00__T_place_MIDDOORS`: contribution `+0.013383`
- `lag_00__CT_kills_last_3s`: contribution `+0.007249`
- `lag_00__kill_diff_last_3s`: contribution `+0.005360`
- `lag_05__CT_place_HOLE`: contribution `-0.005333`
- `lag_12__T1__is_scoped`: contribution `+0.004496`

Top utility-only movements:
- `lag_03__T_B_site_active_infernos`: contribution `-0.002323`

### tick `153104`, seconds `69.50`, LSTM delta `+0.1033`

Top all feature movements:
- `lag_02__CT_place_HOLE`: contribution `+0.014244`
- `lag_00__T_place_MIDDOORS`: contribution `+0.013383`
- `lag_00__CT_kills_last_3s`: contribution `+0.007249`
- `lag_00__kill_diff_last_3s`: contribution `+0.005360`
- `lag_09__T1__is_scoped`: contribution `+0.005147`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `153136`, seconds `70.00`, LSTM delta `+0.0697`

Top all feature movements:
- `lag_03__CT_place_HOLE`: contribution `+0.016114`
- `lag_10__T1__is_scoped`: contribution `+0.005714`
- `lag_01__T_place_MIDDOORS`: contribution `+0.004704`
- `lag_14__CT3__duck_amount`: contribution `+0.003612`
- `lag_10__CT_place_BDOORS`: contribution `+0.003504`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `153456`, seconds `75.00`, LSTM delta `-0.0643`

Top all feature movements:
- `lag_08__CT_place_HOLE`: contribution `-0.007405`
- `lag_00__kill_diff_last_3s`: contribution `-0.005360`
- `lag_01__T1__is_scoped`: contribution `-0.004591`
- `lag_13__CT_place_HOLE`: contribution `-0.004217`
- `lag_03__CT_place_OUTSIDELONG`: contribution `-0.003835`

Top utility-only movements:
- No utility movement among the top local contributors.
