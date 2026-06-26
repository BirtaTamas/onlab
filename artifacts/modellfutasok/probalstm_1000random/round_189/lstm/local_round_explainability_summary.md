# Local Round Explainability

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-tyloo-vs-saw-bo3-Ik7_qO98IbLkRUwtJ3asIP/tyloo-vs-saw-m1-nuke.csv`
- round_num: `22`

## Largest probability jumps

- tick `174713`, seconds `43.00`, LSTM `0.0377`, delta `-0.1436`
- tick `171993`, seconds `0.50`, LSTM `0.0726`, delta `-0.0695`
- tick `174041`, seconds `32.50`, LSTM `0.1603`, delta `-0.0427`
- tick `173817`, seconds `29.00`, LSTM `0.2252`, delta `+0.0424`
- tick `174649`, seconds `42.00`, LSTM `0.2007`, delta `+0.0423`
- tick `172857`, seconds `14.00`, LSTM `0.1733`, delta `+0.0391`
- tick `172953`, seconds `15.50`, LSTM `0.1186`, delta `-0.0387`
- tick `172697`, seconds `11.50`, LSTM `0.1541`, delta `+0.0381`
- tick `173113`, seconds `18.00`, LSTM `0.1407`, delta `+0.0358`
- tick `172825`, seconds `13.50`, LSTM `0.1343`, delta `-0.0349`

## Top 15 local ridge features

- `lag_00__CT_place_SECRET`: coefficient `-0.001217`, |coef| `0.001217`
- `lag_04__CT_place_MINI`: coefficient `0.000979`, |coef| `0.000979`
- `lag_03__CT_place_MINI`: coefficient `0.000952`, |coef| `0.000952`
- `lag_00__CT_place_LOCKERROOM`: coefficient `0.000943`, |coef| `0.000943`
- `lag_10__CT_place_HELL`: coefficient `0.000933`, |coef| `0.000933`
- `lag_00__CT1__duck_amount`: coefficient `-0.000893`, |coef| `0.000893`
- `lag_00__T_shots_fired_sum`: coefficient `-0.000830`, |coef| `0.000830`
- `lag_07__T_place_SECRET`: coefficient `-0.000742`, |coef| `0.000742`
- `lag_10__CT_place_LOCKERROOM`: coefficient `-0.000721`, |coef| `0.000721`
- `lag_03__CT_place_LOCKERROOM`: coefficient `-0.000701`, |coef| `0.000701`
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000660`, |coef| `0.000660`
- `lag_00__CT_velocity_mean`: coefficient `-0.000653`, |coef| `0.000653`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000646`, |coef| `0.000646`
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000606`, |coef| `0.000606`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000593`, |coef| `0.000593`

## Top 10 utility ridge features

- `lag_13__T_A_site_active_infernos`: coefficient `0.000376` (raises CT win probability)
- `lag_13__T_B_site_active_infernos`: coefficient `0.000357` (raises CT win probability)
- `lag_01__T1__utility_total`: coefficient `-0.000356` (lowers CT win probability)
- `lag_13__T_active_infernos`: coefficient `0.000335` (raises CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000324` (raises CT win probability)
- `lag_14__CT1__flash_duration`: coefficient `-0.000321` (lowers CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000313` (raises CT win probability)
- `lag_01__T1__flash`: coefficient `-0.000306` (lowers CT win probability)
- `lag_01__T_smoke_inv`: coefficient `-0.000285` (lowers CT win probability)
- `lag_05__T3__flash_duration`: coefficient `0.000276` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_SECRET`: coefficient `-0.001217` (lowers CT win probability)
- `lag_04__CT_place_MINI`: coefficient `0.000979` (raises CT win probability)
- `lag_03__CT_place_MINI`: coefficient `0.000952` (raises CT win probability)
- `lag_00__CT_place_LOCKERROOM`: coefficient `0.000943` (raises CT win probability)
- `lag_10__CT_place_HELL`: coefficient `0.000933` (raises CT win probability)
- `lag_00__CT1__duck_amount`: coefficient `-0.000893` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.000830` (lowers CT win probability)
- `lag_07__T_place_SECRET`: coefficient `-0.000742` (lowers CT win probability)
- `lag_10__CT_place_LOCKERROOM`: coefficient `-0.000721` (lowers CT win probability)
- `lag_03__CT_place_LOCKERROOM`: coefficient `-0.000701` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `174713`, seconds `43.00`, LSTM delta `-0.1436`

Top all feature movements:
- `lag_00__CT_place_SECRET`: contribution `-0.012528`
- `lag_10__CT_place_LOCKERROOM`: contribution `-0.008976`
- `lag_04__CT_place_MINI`: contribution `-0.006004`
- `lag_03__CT_place_MINI`: contribution `-0.005835`
- `lag_00__T_shots_fired_sum`: contribution `-0.005597`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `171993`, seconds `0.50`, LSTM delta `-0.0695`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.003158`
- `lag_01__T_place_TSPAWN`: contribution `-0.002861`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.002632`
- `lag_01__T_closest_enemy_dist`: contribution `-0.002513`
- `lag_00__CT_velocity_mean`: contribution `-0.002318`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `-0.000895`
- `lag_01__smoke_inv_diff`: contribution `-0.000824`
- `lag_01__T1__utility_total`: contribution `-0.000802`
- `lag_01__T_smoke_inv`: contribution `-0.000650`
- `lag_01__T1__flash`: contribution `-0.000628`

### tick `174041`, seconds `32.50`, LSTM delta `-0.0427`

Top all feature movements:
- `lag_00__CT_place_LOCKERROOM`: contribution `-0.011739`
- `lag_07__CT_place_LOCKERROOM`: contribution `-0.007079`
- `lag_03__CT_place_MINI`: contribution `+0.005835`
- `lag_02__CT_place_LOCKERROOM`: contribution `-0.002990`
- `lag_07__CT_place_HELL`: contribution `+0.001841`

Top utility-only movements:
- `lag_14__CT1__flash_duration`: contribution `-0.001538`
- `lag_06__T3__flash_duration`: contribution `-0.000972`
- `lag_06__CT1__flash_duration`: contribution `-0.000903`
- `lag_14__T3__flash_duration`: contribution `-0.000896`

### tick `173817`, seconds `29.00`, LSTM delta `+0.0424`

Top all feature movements:
- `lag_00__CT_place_LOCKERROOM`: contribution `+0.011739`
- `lag_04__CT_place_ADMIN`: contribution `+0.003494`
- `lag_05__CT_place_MINI`: contribution `+0.002222`
- `lag_10__CT_place_ADMIN`: contribution `+0.001755`
- `lag_00__T4__duck_amount`: contribution `+0.001651`

Top utility-only movements:
- `lag_07__CT1__flash_duration`: contribution `+0.000851`
- `lag_07__T3__flash_duration`: contribution `+0.000571`

### tick `174649`, seconds `42.00`, LSTM delta `+0.0423`

Top all feature movements:
- `lag_10__CT_place_HELL`: contribution `+0.005058`
- `lag_14__CT_place_LOCKERROOM`: contribution `+0.004731`
- `lag_08__CT_place_LOCKERROOM`: contribution `-0.002553`
- `lag_10__T1__duck_amount`: contribution `+0.001661`
- `lag_01__T2__duck_amount`: contribution `+0.001363`

Top utility-only movements:
- No utility movement among the top local contributors.
