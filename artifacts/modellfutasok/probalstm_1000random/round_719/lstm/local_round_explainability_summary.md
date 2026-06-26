# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-astralis-vs-natus-vincere-bo3-4-6Sb81TUo41h9OxcK0xKz/astralis-vs-natus-vincere-m3-nuke.csv`
- round_num: `5`

## Largest probability jumps

- tick `54359`, seconds `38.00`, LSTM `0.0374`, delta `-0.2173`
- tick `53335`, seconds `22.00`, LSTM `0.2381`, delta `-0.0866`
- tick `51959`, seconds `0.50`, LSTM `0.1829`, delta `-0.0691`
- tick `53911`, seconds `31.00`, LSTM `0.2507`, delta `-0.0484`
- tick `53591`, seconds `26.00`, LSTM `0.2012`, delta `-0.0456`
- tick `53207`, seconds `20.00`, LSTM `0.3153`, delta `+0.0418`
- tick `53847`, seconds `30.00`, LSTM `0.2797`, delta `+0.0415`
- tick `54263`, seconds `36.50`, LSTM `0.2072`, delta `+0.0413`
- tick `54295`, seconds `37.00`, LSTM `0.2481`, delta `+0.0409`
- tick `54199`, seconds `35.50`, LSTM `0.1709`, delta `-0.0352`

## Top 15 local ridge features

- `lag_15__CT_place_HUT`: coefficient `-0.001590`, |coef| `0.001590`
- `lag_06__T_place_GARAGE`: coefficient `-0.001216`, |coef| `0.001216`
- `lag_00__CT_place_LOCKERROOM`: coefficient `0.001151`, |coef| `0.001151`
- `lag_00__CT_place_SQUEAKY`: coefficient `-0.001067`, |coef| `0.001067`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001030`, |coef| `0.001030`
- `lag_04__CT_place_SQUEAKY`: coefficient `0.000913`, |coef| `0.000913`
- `lag_03__CT_place_LOCKERROOM`: coefficient `-0.000906`, |coef| `0.000906`
- `lag_00__CT_place_VENTS`: coefficient `-0.000802`, |coef| `0.000802`
- `lag_02__CT_place_SQUEAKY`: coefficient `-0.000801`, |coef| `0.000801`
- `lag_12__CT_place_HUT`: coefficient `0.000772`, |coef| `0.000772`
- `lag_02__T_place_GARAGE`: coefficient `-0.000768`, |coef| `0.000768`
- `lag_01__T_place_GARAGE`: coefficient `-0.000755`, |coef| `0.000755`
- `lag_06__T_shots_fired_sum`: coefficient `0.000715`, |coef| `0.000715`
- `lag_00__CT_place_MINI`: coefficient `-0.000688`, |coef| `0.000688`
- `lag_00__T3__is_scoped`: coefficient `0.000674`, |coef| `0.000674`

## Top 10 utility ridge features

- `lag_10__T5__flash_duration`: coefficient `-0.000457` (lowers CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000380` (raises CT win probability)
- `lag_10__T_A_site_active_infernos`: coefficient `0.000344` (raises CT win probability)
- `lag_01__T_smoke_inv`: coefficient `-0.000332` (lowers CT win probability)
- `lag_10__T_B_site_active_infernos`: coefficient `0.000328` (raises CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000327` (raises CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000308` (raises CT win probability)
- `lag_01__T_molly_inv`: coefficient `-0.000301` (lowers CT win probability)
- `lag_03__T5__flash_duration`: coefficient `0.000299` (raises CT win probability)
- `lag_10__T_flash_duration_sum`: coefficient `-0.000292` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_15__CT_place_HUT`: coefficient `-0.001590` (lowers CT win probability)
- `lag_06__T_place_GARAGE`: coefficient `-0.001216` (lowers CT win probability)
- `lag_00__CT_place_LOCKERROOM`: coefficient `0.001151` (raises CT win probability)
- `lag_00__CT_place_SQUEAKY`: coefficient `-0.001067` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001030` (raises CT win probability)
- `lag_04__CT_place_SQUEAKY`: coefficient `0.000913` (raises CT win probability)
- `lag_03__CT_place_LOCKERROOM`: coefficient `-0.000906` (lowers CT win probability)
- `lag_00__CT_place_VENTS`: coefficient `-0.000802` (lowers CT win probability)
- `lag_02__CT_place_SQUEAKY`: coefficient `-0.000801` (lowers CT win probability)
- `lag_12__CT_place_HUT`: coefficient `0.000772` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `54359`, seconds `38.00`, LSTM delta `-0.2173`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.018612`
- `lag_15__CT_place_HUT`: contribution `-0.015504`
- `lag_06__T_place_GARAGE`: contribution `-0.014618`
- `lag_00__CT_place_LOCKERROOM`: contribution `-0.014330`
- `lag_04__CT_place_SQUEAKY`: contribution `-0.012138`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `53335`, seconds `22.00`, LSTM delta `-0.0866`

Top all feature movements:
- `lag_15__CT_place_HUT`: contribution `-0.015504`
- `lag_00__CT_place_SQUEAKY`: contribution `-0.014186`
- `lag_06__T_shots_fired_sum`: contribution `-0.006436`
- `lag_06__T2__shots_fired`: contribution `-0.004443`
- `lag_09__T2__duck_amount`: contribution `-0.001274`

Top utility-only movements:
- `lag_10__T_A_site_active_infernos`: contribution `-0.001025`
- `lag_10__T_B_site_active_infernos`: contribution `-0.000926`

### tick `51959`, seconds `0.50`, LSTM delta `-0.0691`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.003047`
- `lag_01__T_place_TSPAWN`: contribution `-0.002578`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.002418`
- `lag_01__T_closest_enemy_dist`: contribution `-0.002382`
- `lag_00__CT_velocity_mean`: contribution `-0.002379`

Top utility-only movements:
- `lag_01__molly_inv_diff`: contribution `-0.001059`
- `lag_01__CT1__flash`: contribution `-0.000812`
- `lag_01__T_smoke_inv`: contribution `-0.000756`
- `lag_01__utility_inv_diff`: contribution `-0.000717`
- `lag_01__T_molly_inv`: contribution `-0.000682`

### tick `53911`, seconds `31.00`, LSTM delta `-0.0484`

Top all feature movements:
- `lag_06__CT_place_TROPHY`: contribution `-0.007251`
- `lag_10__CT_place_CONTROL`: contribution `-0.006758`
- `lag_10__CT_place_TROPHY`: contribution `-0.004094`
- `lag_02__CT_place_CONTROL`: contribution `-0.003090`
- `lag_13__CT_place_LOBBY`: contribution `-0.002576`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `53591`, seconds `26.00`, LSTM delta `-0.0456`

Top all feature movements:
- `lag_00__CT_place_TROPHY`: contribution `-0.006052`
- `lag_08__CT_place_SQUEAKY`: contribution `-0.005435`
- `lag_00__CT_place_MINI`: contribution `-0.004220`
- `lag_03__CT_place_HUT`: contribution `-0.003594`
- `lag_03__CT_place_LOBBY`: contribution `-0.001262`

Top utility-only movements:
- No utility movement among the top local contributors.
