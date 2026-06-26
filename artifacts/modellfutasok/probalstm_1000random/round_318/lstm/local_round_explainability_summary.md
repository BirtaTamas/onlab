# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-flyquest-vs-mibr-bo3-qPrK-wzQgATa8KQ5HjYeOS/flyquest-vs-mibr-m1-nuke.csv`
- round_num: `16`

## Largest probability jumps

- tick `146427`, seconds `47.00`, LSTM `0.9409`, delta `+0.1409`
- tick `145851`, seconds `38.00`, LSTM `0.6999`, delta `-0.1393`
- tick `145659`, seconds `35.00`, LSTM `0.6334`, delta `+0.1338`
- tick `145755`, seconds `36.50`, LSTM `0.8387`, delta `+0.1131`
- tick `145691`, seconds `35.50`, LSTM `0.7159`, delta `+0.0824`
- tick `146395`, seconds `46.50`, LSTM `0.8000`, delta `+0.0742`
- tick `146907`, seconds `54.50`, LSTM `0.9635`, delta `+0.0721`
- tick `145979`, seconds `40.00`, LSTM `0.6626`, delta `-0.0411`
- tick `146075`, seconds `41.50`, LSTM `0.6296`, delta `-0.0340`
- tick `146139`, seconds `42.50`, LSTM `0.6561`, delta `+0.0294`

## Top 15 local ridge features

- `lag_00__CT_place_MINI`: coefficient `0.001666`, |coef| `0.001666`
- `lag_00__CT_kills_last_3s`: coefficient `0.001578`, |coef| `0.001578`
- `lag_12__T4__flash_duration`: coefficient `-0.001536`, |coef| `0.001536`
- `lag_00__kill_diff_last_3s`: coefficient `0.001374`, |coef| `0.001374`
- `lag_11__T4__flash_duration`: coefficient `-0.001274`, |coef| `0.001274`
- `lag_05__T_place_CONTROL`: coefficient `0.001215`, |coef| `0.001215`
- `lag_00__T_place_CONTROL`: coefficient `-0.001201`, |coef| `0.001201`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001148`, |coef| `0.001148`
- `lag_04__T_place_CONTROL`: coefficient `0.001094`, |coef| `0.001094`
- `lag_00__damage_diff_last_5s`: coefficient `0.001033`, |coef| `0.001033`
- `lag_15__T_place_VENTS`: coefficient `0.000995`, |coef| `0.000995`
- `lag_11__T_place_VENDING`: coefficient `-0.000981`, |coef| `0.000981`
- `lag_00__CT_damage_last_5s`: coefficient `0.000980`, |coef| `0.000980`
- `lag_07__CT2__duck_amount`: coefficient `0.000955`, |coef| `0.000955`
- `lag_11__T_place_VENTS`: coefficient `0.000930`, |coef| `0.000930`

## Top 10 utility ridge features

- `lag_12__T4__flash_duration`: coefficient `-0.001536` (lowers CT win probability)
- `lag_11__T4__flash_duration`: coefficient `-0.001274` (lowers CT win probability)
- `lag_02__T4__flash_duration`: coefficient `0.000809` (raises CT win probability)
- `lag_15__CT5__flash_duration`: coefficient `-0.000787` (lowers CT win probability)
- `lag_12__T_flash_duration_sum`: coefficient `-0.000650` (lowers CT win probability)
- `lag_00__T4__molly`: coefficient `-0.000581` (lowers CT win probability)
- `lag_15__T_A_site_active_infernos`: coefficient `-0.000578` (lowers CT win probability)
- `lag_10__T4__flash_duration`: coefficient `-0.000566` (lowers CT win probability)
- `lag_15__CT_B_site_active_infernos`: coefficient `-0.000565` (lowers CT win probability)
- `lag_15__T_B_site_active_infernos`: coefficient `-0.000551` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_MINI`: coefficient `0.001666` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001578` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001374` (raises CT win probability)
- `lag_05__T_place_CONTROL`: coefficient `0.001215` (raises CT win probability)
- `lag_00__T_place_CONTROL`: coefficient `-0.001201` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001148` (raises CT win probability)
- `lag_04__T_place_CONTROL`: coefficient `0.001094` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001033` (raises CT win probability)
- `lag_15__T_place_VENTS`: coefficient `0.000995` (raises CT win probability)
- `lag_11__T_place_VENDING`: coefficient `-0.000981` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `146427`, seconds `47.00`, LSTM delta `+0.1409`

Top all feature movements:
- `lag_12__T4__flash_duration`: contribution `+0.011463`
- `lag_00__CT_place_MINI`: contribution `+0.010215`
- `lag_11__T_place_VENDING`: contribution `+0.004974`
- `lag_00__CT_kills_last_3s`: contribution `+0.004556`
- `lag_00__CT_shots_fired_sum`: contribution `+0.003987`

Top utility-only movements:
- `lag_12__T4__flash_duration`: contribution `+0.011463`
- `lag_12__T_flash_duration_sum`: contribution `+0.001972`
- `lag_15__CT_B_site_active_infernos`: contribution `+0.001941`
- `lag_15__CT_A_site_active_infernos`: contribution `+0.001861`
- `lag_15__T_A_site_active_infernos`: contribution `+0.001720`

### tick `145851`, seconds `38.00`, LSTM delta `-0.1393`

Top all feature movements:
- `lag_02__T_place_VENTS`: contribution `-0.010008`
- `lag_00__kill_diff_last_3s`: contribution `-0.006613`
- `lag_02__CT_place_ADMIN`: contribution `-0.006186`
- `lag_06__T_place_CONTROL`: contribution `-0.005113`
- `lag_00__CT_kills_last_3s`: contribution `-0.004556`

Top utility-only movements:
- `lag_08__T4__flash_duration`: contribution `-0.003584`
- `lag_03__T2__flash_duration`: contribution `-0.002881`

### tick `145659`, seconds `35.00`, LSTM delta `+0.1338`

Top all feature movements:
- `lag_00__T_place_CONTROL`: contribution `+0.008531`
- `lag_04__T_place_CONTROL`: contribution `+0.007772`
- `lag_02__T4__flash_duration`: contribution `+0.006033`
- `lag_01__CT_place_ADMIN`: contribution `+0.004646`
- `lag_00__CT_kills_last_3s`: contribution `+0.004556`

Top utility-only movements:
- `lag_02__T4__flash_duration`: contribution `+0.006033`
- `lag_02__T_flash_duration_sum`: contribution `+0.002328`

### tick `145755`, seconds `36.50`, LSTM delta `+0.1131`

Top all feature movements:
- `lag_05__T_place_CONTROL`: contribution `+0.008637`
- `lag_00__T_place_CONTROL`: contribution `+0.008531`
- `lag_07__T_place_CONTROL`: contribution `+0.005286`
- `lag_00__CT_kills_last_3s`: contribution `+0.004556`
- `lag_03__T_place_CONTROL`: contribution `-0.004291`

Top utility-only movements:
- `lag_05__T4__flash_duration`: contribution `+0.003029`
- `lag_02__T2__flash_duration`: contribution `+0.002272`
- `lag_02__CT5__flash_duration`: contribution `+0.001574`

### tick `145691`, seconds `35.50`, LSTM delta `+0.0824`

Top all feature movements:
- `lag_05__T_place_CONTROL`: contribution `+0.008637`
- `lag_02__CT_place_ADMIN`: contribution `+0.006186`
- `lag_03__T_place_CONTROL`: contribution `+0.004291`
- `lag_07__CT2__duck_amount`: contribution `+0.003302`
- `lag_03__T4__flash_duration`: contribution `+0.003265`

Top utility-only movements:
- `lag_03__T4__flash_duration`: contribution `+0.003265`
- `lag_00__CT5__flash_duration`: contribution `+0.001764`
