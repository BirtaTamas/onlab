# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-vitality-vs-the-mongolz-bo3-JVS9HKMAkaZTRHkoiRSMP6/vitality-vs-the-mongolz-m1-mirage.csv`
- round_num: `6`

## Largest probability jumps

- tick `39273`, seconds `75.50`, LSTM `0.1130`, delta `-0.3822`
- tick `40233`, seconds `90.50`, LSTM `0.6345`, delta `+0.2651`
- tick `39209`, seconds `74.50`, LSTM `0.5911`, delta `+0.2085`
- tick `39401`, seconds `77.50`, LSTM `0.2975`, delta `+0.2021`
- tick `38089`, seconds `57.00`, LSTM `0.3334`, delta `-0.1746`
- tick `38633`, seconds `65.50`, LSTM `0.7086`, delta `+0.1719`
- tick `38985`, seconds `71.00`, LSTM `0.3991`, delta `-0.1683`
- tick `39177`, seconds `74.00`, LSTM `0.3826`, delta `+0.1656`
- tick `38825`, seconds `68.50`, LSTM `0.5675`, delta `-0.1064`
- tick `39465`, seconds `78.50`, LSTM `0.3362`, delta `+0.0999`

## Top 15 local ridge features

- `lag_11__T_place_JUNGLE`: coefficient `0.005031`, |coef| `0.005031`
- `lag_00__kill_diff_last_3s`: coefficient `0.003868`, |coef| `0.003868`
- `lag_00__T_kills_last_3s`: coefficient `-0.003204`, |coef| `0.003204`
- `lag_00__T1__is_scoped`: coefficient `0.003200`, |coef| `0.003200`
- `lag_10__T_place_JUNGLE`: coefficient `0.002873`, |coef| `0.002873`
- `lag_00__CT_place_JUNGLE`: coefficient `0.002671`, |coef| `0.002671`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002569`, |coef| `0.002569`
- `lag_00__damage_diff_last_5s`: coefficient `0.002493`, |coef| `0.002493`
- `lag_15__T_place_STAIRS`: coefficient `-0.002455`, |coef| `0.002455`
- `lag_01__CT_shots_fired_sum`: coefficient `0.002405`, |coef| `0.002405`
- `lag_02__T_place_STAIRS`: coefficient `0.002363`, |coef| `0.002363`
- `lag_01__CT1__duck_amount`: coefficient `-0.001902`, |coef| `0.001902`
- `lag_05__T_bomb_zone_count`: coefficient `-0.001747`, |coef| `0.001747`
- `lag_05__T_duck_amount_mean`: coefficient `-0.001741`, |coef| `0.001741`
- `lag_00__CT_kills_last_3s`: coefficient `0.001720`, |coef| `0.001720`

## Top 10 utility ridge features

- `lag_01__T_A_site_active_infernos`: coefficient `-0.001351` (lowers CT win probability)
- `lag_02__T_A_site_active_infernos`: coefficient `-0.001190` (lowers CT win probability)
- `lag_10__T_A_site_active_infernos`: coefficient `-0.001171` (lowers CT win probability)
- `lag_00__T_A_site_active_infernos`: coefficient `-0.000993` (lowers CT win probability)
- `lag_01__T_active_infernos`: coefficient `-0.000952` (lowers CT win probability)
- `lag_00__CT_smokes_last_5s`: coefficient `0.000933` (raises CT win probability)
- `lag_00__smoke_inv_diff`: coefficient `0.000860` (raises CT win probability)
- `lag_07__T_A_site_active_infernos`: coefficient `0.000849` (raises CT win probability)
- `lag_02__T_active_infernos`: coefficient `-0.000845` (lowers CT win probability)
- `lag_11__T_A_site_active_infernos`: coefficient `-0.000842` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_11__T_place_JUNGLE`: coefficient `0.005031` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003868` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003204` (lowers CT win probability)
- `lag_00__T1__is_scoped`: coefficient `0.003200` (raises CT win probability)
- `lag_10__T_place_JUNGLE`: coefficient `0.002873` (raises CT win probability)
- `lag_00__CT_place_JUNGLE`: coefficient `0.002671` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002569` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002493` (raises CT win probability)
- `lag_15__T_place_STAIRS`: coefficient `-0.002455` (lowers CT win probability)
- `lag_01__CT_shots_fired_sum`: coefficient `0.002405` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `39273`, seconds `75.50`, LSTM delta `-0.3822`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.020301`
- `lag_00__kill_diff_last_3s`: contribution `-0.018621`
- `lag_00__T1__is_scoped`: contribution `-0.018281`
- `lag_00__CT_place_JUNGLE`: contribution `-0.017137`
- `lag_01__CT_shots_fired_sum`: contribution `-0.016705`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `40233`, seconds `90.50`, LSTM delta `+0.2651`

Top all feature movements:
- `lag_11__T_place_JUNGLE`: contribution `+0.065167`
- `lag_00__kill_diff_last_3s`: contribution `+0.009311`
- `lag_15__T_bomb_zone_count`: contribution `+0.008754`
- `lag_01__CT1__duck_amount`: contribution `+0.007256`
- `lag_03__T_bomb_zone_count`: contribution `-0.006768`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `39209`, seconds `74.50`, LSTM delta `+0.2085`

Top all feature movements:
- `lag_15__T_place_SCAFFOLDING`: contribution `+0.031485`
- `lag_00__T1__is_scoped`: contribution `+0.018281`
- `lag_05__T_bomb_zone_count`: contribution `+0.010168`
- `lag_00__kill_diff_last_3s`: contribution `+0.009311`
- `lag_00__CT_shots_fired_sum`: contribution `+0.008923`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `39401`, seconds `77.50`, LSTM delta `+0.2021`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.008923`
- `lag_10__CT_place_SHOP`: contribution `+0.007374`
- `lag_03__T_shots_fired_sum`: contribution `+0.006905`
- `lag_13__T1__is_scoped`: contribution `+0.005939`
- `lag_05__CT5__shots_fired`: contribution `+0.005310`

Top utility-only movements:
- `lag_00__T_A_site_active_infernos`: contribution `+0.002955`

### tick `38089`, seconds `57.00`, LSTM delta `-0.1746`

Top all feature movements:
- `lag_00__T_place_SCAFFOLDING`: contribution `-0.055646`
- `lag_02__T_place_SCAFFOLDING`: contribution `-0.029227`
- `lag_03__CT_place_JUNGLE`: contribution `-0.007860`
- `lag_00__T_macro_A`: contribution `-0.005356`
- `lag_00__T_place_BOMBSITEA`: contribution `-0.005356`

Top utility-only movements:
- `lag_01__T_A_site_active_infernos`: contribution `-0.004021`
- `lag_02__T_A_site_active_infernos`: contribution `-0.003541`
- `lag_05__CT2__flash_duration`: contribution `-0.002244`
- `lag_01__T_active_infernos`: contribution `-0.001983`
- `lag_02__T_active_infernos`: contribution `-0.001760`
