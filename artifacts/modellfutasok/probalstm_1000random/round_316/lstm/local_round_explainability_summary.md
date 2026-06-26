# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-mouz-vs-virtuspro-bo3-RgsQGjmI__aLZMP1KntvtG/mouz-vs-virtus-pro-m2-mirage.csv`
- round_num: `4`

## Largest probability jumps

- tick `32987`, seconds `120.50`, LSTM `0.7266`, delta `+0.4202`
- tick `31771`, seconds `101.50`, LSTM `0.1002`, delta `-0.2502`
- tick `28603`, seconds `52.00`, LSTM `0.2672`, delta `-0.2287`
- tick `32219`, seconds `108.50`, LSTM `0.3232`, delta `+0.2018`
- tick `33083`, seconds `122.00`, LSTM `0.6098`, delta `-0.1978`
- tick `29755`, seconds `70.00`, LSTM `0.5374`, delta `+0.1802`
- tick `30395`, seconds `80.00`, LSTM `0.3113`, delta `-0.1263`
- tick `30491`, seconds `81.50`, LSTM `0.4048`, delta `+0.1147`
- tick `29659`, seconds `68.50`, LSTM `0.3082`, delta `+0.0846`
- tick `31707`, seconds `100.50`, LSTM `0.3531`, delta `+0.0733`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004696`, |coef| `0.004696`
- `lag_00__T4__shots_fired`: coefficient `0.003992`, |coef| `0.003992`
- `lag_01__T4__shots_fired`: coefficient `0.003731`, |coef| `0.003731`
- `lag_00__damage_diff_last_5s`: coefficient `0.003624`, |coef| `0.003624`
- `lag_00__CT_shots_fired_sum`: coefficient `0.003535`, |coef| `0.003535`
- `lag_10__CT1__is_walking`: coefficient `0.003481`, |coef| `0.003481`
- `lag_00__CT_kills_last_3s`: coefficient `0.003443`, |coef| `0.003443`
- `lag_06__CT5__duck_amount`: coefficient `0.003437`, |coef| `0.003437`
- `lag_01__CT_place_SHOP`: coefficient `-0.003342`, |coef| `0.003342`
- `lag_10__T3__duck_amount`: coefficient `-0.003279`, |coef| `0.003279`
- `lag_00__T_macro_B`: coefficient `-0.003018`, |coef| `0.003018`
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.003018`, |coef| `0.003018`
- `lag_03__CT1__is_walking`: coefficient `-0.002973`, |coef| `0.002973`
- `lag_15__CT2__is_scoped`: coefficient `0.002719`, |coef| `0.002719`
- `lag_00__CT_damage_last_5s`: coefficient `0.002624`, |coef| `0.002624`

## Top 10 utility ridge features

- `lag_11__T_B_site_active_infernos`: coefficient `-0.001714` (lowers CT win probability)
- `lag_11__T_active_infernos`: coefficient `-0.001379` (lowers CT win probability)
- `lag_00__T4__flash`: coefficient `-0.001310` (lowers CT win probability)
- `lag_10__CT_B_site_active_infernos`: coefficient `-0.001297` (lowers CT win probability)
- `lag_11__active_infernos_total`: coefficient `-0.001067` (lowers CT win probability)
- `lag_00__CT_B_site_active_infernos`: coefficient `0.001053` (raises CT win probability)
- `lag_13__CT3__molly`: coefficient `0.001048` (raises CT win probability)
- `lag_00__T1__utility_total`: coefficient `-0.001000` (lowers CT win probability)
- `lag_05__T_B_site_active_infernos`: coefficient `0.000943` (raises CT win probability)
- `lag_00__T1__molly`: coefficient `-0.000942` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004696` (raises CT win probability)
- `lag_00__T4__shots_fired`: coefficient `0.003992` (raises CT win probability)
- `lag_01__T4__shots_fired`: coefficient `0.003731` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003624` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.003535` (raises CT win probability)
- `lag_10__CT1__is_walking`: coefficient `0.003481` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003443` (raises CT win probability)
- `lag_06__CT5__duck_amount`: coefficient `0.003437` (raises CT win probability)
- `lag_01__CT_place_SHOP`: coefficient `-0.003342` (lowers CT win probability)
- `lag_10__T3__duck_amount`: coefficient `-0.003279` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `32987`, seconds `120.50`, LSTM delta `+0.4202`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.019648`
- `lag_01__CT_place_SHOP`: contribution `+0.016765`
- `lag_15__CT2__is_scoped`: contribution `+0.016641`
- `lag_06__CT5__duck_amount`: contribution `+0.012975`
- `lag_10__T3__duck_amount`: contribution `+0.012363`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `31771`, seconds `101.50`, LSTM delta `-0.2502`

Top all feature movements:
- `lag_02__T_place_SNIPERSNEST`: contribution `-0.034492`
- `lag_00__kill_diff_last_3s`: contribution `-0.011302`
- `lag_00__CT_shots_fired_sum`: contribution `-0.009824`
- `lag_09__CT_place_SHOP`: contribution `-0.009792`
- `lag_09__CT2__duck_amount`: contribution `-0.008995`

Top utility-only movements:
- `lag_11__T_B_site_active_infernos`: contribution `-0.004846`
- `lag_10__CT_B_site_active_infernos`: contribution `-0.004455`

### tick `28603`, seconds `52.00`, LSTM delta `-0.2287`

Top all feature movements:
- `lag_15__CT_place_JUNGLE`: contribution `-0.015415`
- `lag_00__kill_diff_last_3s`: contribution `-0.011302`
- `lag_12__CT_place_JUNGLE`: contribution `-0.011195`
- `lag_01__CT2__is_scoped`: contribution `-0.009861`
- `lag_00__damage_diff_last_5s`: contribution `-0.008175`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `32219`, seconds `108.50`, LSTM delta `+0.2018`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.012280`
- `lag_12__CT2__is_scoped`: contribution `+0.008957`
- `lag_05__T_bomb_zone_count`: contribution `+0.008589`
- `lag_06__T_place_CTSPAWN`: contribution `+0.006251`
- `lag_00__T_shots_fired_sum`: contribution `+0.005880`

Top utility-only movements:
- `lag_11__T_B_site_active_infernos`: contribution `+0.004846`
- `lag_11__T_active_infernos`: contribution `+0.002871`

### tick `33083`, seconds `122.00`, LSTM delta `-0.1978`

Top all feature movements:
- `lag_02__CT_shots_fired_sum`: contribution `-0.017264`
- `lag_10__T3__duck_amount`: contribution `-0.012363`
- `lag_00__kill_diff_last_3s`: contribution `-0.011302`
- `lag_10__CT2__is_scoped`: contribution `-0.009696`
- `lag_00__T_kills_last_3s`: contribution `-0.007612`

Top utility-only movements:
- No utility movement among the top local contributors.
