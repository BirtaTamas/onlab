# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-mouz-vs-faze-bo3-q02I_n27c_oaVV09Kplodn/mouz-vs-faze-m2-mirage.csv`
- round_num: `11`

## Largest probability jumps

- tick `80213`, seconds `38.50`, LSTM `0.3605`, delta `-0.4002`
- tick `79221`, seconds `23.00`, LSTM `0.3541`, delta `-0.2355`
- tick `79381`, seconds `25.50`, LSTM `0.5960`, delta `+0.1795`
- tick `80245`, seconds `39.00`, LSTM `0.1936`, delta `-0.1669`
- tick `79413`, seconds `26.00`, LSTM `0.7527`, delta `+0.1567`
- tick `80277`, seconds `39.50`, LSTM `0.0636`, delta `-0.1300`
- tick `79349`, seconds `25.00`, LSTM `0.4165`, delta `+0.1006`
- tick `82133`, seconds `68.50`, LSTM `0.0982`, delta `+0.0841`
- tick `79157`, seconds `22.00`, LSTM `0.5493`, delta `+0.0594`
- tick `80117`, seconds `37.00`, LSTM `0.7308`, delta `+0.0578`

## Top 15 local ridge features

- `lag_00__T_place_TRUCK`: coefficient `-0.003522`, |coef| `0.003522`
- `lag_14__CT_place_JUNGLE`: coefficient `0.002909`, |coef| `0.002909`
- `lag_03__CT_place_UNDERPASS`: coefficient `-0.002873`, |coef| `0.002873`
- `lag_00__CT_flashes_last_5s`: coefficient `0.002771`, |coef| `0.002771`
- `lag_00__kill_diff_last_3s`: coefficient `0.002701`, |coef| `0.002701`
- `lag_00__damage_diff_last_5s`: coefficient `0.002358`, |coef| `0.002358`
- `lag_00__T_kills_last_3s`: coefficient `-0.002309`, |coef| `0.002309`
- `lag_01__CT_flashes_last_5s`: coefficient `0.002273`, |coef| `0.002273`
- `lag_01__T_place_TRUCK`: coefficient `-0.002215`, |coef| `0.002215`
- `lag_05__CT_place_UNDERPASS`: coefficient `-0.002197`, |coef| `0.002197`
- `lag_14__CT2__is_scoped`: coefficient `0.002145`, |coef| `0.002145`
- `lag_04__CT_place_UNDERPASS`: coefficient `-0.002118`, |coef| `0.002118`
- `lag_07__T4__duck_amount`: coefficient `-0.002048`, |coef| `0.002048`
- `lag_00__T_damage_last_5s`: coefficient `-0.001951`, |coef| `0.001951`
- `lag_03__T2__flash_duration`: coefficient `-0.001879`, |coef| `0.001879`

## Top 10 utility ridge features

- `lag_00__CT_flashes_last_5s`: coefficient `0.002771` (raises CT win probability)
- `lag_01__CT_flashes_last_5s`: coefficient `0.002273` (raises CT win probability)
- `lag_03__T2__flash_duration`: coefficient `-0.001879` (lowers CT win probability)
- `lag_03__T4__flash_duration`: coefficient `-0.001786` (lowers CT win probability)
- `lag_03__T_flash_duration_sum`: coefficient `-0.001584` (lowers CT win probability)
- `lag_12__T5__flash_duration`: coefficient `0.001569` (raises CT win probability)
- `lag_03__CT_A_site_active_infernos`: coefficient `0.001561` (raises CT win probability)
- `lag_14__CT_A_site_active_infernos`: coefficient `-0.001516` (lowers CT win probability)
- `lag_12__T1__flash_duration`: coefficient `0.001472` (raises CT win probability)
- `lag_04__T4__flash_duration`: coefficient `-0.001260` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_TRUCK`: coefficient `-0.003522` (lowers CT win probability)
- `lag_14__CT_place_JUNGLE`: coefficient `0.002909` (raises CT win probability)
- `lag_03__CT_place_UNDERPASS`: coefficient `-0.002873` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002701` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002358` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002309` (lowers CT win probability)
- `lag_01__T_place_TRUCK`: coefficient `-0.002215` (lowers CT win probability)
- `lag_05__CT_place_UNDERPASS`: coefficient `-0.002197` (lowers CT win probability)
- `lag_14__CT2__is_scoped`: coefficient `0.002145` (raises CT win probability)
- `lag_04__CT_place_UNDERPASS`: coefficient `-0.002118` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `80213`, seconds `38.50`, LSTM delta `-0.4002`

Top all feature movements:
- `lag_00__T_place_TRUCK`: contribution `-0.061170`
- `lag_14__CT_place_JUNGLE`: contribution `-0.018662`
- `lag_03__CT_place_UNDERPASS`: contribution `-0.016661`
- `lag_14__CT2__is_scoped`: contribution `-0.013131`
- `lag_03__T2__flash_duration`: contribution `-0.008685`

Top utility-only movements:
- `lag_03__T2__flash_duration`: contribution `-0.008685`
- `lag_03__T4__flash_duration`: contribution `-0.008577`
- `lag_03__T_flash_duration_sum`: contribution `-0.006090`
- `lag_03__CT_A_site_active_infernos`: contribution `-0.005508`
- `lag_14__CT_A_site_active_infernos`: contribution `-0.005352`

### tick `79221`, seconds `23.00`, LSTM delta `-0.2355`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.009947`
- `lag_12__T5__flash_duration`: contribution `-0.008750`
- `lag_14__CT_place_TRUCK`: contribution `-0.008316`
- `lag_12__T1__flash_duration`: contribution `-0.007699`
- `lag_00__T_kills_last_3s`: contribution `-0.007314`

Top utility-only movements:
- `lag_12__T5__flash_duration`: contribution `-0.008750`
- `lag_12__T1__flash_duration`: contribution `-0.007699`
- `lag_14__CT_B_site_active_infernos`: contribution `-0.003679`

### tick `79381`, seconds `25.50`, LSTM delta `+0.1795`

Top all feature movements:
- `lag_00__CT_flashes_last_5s`: contribution `+0.030472`
- `lag_05__CT_place_UNDERPASS`: contribution `+0.012738`
- `lag_12__CT_place_UNDERPASS`: contribution `+0.007172`
- `lag_12__T_place_CONNECTOR`: contribution `+0.006614`
- `lag_00__kill_diff_last_3s`: contribution `+0.006502`

Top utility-only movements:
- `lag_00__CT_flashes_last_5s`: contribution `+0.030472`

### tick `80245`, seconds `39.00`, LSTM delta `-0.1669`

Top all feature movements:
- `lag_01__T_place_TRUCK`: contribution `-0.038459`
- `lag_04__CT_place_UNDERPASS`: contribution `-0.012282`
- `lag_15__CT2__is_scoped`: contribution `-0.010816`
- `lag_15__CT_place_JUNGLE`: contribution `-0.010807`
- `lag_15__T4__is_scoped`: contribution `-0.006424`

Top utility-only movements:
- `lag_04__T4__flash_duration`: contribution `-0.006051`
- `lag_04__T2__flash_duration`: contribution `-0.004777`
- `lag_04__CT_A_site_active_infernos`: contribution `-0.003672`
- `lag_04__T_flash_duration_sum`: contribution `-0.003346`
- `lag_15__CT_A_site_active_infernos`: contribution `-0.002973`

### tick `79413`, seconds `26.00`, LSTM delta `+0.1567`

Top all feature movements:
- `lag_01__CT_flashes_last_5s`: contribution `+0.024997`
- `lag_00__kill_diff_last_3s`: contribution `+0.013004`
- `lag_00__CT_shots_fired_sum`: contribution `-0.008842`
- `lag_00__T_kills_last_3s`: contribution `+0.007314`
- `lag_13__T_place_CONNECTOR`: contribution `+0.006216`

Top utility-only movements:
- `lag_01__CT_flashes_last_5s`: contribution `+0.024997`
