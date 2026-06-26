# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-liquid-bo3-pfm398EHUpu3zLY0TgcmxO/the-mongolz-vs-liquid-m3-ancient.csv`
- round_num: `11`

## Largest probability jumps

- tick `71290`, seconds `28.00`, LSTM `0.2222`, delta `-0.3362`
- tick `70554`, seconds `16.50`, LSTM `0.4963`, delta `-0.1213`
- tick `70650`, seconds `18.00`, LSTM `0.3258`, delta `-0.1191`
- tick `71258`, seconds `27.50`, LSTM `0.5584`, delta `+0.1027`
- tick `73690`, seconds `65.50`, LSTM `0.0490`, delta `-0.0973`
- tick `70970`, seconds `23.00`, LSTM `0.3366`, delta `-0.0857`
- tick `70874`, seconds `21.50`, LSTM `0.4515`, delta `+0.0658`
- tick `71450`, seconds `30.50`, LSTM `0.0861`, delta `-0.0544`
- tick `70682`, seconds `18.50`, LSTM `0.2717`, delta `-0.0542`
- tick `70810`, seconds `20.50`, LSTM `0.3456`, delta `+0.0495`

## Top 15 local ridge features

- `lag_00__CT_place_UNKNOWN`: coefficient `0.003700`, |coef| `0.003700`
- `lag_10__CT1__is_scoped`: coefficient `0.002676`, |coef| `0.002676`
- `lag_00__T_kills_last_3s`: coefficient `-0.002151`, |coef| `0.002151`
- `lag_05__T2__flash_duration`: coefficient `0.002050`, |coef| `0.002050`
- `lag_00__kill_diff_last_3s`: coefficient `0.002008`, |coef| `0.002008`
- `lag_00__CT_place_SIDEENTRANCE`: coefficient `0.001992`, |coef| `0.001992`
- `lag_10__T3__is_walking`: coefficient `-0.001731`, |coef| `0.001731`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001728`, |coef| `0.001728`
- `lag_00__CT5__flash_duration`: coefficient `0.001705`, |coef| `0.001705`
- `lag_00__CT_burning_players`: coefficient `0.001694`, |coef| `0.001694`
- `lag_08__CT5__flash_duration`: coefficient `-0.001588`, |coef| `0.001588`
- `lag_14__T2__flash_duration`: coefficient `-0.001537`, |coef| `0.001537`
- `lag_09__CT1__is_walking`: coefficient `-0.001504`, |coef| `0.001504`
- `lag_01__CT5__shots_fired`: coefficient `-0.001430`, |coef| `0.001430`
- `lag_04__T_place_MAINHALL`: coefficient `0.001425`, |coef| `0.001425`

## Top 10 utility ridge features

- `lag_05__T2__flash_duration`: coefficient `0.002050` (raises CT win probability)
- `lag_00__CT5__flash_duration`: coefficient `0.001705` (raises CT win probability)
- `lag_08__CT5__flash_duration`: coefficient `-0.001588` (lowers CT win probability)
- `lag_14__T2__flash_duration`: coefficient `-0.001537` (lowers CT win probability)
- `lag_00__CT5__utility_total`: coefficient `0.001319` (raises CT win probability)
- `lag_15__CT_A_site_active_infernos`: coefficient `-0.001283` (lowers CT win probability)
- `lag_00__CT5__molly`: coefficient `0.001200` (raises CT win probability)
- `lag_06__T2__flash_duration`: coefficient `0.001148` (raises CT win probability)
- `lag_11__CT_flashes_last_5s`: coefficient `0.001071` (raises CT win probability)
- `lag_15__CT_active_infernos`: coefficient `-0.001071` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_UNKNOWN`: coefficient `0.003700` (raises CT win probability)
- `lag_10__CT1__is_scoped`: coefficient `0.002676` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002151` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002008` (raises CT win probability)
- `lag_00__CT_place_SIDEENTRANCE`: coefficient `0.001992` (raises CT win probability)
- `lag_10__T3__is_walking`: coefficient `-0.001731` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001728` (raises CT win probability)
- `lag_00__CT_burning_players`: coefficient `0.001694` (raises CT win probability)
- `lag_09__CT1__is_walking`: coefficient `-0.001504` (lowers CT win probability)
- `lag_01__CT5__shots_fired`: coefficient `-0.001430` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `71290`, seconds `28.00`, LSTM delta `-0.3362`

Top all feature movements:
- `lag_10__CT1__is_scoped`: contribution `-0.011459`
- `lag_05__T2__flash_duration`: contribution `-0.010856`
- `lag_00__CT5__flash_duration`: contribution `-0.008282`
- `lag_00__CT_place_SIDEENTRANCE`: contribution `-0.008017`
- `lag_14__T2__flash_duration`: contribution `-0.007746`

Top utility-only movements:
- `lag_05__T2__flash_duration`: contribution `-0.010856`
- `lag_00__CT5__flash_duration`: contribution `-0.008282`
- `lag_14__T2__flash_duration`: contribution `-0.007746`
- `lag_08__CT5__flash_duration`: contribution `-0.007712`
- `lag_15__CT_A_site_active_infernos`: contribution `-0.004526`

### tick `70554`, seconds `16.50`, LSTM delta `-0.1213`

Top all feature movements:
- `lag_00__CT_place_SIDEENTRANCE`: contribution `-0.008017`
- `lag_11__T_place_TUNNEL`: contribution `-0.006910`
- `lag_00__T_kills_last_3s`: contribution `-0.006815`
- `lag_10__CT_utility_damage_last_5s`: contribution `-0.005329`
- `lag_00__kill_diff_last_3s`: contribution `-0.004833`

Top utility-only movements:
- `lag_10__CT_utility_damage_last_5s`: contribution `-0.005329`
- `lag_10__utility_damage_diff_last_5s`: contribution `-0.003500`
- `lag_00__CT_utility_damage_last_5s`: contribution `-0.003103`
- `lag_15__T2__flash_duration`: contribution `-0.002281`

### tick `70650`, seconds `18.00`, LSTM delta `-0.1191`

Top all feature movements:
- `lag_10__CT1__is_scoped`: contribution `-0.011459`
- `lag_06__T2__flash_duration`: contribution `-0.007581`
- `lag_14__T_place_TUNNEL`: contribution `-0.007513`
- `lag_00__CT1__is_scoped`: contribution `-0.006098`
- `lag_00__T_place_SIDEENTRANCE`: contribution `-0.004877`

Top utility-only movements:
- `lag_06__T2__flash_duration`: contribution `-0.007581`
- `lag_03__CT_utility_damage_last_5s`: contribution `-0.003793`
- `lag_03__utility_damage_diff_last_5s`: contribution `-0.002832`

### tick `71258`, seconds `27.50`, LSTM delta `+0.1027`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.006004`
- `lag_04__T2__flash_duration`: contribution `+0.005063`
- `lag_00__T_place_SIDEENTRANCE`: contribution `+0.004877`
- `lag_00__kill_diff_last_3s`: contribution `+0.004833`
- `lag_01__CT1__is_scoped`: contribution `+0.004500`

Top utility-only movements:
- `lag_04__T2__flash_duration`: contribution `+0.005063`
- `lag_07__CT5__flash_duration`: contribution `+0.002907`

### tick `73690`, seconds `65.50`, LSTM delta `-0.0973`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.006815`
- `lag_04__T_place_MAINHALL`: contribution `-0.005144`
- `lag_08__T2__duck_amount`: contribution `-0.004838`
- `lag_00__kill_diff_last_3s`: contribution `-0.004833`
- `lag_00__CT_burning_players`: contribution `-0.004350`

Top utility-only movements:
- `lag_03__T3__molly`: contribution `-0.001863`
