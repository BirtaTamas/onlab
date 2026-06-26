# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-faze-vs-virtuspro-bo3-YDlVsCnS6YPgcr85bBYoPq/faze-vs-virtus-pro-m2-mirage.csv`
- round_num: `12`

## Largest probability jumps

- tick `76612`, seconds `43.00`, LSTM `0.9327`, delta `+0.1264`
- tick `75172`, seconds `20.50`, LSTM `0.8626`, delta `+0.1030`
- tick `74756`, seconds `14.00`, LSTM `0.7702`, delta `+0.0828`
- tick `74692`, seconds `13.00`, LSTM `0.6729`, delta `+0.0704`
- tick `75684`, seconds `28.50`, LSTM `0.8221`, delta `-0.0542`
- tick `77220`, seconds `52.50`, LSTM `0.9765`, delta `+0.0359`
- tick `74788`, seconds `14.50`, LSTM `0.8035`, delta `+0.0332`
- tick `76068`, seconds `34.50`, LSTM `0.7900`, delta `-0.0294`
- tick `76004`, seconds `33.50`, LSTM `0.8089`, delta `+0.0258`
- tick `75012`, seconds `18.00`, LSTM `0.7657`, delta `-0.0255`

## Top 15 local ridge features

- `lag_02__T_place_CONNECTOR`: coefficient `0.001521`, |coef| `0.001521`
- `lag_00__kill_diff_last_3s`: coefficient `0.001432`, |coef| `0.001432`
- `lag_00__CT_kills_last_3s`: coefficient `0.001345`, |coef| `0.001345`
- `lag_00__damage_diff_last_5s`: coefficient `0.001169`, |coef| `0.001169`
- `lag_00__T3__flash`: coefficient `-0.001082`, |coef| `0.001082`
- `lag_00__T3__utility_total`: coefficient `-0.000949`, |coef| `0.000949`
- `lag_00__CT_damage_last_5s`: coefficient `0.000932`, |coef| `0.000932`
- `lag_15__T_burning_players`: coefficient `0.000915`, |coef| `0.000915`
- `lag_00__T_place_BACKALLEY`: coefficient `-0.000895`, |coef| `0.000895`
- `lag_00__T3__alive`: coefficient `-0.000888`, |coef| `0.000888`
- `lag_09__CT4__duck_amount`: coefficient `-0.000884`, |coef| `0.000884`
- `lag_06__CT3__is_scoped`: coefficient `0.000881`, |coef| `0.000881`
- `lag_05__CT_B_site_active_infernos`: coefficient `-0.000874`, |coef| `0.000874`
- `lag_00__T3__armor`: coefficient `-0.000832`, |coef| `0.000832`
- `lag_05__CT3__is_scoped`: coefficient `-0.000824`, |coef| `0.000824`

## Top 10 utility ridge features

- `lag_00__T3__flash`: coefficient `-0.001082` (lowers CT win probability)
- `lag_00__T3__utility_total`: coefficient `-0.000949` (lowers CT win probability)
- `lag_05__CT_B_site_active_infernos`: coefficient `-0.000874` (lowers CT win probability)
- `lag_00__T3__molly`: coefficient `-0.000815` (lowers CT win probability)
- `lag_05__T_B_site_active_infernos`: coefficient `-0.000813` (lowers CT win probability)
- `lag_05__active_infernos_total`: coefficient `-0.000787` (lowers CT win probability)
- `lag_14__T_he_last_5s`: coefficient `0.000737` (raises CT win probability)
- `lag_04__T_he_last_5s`: coefficient `-0.000724` (lowers CT win probability)
- `lag_05__T_active_infernos`: coefficient `-0.000605` (lowers CT win probability)
- `lag_05__CT_active_infernos`: coefficient `-0.000592` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_02__T_place_CONNECTOR`: coefficient `0.001521` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001432` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001345` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001169` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.000932` (raises CT win probability)
- `lag_15__T_burning_players`: coefficient `0.000915` (raises CT win probability)
- `lag_00__T_place_BACKALLEY`: coefficient `-0.000895` (lowers CT win probability)
- `lag_00__T3__alive`: coefficient `-0.000888` (lowers CT win probability)
- `lag_09__CT4__duck_amount`: coefficient `-0.000884` (lowers CT win probability)
- `lag_06__CT3__is_scoped`: coefficient `0.000881` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `76612`, seconds `43.00`, LSTM delta `+0.1264`

Top all feature movements:
- `lag_02__T_place_CONNECTOR`: contribution `+0.007367`
- `lag_06__CT3__is_scoped`: contribution `+0.004005`
- `lag_00__CT_kills_last_3s`: contribution `+0.003883`
- `lag_05__CT3__is_scoped`: contribution `+0.003746`
- `lag_00__kill_diff_last_3s`: contribution `+0.003448`

Top utility-only movements:
- `lag_00__T3__flash`: contribution `+0.003188`
- `lag_05__CT_B_site_active_infernos`: contribution `+0.003004`
- `lag_00__T3__utility_total`: contribution `+0.002318`
- `lag_05__T_B_site_active_infernos`: contribution `+0.002298`
- `lag_05__active_infernos_total`: contribution `+0.002261`

### tick `75172`, seconds `20.50`, LSTM delta `+0.1030`

Top all feature movements:
- `lag_00__T_place_SNIPERSNEST`: contribution `+0.014266`
- `lag_03__T_place_SNIPERSNEST`: contribution `+0.010799`
- `lag_00__CT_kills_last_3s`: contribution `+0.003883`
- `lag_13__T_place_CATWALK`: contribution `+0.003621`
- `lag_00__kill_diff_last_3s`: contribution `+0.003448`

Top utility-only movements:
- `lag_05__CT_B_site_active_infernos`: contribution `-0.003004`
- `lag_04__T1__flash_duration`: contribution `+0.002025`
- `lag_03__T5__flash_duration`: contribution `+0.001903`
- `lag_13__CT1__flash_duration`: contribution `+0.001817`
- `lag_05__CT_active_infernos`: contribution `-0.001364`

### tick `74756`, seconds `14.00`, LSTM delta `+0.0828`

Top all feature movements:
- `lag_06__T_he_last_5s`: contribution `+0.004870`
- `lag_13__CT_place_SHOP`: contribution `+0.004798`
- `lag_13__T_flashed_players`: contribution `+0.002958`
- `lag_00__T_shots_fired_sum`: contribution `+0.002553`
- `lag_14__CT_place_SHOP`: contribution `+0.002341`

Top utility-only movements:
- `lag_06__T_he_last_5s`: contribution `+0.004870`
- `lag_13__T1__flash_duration`: contribution `+0.002179`
- `lag_11__CT1__flash_duration`: contribution `+0.001919`
- `lag_03__T1__flash_duration`: contribution `+0.001874`
- `lag_13__T_flash_duration_sum`: contribution `+0.001370`

### tick `74692`, seconds `13.00`, LSTM delta `+0.0704`

Top all feature movements:
- `lag_14__T_he_last_5s`: contribution `+0.009616`
- `lag_04__T_he_last_5s`: contribution `+0.009449`
- `lag_00__CT_kills_last_3s`: contribution `+0.003883`
- `lag_00__kill_diff_last_3s`: contribution `+0.003448`
- `lag_11__CT_place_SHOP`: contribution `+0.002789`

Top utility-only movements:
- `lag_14__T_he_last_5s`: contribution `+0.009616`
- `lag_04__T_he_last_5s`: contribution `+0.009449`
- `lag_09__CT1__flash_duration`: contribution `+0.001564`
- `lag_07__T1__flash_duration`: contribution `+0.001426`

### tick `75684`, seconds `28.50`, LSTM delta `-0.0542`

Top all feature movements:
- `lag_11__T_place_LADDER`: contribution `-0.005464`
- `lag_05__CT3__is_scoped`: contribution `-0.003746`
- `lag_00__kill_diff_last_3s`: contribution `-0.003448`
- `lag_00__damage_diff_last_5s`: contribution `-0.002637`
- `lag_11__CT5__duck_amount`: contribution `-0.002482`

Top utility-only movements:
- No utility movement among the top local contributors.
