# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-tyloo-vs-vitality-bo3-aF98ikh3PjdqKlkdIJn9tC/tyloo-vs-vitality-m1-inferno.csv`
- round_num: `10`

## Largest probability jumps

- tick `74764`, seconds `57.00`, LSTM `0.8021`, delta `+0.1153`
- tick `75340`, seconds `66.00`, LSTM `0.9391`, delta `+0.1021`
- tick `73196`, seconds `32.50`, LSTM `0.6745`, delta `+0.0737`
- tick `75116`, seconds `62.50`, LSTM `0.8210`, delta `+0.0394`
- tick `74956`, seconds `60.00`, LSTM `0.7795`, delta `-0.0349`
- tick `74860`, seconds `58.50`, LSTM `0.7911`, delta `-0.0336`
- tick `73388`, seconds `35.50`, LSTM `0.6587`, delta `-0.0316`
- tick `78444`, seconds `114.50`, LSTM `0.9389`, delta `-0.0308`
- tick `75180`, seconds `63.50`, LSTM `0.8225`, delta `+0.0303`
- tick `75148`, seconds `63.00`, LSTM `0.7922`, delta `-0.0287`

## Top 15 local ridge features

- `lag_01__CT1__flash_duration`: coefficient `0.001629`, |coef| `0.001629`
- `lag_00__CT_kills_last_3s`: coefficient `0.001469`, |coef| `0.001469`
- `lag_00__kill_diff_last_3s`: coefficient `0.001374`, |coef| `0.001374`
- `lag_15__CT_place_LIBRARY`: coefficient `0.001128`, |coef| `0.001128`
- `lag_08__T_place_UNDERPASS`: coefficient `-0.000925`, |coef| `0.000925`
- `lag_00__CT_damage_last_5s`: coefficient `0.000920`, |coef| `0.000920`
- `lag_00__damage_diff_last_5s`: coefficient `0.000858`, |coef| `0.000858`
- `lag_07__CT1__flash_duration`: coefficient `-0.000846`, |coef| `0.000846`
- `lag_07__T3__duck_amount`: coefficient `-0.000840`, |coef| `0.000840`
- `lag_07__CT_A_site_active_infernos`: coefficient `-0.000836`, |coef| `0.000836`
- `lag_04__T_place_SECONDMID`: coefficient `-0.000829`, |coef| `0.000829`
- `lag_00__T5__alive`: coefficient `-0.000798`, |coef| `0.000798`
- `lag_00__T5__hp`: coefficient `-0.000784`, |coef| `0.000784`
- `lag_14__CT_place_RUINS`: coefficient `-0.000782`, |coef| `0.000782`
- `lag_11__T3__duck_amount`: coefficient `-0.000779`, |coef| `0.000779`

## Top 10 utility ridge features

- `lag_01__CT1__flash_duration`: coefficient `0.001629` (raises CT win probability)
- `lag_07__CT1__flash_duration`: coefficient `-0.000846` (lowers CT win probability)
- `lag_07__CT_A_site_active_infernos`: coefficient `-0.000836` (lowers CT win probability)
- `lag_15__CT4__flash_duration`: coefficient `0.000772` (raises CT win probability)
- `lag_05__CT4__flash_duration`: coefficient `-0.000687` (lowers CT win probability)
- `lag_07__CT_active_infernos`: coefficient `-0.000665` (lowers CT win probability)
- `lag_00__T_A_site_active_infernos`: coefficient `0.000664` (raises CT win probability)
- `lag_01__CT_flash_duration_sum`: coefficient `0.000646` (raises CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `-0.000633` (lowers CT win probability)
- `lag_02__T5__molly`: coefficient `-0.000602` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001469` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001374` (raises CT win probability)
- `lag_15__CT_place_LIBRARY`: coefficient `0.001128` (raises CT win probability)
- `lag_08__T_place_UNDERPASS`: coefficient `-0.000925` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.000920` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000858` (raises CT win probability)
- `lag_07__T3__duck_amount`: coefficient `-0.000840` (lowers CT win probability)
- `lag_04__T_place_SECONDMID`: coefficient `-0.000829` (lowers CT win probability)
- `lag_00__T5__alive`: coefficient `-0.000798` (lowers CT win probability)
- `lag_00__T5__hp`: coefficient `-0.000784` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `74764`, seconds `57.00`, LSTM delta `+0.1153`

Top all feature movements:
- `lag_01__CT1__flash_duration`: contribution `+0.010647`
- `lag_00__CT_kills_last_3s`: contribution `+0.004242`
- `lag_08__T_place_UNDERPASS`: contribution `+0.003622`
- `lag_00__kill_diff_last_3s`: contribution `+0.003306`
- `lag_07__T3__duck_amount`: contribution `+0.003167`

Top utility-only movements:
- `lag_01__CT1__flash_duration`: contribution `+0.010647`
- `lag_07__CT_A_site_active_infernos`: contribution `+0.002949`
- `lag_10__CT_A_site_active_infernos`: contribution `+0.002043`
- `lag_00__T_A_site_active_infernos`: contribution `+0.001977`
- `lag_01__CT_flash_duration_sum`: contribution `+0.001898`

### tick `75340`, seconds `66.00`, LSTM delta `+0.1021`

Top all feature movements:
- `lag_15__CT_place_LIBRARY`: contribution `+0.007230`
- `lag_07__CT1__flash_duration`: contribution `+0.005533`
- `lag_15__CT4__flash_duration`: contribution `+0.004775`
- `lag_05__CT4__flash_duration`: contribution `+0.004252`
- `lag_00__CT_kills_last_3s`: contribution `+0.004242`

Top utility-only movements:
- `lag_07__CT1__flash_duration`: contribution `+0.005533`
- `lag_15__CT4__flash_duration`: contribution `+0.004775`
- `lag_05__CT4__flash_duration`: contribution `+0.004252`
- `lag_07__CT_flash_duration_sum`: contribution `+0.001457`

### tick `73196`, seconds `32.50`, LSTM delta `+0.0737`

Top all feature movements:
- `lag_11__T_place_BALCONY`: contribution `+0.010607`
- `lag_07__T_place_BALCONY`: contribution `+0.007804`
- `lag_00__CT_kills_last_3s`: contribution `+0.004242`
- `lag_00__kill_diff_last_3s`: contribution `+0.003306`
- `lag_01__CT2__shots_fired`: contribution `+0.002286`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `75116`, seconds `62.50`, LSTM delta `+0.0394`

Top all feature movements:
- `lag_08__CT_place_LIBRARY`: contribution `+0.003385`
- `lag_00__CT_shots_fired_sum`: contribution `+0.002041`
- `lag_05__CT_shots_fired_sum`: contribution `+0.001829`
- `lag_07__CT3__is_walking`: contribution `+0.001623`
- `lag_07__CT3__duck_amount`: contribution `+0.001576`

Top utility-only movements:
- `lag_08__CT4__flash_duration`: contribution `+0.001061`
- `lag_12__CT1__flash_duration`: contribution `+0.000851`

### tick `74956`, seconds `60.00`, LSTM delta `-0.0349`

Top all feature movements:
- `lag_07__CT1__flash_duration`: contribution `-0.005533`
- `lag_00__CT_kills_last_3s`: contribution `-0.004242`
- `lag_00__kill_diff_last_3s`: contribution `-0.003306`
- `lag_00__CT_shots_fired_sum`: contribution `-0.002449`
- `lag_13__CT_A_site_active_infernos`: contribution `-0.001614`

Top utility-only movements:
- `lag_07__CT1__flash_duration`: contribution `-0.005533`
- `lag_13__CT_A_site_active_infernos`: contribution `-0.001614`
- `lag_07__CT_flash_duration_sum`: contribution `-0.001457`
