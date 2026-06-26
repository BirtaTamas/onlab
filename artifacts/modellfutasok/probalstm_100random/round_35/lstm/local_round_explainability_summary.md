# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-spirit-vs-heroic-bo3-NSK1XoxyhXK-sIe0204dzp/spirit-vs-heroic-m2-nuke.csv`
- round_num: `5`

## Largest probability jumps

- tick `34077`, seconds `38.50`, LSTM `0.7503`, delta `+0.1935`
- tick `34237`, seconds `41.00`, LSTM `0.6080`, delta `-0.1711`
- tick `35933`, seconds `67.50`, LSTM `0.9100`, delta `+0.1483`
- tick `33117`, seconds `23.50`, LSTM `0.5667`, delta `+0.0587`
- tick `36285`, seconds `73.00`, LSTM `0.8870`, delta `-0.0574`
- tick `35645`, seconds `63.00`, LSTM `0.6834`, delta `+0.0513`
- tick `35293`, seconds `57.50`, LSTM `0.6428`, delta `-0.0470`
- tick `34269`, seconds `41.50`, LSTM `0.5649`, delta `-0.0431`
- tick `32509`, seconds `14.00`, LSTM `0.4278`, delta `-0.0372`
- tick `35741`, seconds `64.50`, LSTM `0.7400`, delta `+0.0363`

## Top 15 local ridge features

- `lag_02__CT_place_LOCKERROOM`: coefficient `0.003344`, |coef| `0.003344`
- `lag_09__CT_place_CONTROL`: coefficient `0.002856`, |coef| `0.002856`
- `lag_05__CT_place_VENTS`: coefficient `0.001799`, |coef| `0.001799`
- `lag_13__CT_place_RAFTERS`: coefficient `-0.001660`, |coef| `0.001660`
- `lag_00__damage_diff_last_5s`: coefficient `0.001653`, |coef| `0.001653`
- `lag_00__CT_kills_last_3s`: coefficient `0.001619`, |coef| `0.001619`
- `lag_00__kill_diff_last_3s`: coefficient `0.001527`, |coef| `0.001527`
- `lag_00__CT_damage_last_5s`: coefficient `0.001465`, |coef| `0.001465`
- `lag_10__T_place_TROPHY`: coefficient `0.001210`, |coef| `0.001210`
- `lag_00__T4__is_walking`: coefficient `-0.001174`, |coef| `0.001174`
- `lag_10__T_place_SQUEAKY`: coefficient `-0.001166`, |coef| `0.001166`
- `lag_02__CT_place_CONTROL`: coefficient `0.001155`, |coef| `0.001155`
- `lag_15__T_place_SQUEAKY`: coefficient `0.001102`, |coef| `0.001102`
- `lag_06__T_A_site_active_infernos`: coefficient `-0.001021`, |coef| `0.001021`
- `lag_06__T_place_MINI`: coefficient `0.001015`, |coef| `0.001015`

## Top 10 utility ridge features

- `lag_06__T_A_site_active_infernos`: coefficient `-0.001021` (lowers CT win probability)
- `lag_06__T_B_site_active_infernos`: coefficient `-0.000971` (lowers CT win probability)
- `lag_06__T_active_infernos`: coefficient `-0.000719` (lowers CT win probability)
- `lag_00__T_A_site_active_infernos`: coefficient `-0.000694` (lowers CT win probability)
- `lag_00__T_B_site_active_infernos`: coefficient `-0.000659` (lowers CT win probability)
- `lag_14__T3__flash_duration`: coefficient `-0.000538` (lowers CT win probability)
- `lag_02__T5__flash_duration`: coefficient `0.000510` (raises CT win probability)
- `lag_11__T3__flash_duration`: coefficient `-0.000504` (lowers CT win probability)
- `lag_06__active_infernos_total`: coefficient `-0.000501` (lowers CT win probability)
- `lag_00__T_active_infernos`: coefficient `-0.000486` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_02__CT_place_LOCKERROOM`: coefficient `0.003344` (raises CT win probability)
- `lag_09__CT_place_CONTROL`: coefficient `0.002856` (raises CT win probability)
- `lag_05__CT_place_VENTS`: coefficient `0.001799` (raises CT win probability)
- `lag_13__CT_place_RAFTERS`: coefficient `-0.001660` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001653` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001619` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001527` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001465` (raises CT win probability)
- `lag_10__T_place_TROPHY`: coefficient `0.001210` (raises CT win probability)
- `lag_00__T4__is_walking`: coefficient `-0.001174` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `34077`, seconds `38.50`, LSTM delta `+0.1935`

Top all feature movements:
- `lag_02__CT_place_LOCKERROOM`: contribution `+0.041623`
- `lag_05__CT_place_VENTS`: contribution `+0.015092`
- `lag_10__T_place_TROPHY`: contribution `+0.007670`
- `lag_07__T_place_CONTROL`: contribution `+0.006745`
- `lag_03__T_place_CONTROL`: contribution `+0.006337`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `34237`, seconds `41.00`, LSTM delta `-0.1711`

Top all feature movements:
- `lag_02__CT_place_LOCKERROOM`: contribution `-0.041623`
- `lag_07__CT_place_LOCKERROOM`: contribution `-0.012587`
- `lag_00__CT_place_SQUEAKY`: contribution `-0.012043`
- `lag_01__CT_place_SQUEAKY`: contribution `-0.012021`
- `lag_05__T_place_CONTROL`: contribution `-0.007213`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `35933`, seconds `67.50`, LSTM delta `+0.1483`

Top all feature movements:
- `lag_09__CT_place_CONTROL`: contribution `+0.029646`
- `lag_13__CT_place_RAFTERS`: contribution `+0.008867`
- `lag_10__T_place_SQUEAKY`: contribution `+0.007259`
- `lag_00__CT_kills_last_3s`: contribution `+0.004673`
- `lag_00__kill_diff_last_3s`: contribution `+0.003675`

Top utility-only movements:
- `lag_06__T_A_site_active_infernos`: contribution `+0.003040`
- `lag_06__T_B_site_active_infernos`: contribution `+0.002745`

### tick `33117`, seconds `23.50`, LSTM delta `+0.0587`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.004673`
- `lag_14__T_place_TROPHY`: contribution `+0.004134`
- `lag_00__CT_place_HEAVEN`: contribution `+0.004001`
- `lag_00__kill_diff_last_3s`: contribution `+0.003675`
- `lag_14__T3__flash_duration`: contribution `+0.003660`

Top utility-only movements:
- `lag_14__T3__flash_duration`: contribution `+0.003660`

### tick `36285`, seconds `73.00`, LSTM delta `-0.0574`

Top all feature movements:
- `lag_06__T_place_MINI`: contribution `-0.014127`
- `lag_00__CT_shots_fired_sum`: contribution `-0.004702`
- `lag_00__kill_diff_last_3s`: contribution `-0.003675`
- `lag_04__CT_place_GARAGE`: contribution `-0.002797`
- `lag_07__CT4__is_walking`: contribution `-0.001935`

Top utility-only movements:
- No utility movement among the top local contributors.
