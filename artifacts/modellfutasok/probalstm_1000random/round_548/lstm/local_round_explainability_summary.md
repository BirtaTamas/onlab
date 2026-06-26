# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-faze-vs-aurora-bo3-OcxcOl9bFIHQQ2588nwUWG/faze-vs-aurora-m3-overpass.csv`
- round_num: `11`

## Largest probability jumps

- tick `81324`, seconds `97.50`, LSTM `0.8953`, delta `+0.1589`
- tick `76460`, seconds `21.50`, LSTM `0.8884`, delta `+0.1290`
- tick `75916`, seconds `13.00`, LSTM `0.6556`, delta `+0.0524`
- tick `75820`, seconds `11.50`, LSTM `0.6217`, delta `-0.0466`
- tick `76236`, seconds `18.00`, LSTM `0.7319`, delta `+0.0435`
- tick `81260`, seconds `96.50`, LSTM `0.7059`, delta `-0.0433`
- tick `80492`, seconds `84.50`, LSTM `0.7327`, delta `-0.0400`
- tick `80204`, seconds `80.00`, LSTM `0.8446`, delta `-0.0377`
- tick `80780`, seconds `89.00`, LSTM `0.7139`, delta `+0.0376`
- tick `80332`, seconds `82.00`, LSTM `0.7783`, delta `-0.0366`

## Top 15 local ridge features

- `lag_02__CT_place_BRIDGE`: coefficient `0.002250`, |coef| `0.002250`
- `lag_07__T2__flash_duration`: coefficient `-0.001408`, |coef| `0.001408`
- `lag_04__T_bomb_zone_count`: coefficient `-0.001213`, |coef| `0.001213`
- `lag_14__CT_place_WATER`: coefficient `0.001191`, |coef| `0.001191`
- `lag_01__CT2__is_scoped`: coefficient `0.001144`, |coef| `0.001144`
- `lag_06__CT_shots_fired_sum`: coefficient `-0.001058`, |coef| `0.001058`
- `lag_14__CT_place_WALKWAY`: coefficient `-0.001045`, |coef| `0.001045`
- `lag_06__CT1__shots_fired`: coefficient `-0.000984`, |coef| `0.000984`
- `lag_00__kill_diff_last_3s`: coefficient `0.000944`, |coef| `0.000944`
- `lag_00__CT_place_FOUNTAIN`: coefficient `-0.000897`, |coef| `0.000897`
- `lag_00__T2__is_walking`: coefficient `-0.000877`, |coef| `0.000877`
- `lag_14__CT1__duck_amount`: coefficient `0.000858`, |coef| `0.000858`
- `lag_07__CT_shots_fired_sum`: coefficient `0.000795`, |coef| `0.000795`
- `lag_04__T2__has_bomb`: coefficient `-0.000779`, |coef| `0.000779`
- `lag_03__T_B_site_active_infernos`: coefficient `-0.000770`, |coef| `0.000770`

## Top 10 utility ridge features

- `lag_07__T2__flash_duration`: coefficient `-0.001408` (lowers CT win probability)
- `lag_03__T_B_site_active_infernos`: coefficient `-0.000770` (lowers CT win probability)
- `lag_00__T_B_site_active_infernos`: coefficient `-0.000757` (lowers CT win probability)
- `lag_05__T_A_site_active_infernos`: coefficient `-0.000747` (lowers CT win probability)
- `lag_05__T_B_site_active_infernos`: coefficient `-0.000742` (lowers CT win probability)
- `lag_00__T_A_site_active_infernos`: coefficient `-0.000712` (lowers CT win probability)
- `lag_08__T2__flash_duration`: coefficient `-0.000696` (lowers CT win probability)
- `lag_14__T2__flash_duration`: coefficient `-0.000648` (lowers CT win probability)
- `lag_03__T_A_site_active_infernos`: coefficient `-0.000631` (lowers CT win probability)
- `lag_09__T2__flash_duration`: coefficient `-0.000615` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_02__CT_place_BRIDGE`: coefficient `0.002250` (raises CT win probability)
- `lag_04__T_bomb_zone_count`: coefficient `-0.001213` (lowers CT win probability)
- `lag_14__CT_place_WATER`: coefficient `0.001191` (raises CT win probability)
- `lag_01__CT2__is_scoped`: coefficient `0.001144` (raises CT win probability)
- `lag_06__CT_shots_fired_sum`: coefficient `-0.001058` (lowers CT win probability)
- `lag_14__CT_place_WALKWAY`: coefficient `-0.001045` (lowers CT win probability)
- `lag_06__CT1__shots_fired`: coefficient `-0.000984` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000944` (raises CT win probability)
- `lag_00__CT_place_FOUNTAIN`: coefficient `-0.000897` (lowers CT win probability)
- `lag_00__T2__is_walking`: coefficient `-0.000877` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `81324`, seconds `97.50`, LSTM delta `+0.1589`

Top all feature movements:
- `lag_02__CT_place_BRIDGE`: contribution `+0.025792`
- `lag_07__T2__flash_duration`: contribution `+0.010765`
- `lag_14__CT_place_WATER`: contribution `+0.007235`
- `lag_04__T_bomb_zone_count`: contribution `+0.007061`
- `lag_01__CT2__is_scoped`: contribution `+0.007002`

Top utility-only movements:
- `lag_07__T2__flash_duration`: contribution `+0.010765`

### tick `76460`, seconds `21.50`, LSTM delta `+0.1290`

Top all feature movements:
- `lag_00__CT_place_FOUNTAIN`: contribution `+0.009433`
- `lag_14__T_shots_fired_sum`: contribution `+0.005520`
- `lag_06__CT_shots_fired_sum`: contribution `+0.003674`
- `lag_11__CT_place_WATER`: contribution `+0.003608`
- `lag_00__CT1__flash_duration`: contribution `+0.003375`

Top utility-only movements:
- `lag_00__CT1__flash_duration`: contribution `+0.003375`
- `lag_07__CT_utility_damage_last_5s`: contribution `+0.003065`
- `lag_00__T2__flash_duration`: contribution `+0.002925`
- `lag_00__T4__flash_duration`: contribution `+0.002520`
- `lag_00__CT_flash_duration_sum`: contribution `+0.002226`

### tick `75916`, seconds `13.00`, LSTM delta `+0.0524`

Top all feature movements:
- `lag_12__T_place_TSTAIRS`: contribution `+0.005325`
- `lag_00__CT_shots_fired_sum`: contribution `-0.004088`
- `lag_13__T_place_TSTAIRS`: contribution `+0.003297`
- `lag_09__CT_place_WATER`: contribution `+0.003136`
- `lag_04__CT_place_UPPERPARK`: contribution `+0.003075`

Top utility-only movements:
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.001807`
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.001508`

### tick `75820`, seconds `11.50`, LSTM delta `-0.0466`

Top all feature movements:
- `lag_09__T_place_TSTAIRS`: contribution `-0.006979`
- `lag_13__T_place_TSTAIRS`: contribution `-0.006594`
- `lag_15__CT_place_STAIRS`: contribution `+0.003634`
- `lag_14__CT_place_STAIRS`: contribution `-0.002748`
- `lag_00__T_place_ALLEY`: contribution `-0.002696`

Top utility-only movements:
- `lag_03__T_B_site_active_infernos`: contribution `-0.002178`
- `lag_02__T5__flash_duration`: contribution `-0.001481`
- `lag_03__T_active_infernos`: contribution `-0.001197`

### tick `76236`, seconds `18.00`, LSTM delta `+0.0435`

Top all feature movements:
- `lag_06__CT2__is_scoped`: contribution `+0.004492`
- `lag_15__CT_place_WATER`: contribution `-0.002959`
- `lag_07__CT_shots_fired_sum`: contribution `-0.002763`
- `lag_04__CT_place_WALKWAY`: contribution `+0.002277`
- `lag_00__kill_diff_last_3s`: contribution `+0.002273`

Top utility-only movements:
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.001807`
- `lag_00__CT_utility_damage_last_5s`: contribution `-0.001508`
