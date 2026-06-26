# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-virtuspro-vs-vitality-bo3-8Ft8K1evi_LZ8kW_kkrYdB/virtus-pro-vs-vitality-m1-train.csv`
- round_num: `9`

## Largest probability jumps

- tick `70207`, seconds `78.50`, LSTM `0.9470`, delta `+0.2148`
- tick `69599`, seconds `69.00`, LSTM `0.6790`, delta `+0.1788`
- tick `69727`, seconds `71.00`, LSTM `0.8059`, delta `+0.1480`
- tick `71391`, seconds `97.00`, LSTM `0.9712`, delta `+0.1342`
- tick `69183`, seconds `62.50`, LSTM `0.4217`, delta `+0.1275`
- tick `71359`, seconds `96.50`, LSTM `0.8370`, delta `-0.1030`
- tick `69119`, seconds `61.50`, LSTM `0.3103`, delta `-0.0735`
- tick `69087`, seconds `61.00`, LSTM `0.3838`, delta `-0.0529`
- tick `70079`, seconds `76.50`, LSTM `0.7823`, delta `+0.0368`
- tick `69215`, seconds `63.00`, LSTM `0.4565`, delta `+0.0348`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001516`, |coef| `0.001516`
- `lag_00__kill_diff_last_3s`: coefficient `0.001497`, |coef| `0.001497`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001399`, |coef| `0.001399`
- `lag_08__CT_place_ENTRANCE`: coefficient `0.001391`, |coef| `0.001391`
- `lag_03__T_flashes_last_5s`: coefficient `-0.001336`, |coef| `0.001336`
- `lag_06__T5__flash_duration`: coefficient `-0.001331`, |coef| `0.001331`
- `lag_00__T3__flash_duration`: coefficient `-0.001294`, |coef| `0.001294`
- `lag_00__CT_damage_last_5s`: coefficient `0.001284`, |coef| `0.001284`
- `lag_15__T_bomb_zone_count`: coefficient `0.001267`, |coef| `0.001267`
- `lag_03__T_bomb_zone_count`: coefficient `0.001248`, |coef| `0.001248`
- `lag_07__CT_flash_duration_sum`: coefficient `-0.001248`, |coef| `0.001248`
- `lag_02__T_flashes_last_5s`: coefficient `-0.001243`, |coef| `0.001243`
- `lag_05__T_flashes_last_5s`: coefficient `0.001232`, |coef| `0.001232`
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.001227`, |coef| `0.001227`
- `lag_07__CT_flashed_players`: coefficient `-0.001208`, |coef| `0.001208`

## Top 10 utility ridge features

- `lag_03__T_flashes_last_5s`: coefficient `-0.001336` (lowers CT win probability)
- `lag_06__T5__flash_duration`: coefficient `-0.001331` (lowers CT win probability)
- `lag_00__T3__flash_duration`: coefficient `-0.001294` (lowers CT win probability)
- `lag_07__CT_flash_duration_sum`: coefficient `-0.001248` (lowers CT win probability)
- `lag_02__T_flashes_last_5s`: coefficient `-0.001243` (lowers CT win probability)
- `lag_05__T_flashes_last_5s`: coefficient `0.001232` (raises CT win probability)
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.001227` (raises CT win probability)
- `lag_13__CT_utility_damage_last_5s`: coefficient `0.001065` (raises CT win probability)
- `lag_07__CT_utility_damage_last_5s`: coefficient `-0.001064` (lowers CT win probability)
- `lag_07__CT1__flash_duration`: coefficient `-0.001046` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001516` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001497` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001399` (raises CT win probability)
- `lag_08__CT_place_ENTRANCE`: coefficient `0.001391` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001284` (raises CT win probability)
- `lag_15__T_bomb_zone_count`: coefficient `0.001267` (raises CT win probability)
- `lag_03__T_bomb_zone_count`: coefficient `0.001248` (raises CT win probability)
- `lag_07__CT_flashed_players`: coefficient `-0.001208` (lowers CT win probability)
- `lag_07__CT_place_ENTRANCE`: coefficient `-0.001204` (lowers CT win probability)
- `lag_05__CT_flashed_players`: coefficient `0.001117` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `70207`, seconds `78.50`, LSTM delta `+0.2148`

Top all feature movements:
- `lag_07__CT_flash_duration_sum`: contribution `+0.009389`
- `lag_06__T5__flash_duration`: contribution `+0.007959`
- `lag_07__CT_flashed_players`: contribution `+0.007935`
- `lag_15__T_bomb_zone_count`: contribution `+0.007373`
- `lag_03__T_bomb_zone_count`: contribution `+0.007267`

Top utility-only movements:
- `lag_07__CT_flash_duration_sum`: contribution `+0.009389`
- `lag_06__T5__flash_duration`: contribution `+0.007959`
- `lag_07__CT1__flash_duration`: contribution `+0.005620`
- `lag_07__CT3__flash_duration`: contribution `+0.005106`
- `lag_07__CT5__flash_duration`: contribution `+0.004695`

### tick `69599`, seconds `69.00`, LSTM delta `+0.1788`

Top all feature movements:
- `lag_13__CT_utility_damage_last_5s`: contribution `+0.011132`
- `lag_00__T3__flash_duration`: contribution `+0.009975`
- `lag_00__CT_shots_fired_sum`: contribution `+0.008746`
- `lag_03__CT_utility_damage_last_5s`: contribution `+0.007523`
- `lag_13__utility_damage_diff_last_5s`: contribution `+0.007449`

Top utility-only movements:
- `lag_13__CT_utility_damage_last_5s`: contribution `+0.011132`
- `lag_00__T3__flash_duration`: contribution `+0.009975`
- `lag_03__CT_utility_damage_last_5s`: contribution `+0.007523`
- `lag_13__utility_damage_diff_last_5s`: contribution `+0.007449`
- `lag_12__T2__flash_duration`: contribution `+0.006527`

### tick `69727`, seconds `71.00`, LSTM delta `+0.1480`

Top all feature movements:
- `lag_07__CT_utility_damage_last_5s`: contribution `+0.011121`
- `lag_12__T_flashes_last_5s`: contribution `+0.007789`
- `lag_07__utility_damage_diff_last_5s`: contribution `+0.007424`
- `lag_05__CT_flashed_players`: contribution `+0.004894`
- `lag_15__T3__flash_duration`: contribution `+0.004582`

Top utility-only movements:
- `lag_07__CT_utility_damage_last_5s`: contribution `+0.011121`
- `lag_12__T_flashes_last_5s`: contribution `+0.007789`
- `lag_07__utility_damage_diff_last_5s`: contribution `+0.007424`
- `lag_15__T3__flash_duration`: contribution `+0.004582`
- `lag_05__CT1__flash_duration`: contribution `+0.004545`

### tick `71391`, seconds `97.00`, LSTM delta `+0.1342`

Top all feature movements:
- `lag_08__CT_place_ENTRANCE`: contribution `+0.024682`
- `lag_00__CT_shots_fired_sum`: contribution `+0.006802`
- `lag_00__T1__is_scoped`: contribution `+0.005594`
- `lag_06__CT_place_ENTRANCE`: contribution `+0.005095`
- `lag_00__CT_kills_last_3s`: contribution `+0.004376`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `69183`, seconds `62.50`, LSTM delta `+0.1275`

Top all feature movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.012835`
- `lag_05__T_flashes_last_5s`: contribution `+0.011167`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.007999`
- `lag_05__CT_flashed_players`: contribution `+0.007341`
- `lag_05__CT5__flash_duration`: contribution `+0.004626`

Top utility-only movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.012835`
- `lag_05__T_flashes_last_5s`: contribution `+0.011167`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.007999`
- `lag_05__CT5__flash_duration`: contribution `+0.004626`
- `lag_03__T3__flash_duration`: contribution `+0.003214`
