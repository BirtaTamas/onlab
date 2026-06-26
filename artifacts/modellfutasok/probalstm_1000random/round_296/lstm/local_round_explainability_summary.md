# Local Round Explainability

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-spirit-vs-flyquest-bo3-ElcEZT56lTCLJYDcWlMY2d/spirit-vs-flyquest-m1-mirage.csv`
- round_num: `7`

## Largest probability jumps

- tick `62200`, seconds `63.50`, LSTM `0.2307`, delta `-0.3388`
- tick `62104`, seconds `62.00`, LSTM `0.5492`, delta `+0.2937`
- tick `62296`, seconds `65.00`, LSTM `0.0485`, delta `-0.1596`
- tick `62040`, seconds `61.00`, LSTM `0.1910`, delta `-0.1221`
- tick `61976`, seconds `60.00`, LSTM `0.3066`, delta `-0.0726`
- tick `61912`, seconds `59.00`, LSTM `0.3753`, delta `-0.0717`
- tick `62072`, seconds `61.50`, LSTM `0.2555`, delta `+0.0645`
- tick `61848`, seconds `58.00`, LSTM `0.4963`, delta `-0.0569`
- tick `58904`, seconds `12.00`, LSTM `0.3606`, delta `-0.0566`
- tick `61880`, seconds `58.50`, LSTM `0.4470`, delta `-0.0493`

## Top 15 local ridge features

- `lag_05__T_shots_fired_sum`: coefficient `0.001981`, |coef| `0.001981`
- `lag_02__T_shots_fired_sum`: coefficient `-0.001823`, |coef| `0.001823`
- `lag_05__CT_place_JUNGLE`: coefficient `-0.001794`, |coef| `0.001794`
- `lag_02__T5__shots_fired`: coefficient `-0.001771`, |coef| `0.001771`
- `lag_00__CT_place_TRUCK`: coefficient `0.001706`, |coef| `0.001706`
- `lag_07__T_place_PALACEINTERIOR`: coefficient `-0.001668`, |coef| `0.001668`
- `lag_02__CT_shots_fired_sum`: coefficient `0.001601`, |coef| `0.001601`
- `lag_05__T5__shots_fired`: coefficient `0.001563`, |coef| `0.001563`
- `lag_01__T3__is_scoped`: coefficient `-0.001447`, |coef| `0.001447`
- `lag_00__T_place_BOMBSITEA`: coefficient `-0.001401`, |coef| `0.001401`
- `lag_00__T_macro_A`: coefficient `-0.001401`, |coef| `0.001401`
- `lag_02__T_place_SCAFFOLDING`: coefficient `-0.001398`, |coef| `0.001398`
- `lag_02__CT_place_PALACEINTERIOR`: coefficient `-0.001327`, |coef| `0.001327`
- `lag_08__CT_place_PALACEINTERIOR`: coefficient `0.001296`, |coef| `0.001296`
- `lag_10__T_place_PALACEINTERIOR`: coefficient `0.001276`, |coef| `0.001276`

## Top 10 utility ridge features

- `lag_02__CT4__flash_duration`: coefficient `-0.001235` (lowers CT win probability)
- `lag_08__CT4__flash_duration`: coefficient `0.001207` (raises CT win probability)
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.000935` (lowers CT win probability)
- `lag_11__CT_utility_damage_last_5s`: coefficient `-0.000865` (lowers CT win probability)
- `lag_00__CT4__flash`: coefficient `0.000840` (raises CT win probability)
- `lag_11__CT4__flash_duration`: coefficient `-0.000765` (lowers CT win probability)
- `lag_05__CT4__flash_duration`: coefficient `0.000751` (raises CT win probability)
- `lag_11__utility_damage_diff_last_5s`: coefficient `-0.000707` (lowers CT win probability)
- `lag_07__CT_B_site_active_infernos`: coefficient `-0.000699` (lowers CT win probability)
- `lag_02__CT_flash_duration_sum`: coefficient `-0.000698` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_05__T_shots_fired_sum`: coefficient `0.001981` (raises CT win probability)
- `lag_02__T_shots_fired_sum`: coefficient `-0.001823` (lowers CT win probability)
- `lag_05__CT_place_JUNGLE`: coefficient `-0.001794` (lowers CT win probability)
- `lag_02__T5__shots_fired`: coefficient `-0.001771` (lowers CT win probability)
- `lag_00__CT_place_TRUCK`: coefficient `0.001706` (raises CT win probability)
- `lag_07__T_place_PALACEINTERIOR`: coefficient `-0.001668` (lowers CT win probability)
- `lag_02__CT_shots_fired_sum`: coefficient `0.001601` (raises CT win probability)
- `lag_05__T5__shots_fired`: coefficient `0.001563` (raises CT win probability)
- `lag_01__T3__is_scoped`: coefficient `-0.001447` (lowers CT win probability)
- `lag_00__T_place_BOMBSITEA`: coefficient `-0.001401` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `62200`, seconds `63.50`, LSTM delta `-0.3388`

Top all feature movements:
- `lag_05__T_shots_fired_sum`: contribution `-0.016341`
- `lag_02__CT_shots_fired_sum`: contribution `-0.013347`
- `lag_05__CT_place_JUNGLE`: contribution `-0.011507`
- `lag_05__T5__shots_fired`: contribution `-0.010572`
- `lag_09__CT_place_TRUCK`: contribution `-0.008070`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `62104`, seconds `62.00`, LSTM delta `+0.2937`

Top all feature movements:
- `lag_02__T_shots_fired_sum`: contribution `+0.015031`
- `lag_02__T5__shots_fired`: contribution `+0.011980`
- `lag_05__CT_place_JUNGLE`: contribution `+0.011507`
- `lag_02__CT_place_JUNGLE`: contribution `+0.007829`
- `lag_04__T3__is_scoped`: contribution `+0.007432`

Top utility-only movements:
- `lag_02__CT4__flash_duration`: contribution `+0.004686`
- `lag_08__CT4__flash_duration`: contribution `+0.004583`

### tick `62296`, seconds `65.00`, LSTM delta `-0.1596`

Top all feature movements:
- `lag_02__T_place_SCAFFOLDING`: contribution `-0.047597`
- `lag_05__CT_shots_fired_sum`: contribution `-0.006916`
- `lag_08__CT_place_JUNGLE`: contribution `+0.006388`
- `lag_08__CT_place_PALACEINTERIOR`: contribution `-0.005281`
- `lag_08__CT4__flash_duration`: contribution `-0.004583`

Top utility-only movements:
- `lag_08__CT4__flash_duration`: contribution `-0.004583`

### tick `62040`, seconds `61.00`, LSTM delta `-0.1221`

Top all feature movements:
- `lag_01__T3__is_scoped`: contribution `-0.009283`
- `lag_02__T_shots_fired_sum`: contribution `-0.006832`
- `lag_07__T_place_PALACEINTERIOR`: contribution `-0.005594`
- `lag_02__T5__shots_fired`: contribution `-0.005445`
- `lag_06__CT_place_TRUCK`: contribution `+0.004388`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `61976`, seconds `60.00`, LSTM delta `-0.0726`

Top all feature movements:
- `lag_01__T3__is_scoped`: contribution `-0.009283`
- `lag_00__T3__is_scoped`: contribution `+0.004601`
- `lag_13__CT_place_CATWALK`: contribution `+0.003465`
- `lag_03__CT_place_TRUCK`: contribution `-0.003422`
- `lag_04__CT_place_TRUCK`: contribution `-0.003416`

Top utility-only movements:
- `lag_00__T_utility_damage_last_5s`: contribution `-0.001603`
- `lag_14__CT_B_site_active_infernos`: contribution `-0.001491`
- `lag_07__utility_damage_diff_last_5s`: contribution `-0.001459`
- `lag_03__CT_B_site_active_infernos`: contribution `-0.001402`
