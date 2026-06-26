# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-furia-vs-aurora-bo3-0icw3xvkvOZhHsCT2PEavZ/furia-vs-aurora-m1-inferno.csv`
- round_num: `11`

## Largest probability jumps

- tick `91343`, seconds `86.50`, LSTM `0.8597`, delta `+0.2175`
- tick `90479`, seconds `73.00`, LSTM `0.7304`, delta `+0.1703`
- tick `87887`, seconds `32.50`, LSTM `0.6557`, delta `+0.1110`
- tick `90575`, seconds `74.50`, LSTM `0.8426`, delta `+0.0806`
- tick `90767`, seconds `77.50`, LSTM `0.7939`, delta `-0.0702`
- tick `90543`, seconds `74.00`, LSTM `0.7620`, delta `+0.0518`
- tick `86607`, seconds `12.50`, LSTM `0.4912`, delta `-0.0469`
- tick `91439`, seconds `88.00`, LSTM `0.9548`, delta `+0.0453`
- tick `89327`, seconds `55.00`, LSTM `0.6895`, delta `-0.0418`
- tick `88143`, seconds `36.50`, LSTM `0.6525`, delta `-0.0403`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.003254`, |coef| `0.003254`
- `lag_00__kill_diff_last_3s`: coefficient `0.002504`, |coef| `0.002504`
- `lag_10__CT_place_LIBRARY`: coefficient `-0.002395`, |coef| `0.002395`
- `lag_13__T_bomb_zone_count`: coefficient `0.002268`, |coef| `0.002268`
- `lag_05__T_bomb_zone_count`: coefficient `-0.002237`, |coef| `0.002237`
- `lag_03__CT1__flash_duration`: coefficient `0.002135`, |coef| `0.002135`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002126`, |coef| `0.002126`
- `lag_00__CT_damage_last_5s`: coefficient `0.002048`, |coef| `0.002048`
- `lag_01__T5__is_scoped`: coefficient `0.001874`, |coef| `0.001874`
- `lag_14__CT_place_LIBRARY`: coefficient `0.001866`, |coef| `0.001866`
- `lag_00__CT4__flash_duration`: coefficient `0.001819`, |coef| `0.001819`
- `lag_00__damage_diff_last_5s`: coefficient `0.001758`, |coef| `0.001758`
- `lag_01__CT4__is_scoped`: coefficient `0.001603`, |coef| `0.001603`
- `lag_00__CT1__is_walking`: coefficient `-0.001541`, |coef| `0.001541`
- `lag_00__T_flashed_players`: coefficient `0.001520`, |coef| `0.001520`

## Top 10 utility ridge features

- `lag_03__CT1__flash_duration`: coefficient `0.002135` (raises CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `0.001819` (raises CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `0.001331` (raises CT win probability)
- `lag_03__CT_flash_duration_sum`: coefficient `0.001257` (raises CT win probability)
- `lag_15__T3__molly`: coefficient `-0.001161` (lowers CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `0.001111` (raises CT win probability)
- `lag_11__T_B_site_active_infernos`: coefficient `0.000989` (raises CT win probability)
- `lag_02__CT3__flash`: coefficient `-0.000929` (lowers CT win probability)
- `lag_03__T_utility_damage_last_5s`: coefficient `0.000898` (raises CT win probability)
- `lag_11__T_active_infernos`: coefficient `0.000828` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.003254` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002504` (raises CT win probability)
- `lag_10__CT_place_LIBRARY`: coefficient `-0.002395` (lowers CT win probability)
- `lag_13__T_bomb_zone_count`: coefficient `0.002268` (raises CT win probability)
- `lag_05__T_bomb_zone_count`: coefficient `-0.002237` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002126` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002048` (raises CT win probability)
- `lag_01__T5__is_scoped`: coefficient `0.001874` (raises CT win probability)
- `lag_14__CT_place_LIBRARY`: coefficient `0.001866` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001758` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `91343`, seconds `86.50`, LSTM delta `+0.2175`

Top all feature movements:
- `lag_10__CT_place_LIBRARY`: contribution `+0.015354`
- `lag_13__T_bomb_zone_count`: contribution `+0.013200`
- `lag_05__T_bomb_zone_count`: contribution `+0.013021`
- `lag_14__CT_place_LIBRARY`: contribution `+0.011966`
- `lag_00__CT_kills_last_3s`: contribution `+0.009396`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `90479`, seconds `73.00`, LSTM delta `+0.1703`

Top all feature movements:
- `lag_03__CT1__flash_duration`: contribution `+0.011088`
- `lag_00__CT_kills_last_3s`: contribution `+0.009396`
- `lag_00__CT4__flash_duration`: contribution `+0.008616`
- `lag_00__CT_shots_fired_sum`: contribution `+0.007386`
- `lag_00__kill_diff_last_3s`: contribution `+0.006027`

Top utility-only movements:
- `lag_03__CT1__flash_duration`: contribution `+0.011088`
- `lag_00__CT4__flash_duration`: contribution `+0.008616`
- `lag_00__CT_flash_duration_sum`: contribution `+0.005170`
- `lag_03__CT_flash_duration_sum`: contribution `+0.003607`
- `lag_03__T_utility_damage_last_5s`: contribution `+0.003205`

### tick `87887`, seconds `32.50`, LSTM delta `+0.1110`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.009396`
- `lag_00__CT_shots_fired_sum`: contribution `+0.007386`
- `lag_00__kill_diff_last_3s`: contribution `+0.006027`
- `lag_00__CT5__duck_amount`: contribution `+0.005561`
- `lag_11__CT_place_LIBRARY`: contribution `+0.005333`

Top utility-only movements:
- `lag_03__CT_utility_damage_last_5s`: contribution `+0.002373`
- `lag_05__T_A_site_active_infernos`: contribution `+0.001763`

### tick `90575`, seconds `74.50`, LSTM delta `+0.0806`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.009396`
- `lag_01__T5__is_scoped`: contribution `+0.008936`
- `lag_00__CT_shots_fired_sum`: contribution `-0.007386`
- `lag_03__CT_flash_duration_sum`: contribution `+0.004884`
- `lag_03__CT1__flash_duration`: contribution `+0.004752`

Top utility-only movements:
- `lag_03__CT_flash_duration_sum`: contribution `+0.004884`
- `lag_03__CT1__flash_duration`: contribution `+0.004752`
- `lag_03__CT4__flash_duration`: contribution `+0.003672`
- `lag_06__CT1__flash_duration`: contribution `+0.002431`
- `lag_00__CT_flash_duration_sum`: contribution `-0.001925`

### tick `90767`, seconds `77.50`, LSTM delta `-0.0702`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `-0.009396`
- `lag_00__CT1__flash_duration`: contribution `-0.008242`
- `lag_00__kill_diff_last_3s`: contribution `-0.006027`
- `lag_00__CT_flash_duration_sum`: contribution `-0.004437`
- `lag_00__CT_shots_fired_sum`: contribution `-0.002954`

Top utility-only movements:
- `lag_00__CT1__flash_duration`: contribution `-0.008242`
- `lag_00__CT_flash_duration_sum`: contribution `-0.004437`
- `lag_06__CT3__flash_duration`: contribution `-0.001659`
