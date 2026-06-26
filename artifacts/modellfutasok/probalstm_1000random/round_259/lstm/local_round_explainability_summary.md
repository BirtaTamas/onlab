# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-natus-vincere-vs-3dmax-bo3-JB3JZO-5zNCohi5tAgyHtq/natus-vincere-vs-3dmax-m2-inferno.csv`
- round_num: `4`

## Largest probability jumps

- tick `33152`, seconds `0.50`, LSTM `0.0718`, delta `-0.0653`
- tick `33568`, seconds `7.00`, LSTM `0.1730`, delta `-0.0463`
- tick `36224`, seconds `48.50`, LSTM `0.1057`, delta `-0.0394`
- tick `33600`, seconds `7.50`, LSTM `0.1391`, delta `-0.0340`
- tick `36288`, seconds `49.50`, LSTM `0.0711`, delta `-0.0327`
- tick `33472`, seconds `5.50`, LSTM `0.1989`, delta `+0.0320`
- tick `33440`, seconds `5.00`, LSTM `0.1669`, delta `+0.0312`
- tick `35040`, seconds `30.00`, LSTM `0.1074`, delta `-0.0305`
- tick `35616`, seconds `39.00`, LSTM `0.1201`, delta `-0.0293`
- tick `35744`, seconds `41.00`, LSTM `0.1562`, delta `+0.0283`

## Top 15 local ridge features

- `lag_00__T_utility_damage_last_5s`: coefficient `-0.000950`, |coef| `0.000950`
- `lag_00__T_place_LOWERMID`: coefficient `0.000944`, |coef| `0.000944`
- `lag_01__T_utility_damage_last_5s`: coefficient `-0.000803`, |coef| `0.000803`
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000782`, |coef| `0.000782`
- `lag_02__T_utility_damage_last_5s`: coefficient `-0.000619`, |coef| `0.000619`
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000601`, |coef| `0.000601`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000574`, |coef| `0.000574`
- `lag_00__T_place_SECONDMID`: coefficient `0.000539`, |coef| `0.000539`
- `lag_01__utility_damage_diff_last_5s`: coefficient `0.000527`, |coef| `0.000527`
- `lag_00__CT5__flash_duration`: coefficient `0.000522`, |coef| `0.000522`
- `lag_03__T_utility_damage_last_5s`: coefficient `-0.000518`, |coef| `0.000518`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000516`, |coef| `0.000516`
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000515`, |coef| `0.000515`
- `lag_01__T_place_LOWERMID`: coefficient `0.000494`, |coef| `0.000494`
- `lag_01__centroid_distance_xy`: coefficient `-0.000492`, |coef| `0.000492`

## Top 10 utility ridge features

- `lag_00__T_utility_damage_last_5s`: coefficient `-0.000950` (lowers CT win probability)
- `lag_01__T_utility_damage_last_5s`: coefficient `-0.000803` (lowers CT win probability)
- `lag_02__T_utility_damage_last_5s`: coefficient `-0.000619` (lowers CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000601` (raises CT win probability)
- `lag_01__utility_damage_diff_last_5s`: coefficient `0.000527` (raises CT win probability)
- `lag_00__CT5__flash_duration`: coefficient `0.000522` (raises CT win probability)
- `lag_03__T_utility_damage_last_5s`: coefficient `-0.000518` (lowers CT win probability)
- `lag_04__T_utility_damage_last_5s`: coefficient `-0.000486` (lowers CT win probability)
- `lag_00__T_B_site_active_infernos`: coefficient `0.000486` (raises CT win probability)
- `lag_07__T_B_site_active_infernos`: coefficient `0.000433` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_LOWERMID`: coefficient `0.000944` (raises CT win probability)
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000782` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000574` (lowers CT win probability)
- `lag_00__T_place_SECONDMID`: coefficient `0.000539` (raises CT win probability)
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000516` (lowers CT win probability)
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000515` (lowers CT win probability)
- `lag_01__T_place_LOWERMID`: coefficient `0.000494` (raises CT win probability)
- `lag_01__centroid_distance_xy`: coefficient `-0.000492` (lowers CT win probability)
- `lag_08__T5__duck_amount`: coefficient `0.000457` (raises CT win probability)
- `lag_00__CT5__duck_amount`: coefficient `-0.000445` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `33152`, seconds `0.50`, LSTM delta `-0.0653`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.003742`
- `lag_01__T_place_TSPAWN`: contribution `-0.002543`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.002148`
- `lag_01__T_closest_enemy_dist`: contribution `-0.002104`
- `lag_01__centroid_distance_xy`: contribution `-0.001912`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `-0.000985`
- `lag_01__molly_inv_diff`: contribution `-0.000946`
- `lag_01__smoke_inv_diff`: contribution `-0.000893`
- `lag_01__T3__utility_total`: contribution `-0.000617`
- `lag_01__T_molly_inv`: contribution `-0.000604`

### tick `33568`, seconds `7.00`, LSTM delta `-0.0463`

Top all feature movements:
- `lag_00__T_place_LOWERMID`: contribution `-0.006283`
- `lag_09__CT_place_LIBRARY`: contribution `-0.004800`
- `lag_04__T_place_LOWERMID`: contribution `-0.003211`
- `lag_05__CT_place_LIBRARY`: contribution `-0.002345`
- `lag_00__T_place_SECONDMID`: contribution `+0.001766`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `36224`, seconds `48.50`, LSTM delta `-0.0394`

Top all feature movements:
- `lag_04__CT_shots_fired_sum`: contribution `-0.003355`
- `lag_08__CT_place_BALCONY`: contribution `-0.001879`
- `lag_04__CT5__shots_fired`: contribution `-0.001836`
- `lag_02__CT_place_LIBRARY`: contribution `-0.001497`
- `lag_06__CT2__duck_amount`: contribution `-0.001342`

Top utility-only movements:
- `lag_10__CT5__flash_duration`: contribution `-0.001265`
- `lag_07__T_B_site_active_infernos`: contribution `-0.001224`
- `lag_07__T_active_infernos`: contribution `-0.000638`

### tick `33600`, seconds `7.50`, LSTM delta `-0.0340`

Top all feature movements:
- `lag_00__T_place_LOWERMID`: contribution `-0.006283`
- `lag_01__T_place_LOWERMID`: contribution `-0.003285`
- `lag_03__CT_place_RUINS`: contribution `+0.002880`
- `lag_09__CT_place_LIBRARY`: contribution `-0.002400`
- `lag_05__CT_place_LIBRARY`: contribution `-0.002345`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `36288`, seconds `49.50`, LSTM delta `-0.0327`

Top all feature movements:
- `lag_00__CT5__flash_duration`: contribution `-0.003368`
- `lag_04__CT_place_LIBRARY`: contribution `+0.001682`
- `lag_07__CT_place_BALCONY`: contribution `-0.001123`
- `lag_07__CT_shots_fired_sum`: contribution `-0.001032`
- `lag_06__CT2__duck_amount`: contribution `-0.001004`

Top utility-only movements:
- `lag_00__CT5__flash_duration`: contribution `-0.003368`
- `lag_00__CT_flash_duration_sum`: contribution `-0.000762`
- `lag_01__T_B_site_active_infernos`: contribution `-0.000674`
