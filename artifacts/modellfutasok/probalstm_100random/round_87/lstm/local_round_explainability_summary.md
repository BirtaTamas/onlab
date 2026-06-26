# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-3dmax-vs-mibr-bo3-O12tFfVag47APQdKBJkGZl/3dmax-vs-mibr-m2-ancient-p3.csv`
- round_num: `4`

## Largest probability jumps

- tick `35983`, seconds `61.00`, LSTM `0.1542`, delta `-0.3927`
- tick `37679`, seconds `87.50`, LSTM `0.0468`, delta `-0.3321`
- tick `33007`, seconds `14.50`, LSTM `0.3066`, delta `-0.2309`
- tick `36271`, seconds `65.50`, LSTM `0.2741`, delta `+0.2247`
- tick `33551`, seconds `23.00`, LSTM `0.5370`, delta `+0.1763`
- tick `34351`, seconds `35.50`, LSTM `0.6483`, delta `+0.1189`
- tick `34831`, seconds `43.00`, LSTM `0.5668`, delta `-0.1028`
- tick `37135`, seconds `79.00`, LSTM `0.4658`, delta `+0.0737`
- tick `33071`, seconds `15.50`, LSTM `0.2484`, delta `-0.0698`
- tick `36655`, seconds `71.50`, LSTM `0.3423`, delta `-0.0612`

## Top 15 local ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.006615`, |coef| `0.006615`
- `lag_00__kill_diff_last_3s`: coefficient `0.006102`, |coef| `0.006102`
- `lag_00__T_kills_last_3s`: coefficient `-0.006087`, |coef| `0.006087`
- `lag_00__closest_enemy_dist_diff`: coefficient `0.004795`, |coef| `0.004795`
- `lag_00__T_closest_enemy_dist`: coefficient `-0.004550`, |coef| `0.004550`
- `lag_00__CT_spread_xy`: coefficient `0.004321`, |coef| `0.004321`
- `lag_00__damage_diff_last_5s`: coefficient `0.004039`, |coef| `0.004039`
- `lag_00__T_damage_last_5s`: coefficient `-0.004037`, |coef| `0.004037`
- `lag_00__spread_diff`: coefficient `0.003666`, |coef| `0.003666`
- `lag_01__T_shots_fired_sum`: coefficient `-0.003364`, |coef| `0.003364`
- `lag_00__CT_place_ALLEY`: coefficient `0.003256`, |coef| `0.003256`
- `lag_11__bomb_events_last_5s`: coefficient `-0.003106`, |coef| `0.003106`
- `lag_00__CT4__alive`: coefficient `0.003063`, |coef| `0.003063`
- `lag_08__CT_B_site_active_infernos`: coefficient `-0.003059`, |coef| `0.003059`
- `lag_00__CT_velocity_mean`: coefficient `-0.003042`, |coef| `0.003042`

## Top 10 utility ridge features

- `lag_08__CT_B_site_active_infernos`: coefficient `-0.003059` (lowers CT win probability)
- `lag_00__CT5__molly`: coefficient `0.002258` (raises CT win probability)
- `lag_00__CT4__flash`: coefficient `0.002165` (raises CT win probability)
- `lag_11__CT4__molly`: coefficient `0.002125` (raises CT win probability)
- `lag_08__CT_active_infernos`: coefficient `-0.002040` (lowers CT win probability)
- `lag_05__T4__smoke`: coefficient `0.001882` (raises CT win probability)
- `lag_09__T_flashes_last_5s`: coefficient `-0.001861` (lowers CT win probability)
- `lag_00__CT5__utility_total`: coefficient `0.001770` (raises CT win probability)
- `lag_03__T_B_site_active_smokes`: coefficient `0.001751` (raises CT win probability)
- `lag_09__T_B_site_active_smokes`: coefficient `0.001748` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.006615` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.006102` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.006087` (lowers CT win probability)
- `lag_00__closest_enemy_dist_diff`: coefficient `0.004795` (raises CT win probability)
- `lag_00__T_closest_enemy_dist`: coefficient `-0.004550` (lowers CT win probability)
- `lag_00__CT_spread_xy`: coefficient `0.004321` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.004039` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.004037` (lowers CT win probability)
- `lag_00__spread_diff`: coefficient `0.003666` (raises CT win probability)
- `lag_01__T_shots_fired_sum`: coefficient `-0.003364` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `35983`, seconds `61.00`, LSTM delta `-0.3927`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.024797`
- `lag_00__T_kills_last_3s`: contribution `-0.019285`
- `lag_00__kill_diff_last_3s`: contribution `-0.014687`
- `lag_08__CT_B_site_active_infernos`: contribution `-0.010510`
- `lag_02__T_place_RAMP`: contribution `+0.010373`

Top utility-only movements:
- `lag_08__CT_B_site_active_infernos`: contribution `-0.010510`
- `lag_00__CT5__molly`: contribution `-0.005601`

### tick `37679`, seconds `87.50`, LSTM delta `-0.3321`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.024797`
- `lag_00__T_kills_last_3s`: contribution `-0.019285`
- `lag_00__CT_spread_xy`: contribution `-0.017090`
- `lag_11__CT_place_MAINHALL`: contribution `-0.015000`
- `lag_00__T_closest_enemy_dist`: contribution `-0.014701`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `33007`, seconds `14.50`, LSTM delta `-0.2309`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.054554`
- `lag_08__CT_B_site_active_infernos`: contribution `-0.021021`
- `lag_00__T_kills_last_3s`: contribution `-0.019285`
- `lag_09__T_flashes_last_5s`: contribution `-0.016860`
- `lag_00__kill_diff_last_3s`: contribution `-0.014687`

Top utility-only movements:
- `lag_08__CT_B_site_active_infernos`: contribution `-0.021021`
- `lag_09__T_flashes_last_5s`: contribution `-0.016860`
- `lag_08__CT_active_infernos`: contribution `-0.009400`
- `lag_08__active_infernos_total`: contribution `-0.005523`
- `lag_06__CT_B_site_active_infernos`: contribution `+0.002948`

### tick `36271`, seconds `65.50`, LSTM delta `+0.2247`

Top all feature movements:
- `lag_07__T_shots_fired_sum`: contribution `+0.024194`
- `lag_07__T4__shots_fired`: contribution `+0.020723`
- `lag_00__kill_diff_last_3s`: contribution `+0.014687`
- `lag_00__damage_diff_last_5s`: contribution `+0.011209`
- `lag_02__T_place_RAMP`: contribution `+0.010373`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `33551`, seconds `23.00`, LSTM delta `+0.1763`

Top all feature movements:
- `lag_09__T_flashes_last_5s`: contribution `+0.016860`
- `lag_00__kill_diff_last_3s`: contribution `+0.014687`
- `lag_12__T4__flash_duration`: contribution `+0.010160`
- `lag_00__T_shots_fired_sum`: contribution `+0.009919`
- `lag_00__CT_shots_fired_sum`: contribution `+0.008438`

Top utility-only movements:
- `lag_09__T_flashes_last_5s`: contribution `+0.016860`
- `lag_12__T4__flash_duration`: contribution `+0.010160`
- `lag_01__T4__flash_duration`: contribution `+0.007105`
- `lag_14__CT_B_site_active_infernos`: contribution `+0.004060`
