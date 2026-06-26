# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-eternal-fire-vs-falcons-bo3-Bm3FkXiO5h_cvpKxUnOmaW/eternal-fire-vs-falcons-m1-inferno.csv`
- round_num: `4`

## Largest probability jumps

- tick `34406`, seconds `0.50`, LSTM `0.0473`, delta `-0.0535`
- tick `35494`, seconds `17.50`, LSTM `0.0364`, delta `-0.0280`
- tick `35526`, seconds `18.00`, LSTM `0.0238`, delta `-0.0126`
- tick `34438`, seconds `1.00`, LSTM `0.0352`, delta `-0.0121`
- tick `35814`, seconds `22.50`, LSTM `0.0125`, delta `-0.0109`
- tick `34854`, seconds `7.50`, LSTM `0.0457`, delta `-0.0106`
- tick `35174`, seconds `12.50`, LSTM `0.0574`, delta `+0.0104`
- tick `35206`, seconds `13.00`, LSTM `0.0477`, delta `-0.0098`
- tick `34822`, seconds `7.00`, LSTM `0.0563`, delta `-0.0085`
- tick `35462`, seconds `17.00`, LSTM `0.0644`, delta `+0.0081`

## Top 15 local ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000497`, |coef| `0.000497`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000446`, |coef| `0.000446`
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000405`, |coef| `0.000405`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000403`, |coef| `0.000403`
- `lag_01__centroid_distance_xy`: coefficient `-0.000382`, |coef| `0.000382`
- `lag_00__T_velocity_mean`: coefficient `-0.000362`, |coef| `0.000362`
- `lag_00__CT_velocity_mean`: coefficient `-0.000342`, |coef| `0.000342`
- `lag_01__T_round_start_equip_sum`: coefficient `-0.000330`, |coef| `0.000330`
- `lag_01__smoke_inv_diff`: coefficient `0.000329`, |coef| `0.000329`
- `lag_01__utility_inv_diff`: coefficient `0.000291`, |coef| `0.000291`
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.000268`, |coef| `0.000268`
- `lag_00__T_place_UPSTAIRS`: coefficient `-0.000268`, |coef| `0.000268`
- `lag_01__molly_inv_diff`: coefficient `0.000256`, |coef| `0.000256`
- `lag_01__equip_diff`: coefficient `0.000241`, |coef| `0.000241`
- `lag_01__CT_mean_X`: coefficient `-0.000238`, |coef| `0.000238`

## Top 10 utility ridge features

- `lag_01__smoke_inv_diff`: coefficient `0.000329` (raises CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000291` (raises CT win probability)
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.000268` (lowers CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000256` (raises CT win probability)
- `lag_01__T_smoke_inv`: coefficient `-0.000221` (lowers CT win probability)
- `lag_01__T_molly_inv`: coefficient `-0.000191` (lowers CT win probability)
- `lag_01__T_utility_inv`: coefficient `-0.000180` (lowers CT win probability)
- `lag_01__T4__smoke`: coefficient `-0.000178` (lowers CT win probability)
- `lag_02__T5__flash_duration`: coefficient `-0.000177` (lowers CT win probability)
- `lag_01__T1__smoke`: coefficient `-0.000175` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000497` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000446` (lowers CT win probability)
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000405` (lowers CT win probability)
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000403` (lowers CT win probability)
- `lag_01__centroid_distance_xy`: coefficient `-0.000382` (lowers CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000362` (lowers CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000342` (lowers CT win probability)
- `lag_01__T_round_start_equip_sum`: coefficient `-0.000330` (lowers CT win probability)
- `lag_00__T_place_UPSTAIRS`: coefficient `-0.000268` (lowers CT win probability)
- `lag_01__equip_diff`: coefficient `0.000241` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `34406`, seconds `0.50`, LSTM delta `-0.0535`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.002378`
- `lag_01__T_place_TSPAWN`: contribution `-0.001976`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.001664`
- `lag_01__T_closest_enemy_dist`: contribution `-0.001621`
- `lag_01__centroid_distance_xy`: contribution `-0.001461`

Top utility-only movements:
- `lag_01__smoke_inv_diff`: contribution `-0.001047`
- `lag_01__utility_inv_diff`: contribution `-0.000831`
- `lag_01__molly_inv_diff`: contribution `-0.000714`
- `lag_01__T_smoke_inv`: contribution `-0.000504`
- `lag_01__T_molly_inv`: contribution `-0.000433`

### tick `35494`, seconds `17.50`, LSTM delta `-0.0280`

Top all feature movements:
- `lag_00__T_place_UPSTAIRS`: contribution `-0.004522`
- `lag_00__T_utility_damage_last_5s`: contribution `-0.001839`
- `lag_02__T5__flash_duration`: contribution `-0.001229`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.000608`
- `lag_10__CT2__duck_amount`: contribution `-0.000541`

Top utility-only movements:
- `lag_00__T_utility_damage_last_5s`: contribution `-0.001839`
- `lag_02__T5__flash_duration`: contribution `-0.001229`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.000608`
- `lag_02__CT_utility_damage_last_5s`: contribution `-0.000376`
- `lag_02__T_A_site_active_infernos`: contribution `-0.000375`

### tick `35526`, seconds `18.00`, LSTM delta `-0.0126`

Top all feature movements:
- `lag_00__T_utility_damage_last_5s`: contribution `-0.001954`
- `lag_03__T5__flash_duration`: contribution `-0.000772`
- `lag_01__T_utility_damage_last_5s`: contribution `-0.000716`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.000646`
- `lag_01__CT2__duck_amount`: contribution `-0.000576`

Top utility-only movements:
- `lag_00__T_utility_damage_last_5s`: contribution `-0.001954`
- `lag_03__T5__flash_duration`: contribution `-0.000772`
- `lag_01__T_utility_damage_last_5s`: contribution `-0.000716`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.000646`
- `lag_01__utility_damage_diff_last_5s`: contribution `-0.000223`

### tick `34438`, seconds `1.00`, LSTM delta `-0.0121`

Top all feature movements:
- `lag_02__CT_place_CTSPAWN`: contribution `-0.001027`
- `lag_02__T_place_TSPAWN`: contribution `-0.000800`
- `lag_02__CT_closest_enemy_dist`: contribution `-0.000715`
- `lag_02__T_closest_enemy_dist`: contribution `-0.000687`
- `lag_02__centroid_distance_xy`: contribution `-0.000621`

Top utility-only movements:
- `lag_02__smoke_inv_diff`: contribution `-0.000471`
- `lag_02__utility_inv_diff`: contribution `-0.000380`
- `lag_02__molly_inv_diff`: contribution `-0.000324`
- `lag_02__T_smoke_inv`: contribution `-0.000210`
- `lag_02__T_molly_inv`: contribution `-0.000186`

### tick `35814`, seconds `22.50`, LSTM delta `-0.0109`

Top all feature movements:
- `lag_00__T_utility_damage_last_5s`: contribution `+0.001839`
- `lag_06__T_place_UPSTAIRS`: contribution `-0.001705`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.000608`
- `lag_05__T3__flash_duration`: contribution `-0.000550`
- `lag_10__CT2__duck_amount`: contribution `-0.000541`

Top utility-only movements:
- `lag_00__T_utility_damage_last_5s`: contribution `+0.001839`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.000608`
- `lag_05__T3__flash_duration`: contribution `-0.000550`
- `lag_10__T_utility_damage_last_5s`: contribution `-0.000434`
- `lag_12__T5__flash_duration`: contribution `-0.000415`
