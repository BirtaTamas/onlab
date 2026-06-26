# Local Round Explainability

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-gamerlegion-vs-the-mongolz-bo3-zdjI5BKx0DIgDYoNAnfKpI/gamerlegion-vs-the-mongolz-m2-mirage.csv`
- round_num: `11`

## Largest probability jumps

- tick `78177`, seconds `88.00`, LSTM `0.2552`, delta `-0.3250`
- tick `74689`, seconds `33.50`, LSTM `0.8559`, delta `+0.2542`
- tick `78817`, seconds `98.00`, LSTM `0.2586`, delta `+0.2216`
- tick `78209`, seconds `88.50`, LSTM `0.0785`, delta `-0.1767`
- tick `78081`, seconds `86.50`, LSTM `0.5893`, delta `+0.1509`
- tick `77825`, seconds `82.50`, LSTM `0.2764`, delta `-0.1277`
- tick `77761`, seconds `81.50`, LSTM `0.4394`, delta `-0.1178`
- tick `73697`, seconds `18.00`, LSTM `0.6102`, delta `+0.1115`
- tick `77281`, seconds `74.00`, LSTM `0.5397`, delta `-0.0924`
- tick `78049`, seconds `86.00`, LSTM `0.4384`, delta `+0.0785`

## Top 15 local ridge features

- `lag_13__T_utility_damage_last_5s`: coefficient `-0.003827`, |coef| `0.003827`
- `lag_00__T_bomb_zone_count`: coefficient `-0.003682`, |coef| `0.003682`
- `lag_00__kill_diff_last_3s`: coefficient `0.003289`, |coef| `0.003289`
- `lag_11__T_place_STAIRS`: coefficient `0.003254`, |coef| `0.003254`
- `lag_08__T_place_STAIRS`: coefficient `-0.003084`, |coef| `0.003084`
- `lag_00__T1__is_scoped`: coefficient `0.002853`, |coef| `0.002853`
- `lag_00__CT_kills_last_3s`: coefficient `0.002760`, |coef| `0.002760`
- `lag_00__damage_diff_last_5s`: coefficient `0.002661`, |coef| `0.002661`
- `lag_03__T_utility_damage_last_5s`: coefficient `-0.002526`, |coef| `0.002526`
- `lag_12__T_place_STAIRS`: coefficient `0.002463`, |coef| `0.002463`
- `lag_00__CT_place_STAIRS`: coefficient `0.002446`, |coef| `0.002446`
- `lag_13__utility_damage_diff_last_5s`: coefficient `0.002429`, |coef| `0.002429`
- `lag_08__T_bomb_zone_count`: coefficient `-0.002421`, |coef| `0.002421`
- `lag_05__CT_duck_amount_mean`: coefficient `0.002242`, |coef| `0.002242`
- `lag_00__CT_place_TRUCK`: coefficient `0.002173`, |coef| `0.002173`

## Top 10 utility ridge features

- `lag_13__T_utility_damage_last_5s`: coefficient `-0.003827` (lowers CT win probability)
- `lag_03__T_utility_damage_last_5s`: coefficient `-0.002526` (lowers CT win probability)
- `lag_13__utility_damage_diff_last_5s`: coefficient `0.002429` (raises CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `0.002032` (raises CT win probability)
- `lag_03__utility_damage_diff_last_5s`: coefficient `0.001605` (raises CT win probability)
- `lag_04__T_utility_damage_last_5s`: coefficient `-0.001471` (lowers CT win probability)
- `lag_08__CT1__flash_duration`: coefficient `0.001456` (raises CT win probability)
- `lag_05__CT3__flash_duration`: coefficient `0.001318` (raises CT win probability)
- `lag_13__CT3__flash_duration`: coefficient `0.001232` (raises CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `0.001196` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_bomb_zone_count`: coefficient `-0.003682` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003289` (raises CT win probability)
- `lag_11__T_place_STAIRS`: coefficient `0.003254` (raises CT win probability)
- `lag_08__T_place_STAIRS`: coefficient `-0.003084` (lowers CT win probability)
- `lag_00__T1__is_scoped`: coefficient `0.002853` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002760` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002661` (raises CT win probability)
- `lag_12__T_place_STAIRS`: coefficient `0.002463` (raises CT win probability)
- `lag_00__CT_place_STAIRS`: coefficient `0.002446` (raises CT win probability)
- `lag_08__T_bomb_zone_count`: coefficient `-0.002421` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `78177`, seconds `88.00`, LSTM delta `-0.3250`

Top all feature movements:
- `lag_11__T_place_STAIRS`: contribution `-0.062294`
- `lag_03__T_utility_damage_last_5s`: contribution `-0.020193`
- `lag_00__T1__is_scoped`: contribution `-0.016301`
- `lag_07__CT_place_STAIRS`: contribution `-0.015443`
- `lag_13__CT_place_JUNGLE`: contribution `-0.012532`

Top utility-only movements:
- `lag_03__T_utility_damage_last_5s`: contribution `-0.020193`
- `lag_08__CT1__flash_duration`: contribution `-0.011490`
- `lag_13__CT3__flash_duration`: contribution `-0.009195`
- `lag_03__utility_damage_diff_last_5s`: contribution `-0.008116`

### tick `74689`, seconds `33.50`, LSTM delta `+0.2542`

Top all feature movements:
- `lag_00__CT_place_STAIRS`: contribution `+0.019039`
- `lag_00__CT_place_TRUCK`: contribution `+0.014016`
- `lag_09__CT_place_JUNGLE`: contribution `+0.011226`
- `lag_00__CT_kills_last_3s`: contribution `+0.007970`
- `lag_00__kill_diff_last_3s`: contribution `+0.007917`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `78817`, seconds `98.00`, LSTM delta `+0.2216`

Top all feature movements:
- `lag_13__T_utility_damage_last_5s`: contribution `+0.030601`
- `lag_00__T_bomb_zone_count`: contribution `+0.021435`
- `lag_08__T_bomb_zone_count`: contribution `+0.014092`
- `lag_05__CT_duck_amount_mean`: contribution `+0.013424`
- `lag_13__utility_damage_diff_last_5s`: contribution `+0.012279`

Top utility-only movements:
- `lag_13__T_utility_damage_last_5s`: contribution `+0.030601`
- `lag_13__utility_damage_diff_last_5s`: contribution `+0.012279`

### tick `78209`, seconds `88.50`, LSTM delta `-0.1767`

Top all feature movements:
- `lag_12__T_place_STAIRS`: contribution `-0.047155`
- `lag_00__CT_place_STAIRS`: contribution `-0.019039`
- `lag_04__T_utility_damage_last_5s`: contribution `-0.011764`
- `lag_00__kill_diff_last_3s`: contribution `-0.007917`
- `lag_11__T_bomb_zone_count`: contribution `-0.007865`

Top utility-only movements:
- `lag_04__T_utility_damage_last_5s`: contribution `-0.011764`
- `lag_04__utility_damage_diff_last_5s`: contribution `-0.004734`
- `lag_07__T_A_site_active_infernos`: contribution `-0.002341`

### tick `78081`, seconds `86.50`, LSTM delta `+0.1509`

Top all feature movements:
- `lag_08__T_place_STAIRS`: contribution `+0.059044`
- `lag_00__CT_kills_last_3s`: contribution `+0.007970`
- `lag_00__kill_diff_last_3s`: contribution `+0.007917`
- `lag_11__T_bomb_zone_count`: contribution `+0.007865`
- `lag_05__CT1__flash_duration`: contribution `+0.006990`

Top utility-only movements:
- `lag_05__CT1__flash_duration`: contribution `+0.006990`
- `lag_00__T_utility_damage_last_5s`: contribution `+0.003693`
- `lag_10__CT3__flash_duration`: contribution `+0.002291`
