# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-the-mongolz-vs-aurora-bo3-BL3Rcvf733R8cy7TtLY5HE/the-mongolz-vs-aurora-m1-dust2.csv`
- round_num: `24`

## Largest probability jumps

- tick `206639`, seconds `87.50`, LSTM `0.0179`, delta `-0.0750`
- tick `201071`, seconds `0.50`, LSTM `0.2036`, delta `-0.0666`
- tick `202575`, seconds `24.00`, LSTM `0.1992`, delta `-0.0379`
- tick `205775`, seconds `74.00`, LSTM `0.2289`, delta `+0.0348`
- tick `205455`, seconds `69.00`, LSTM `0.1915`, delta `-0.0342`
- tick `201327`, seconds `4.50`, LSTM `0.1740`, delta `-0.0337`
- tick `203343`, seconds `36.00`, LSTM `0.1962`, delta `+0.0289`
- tick `201135`, seconds `1.50`, LSTM `0.1569`, delta `-0.0283`
- tick `203023`, seconds `31.00`, LSTM `0.2058`, delta `-0.0280`
- tick `205807`, seconds `74.50`, LSTM `0.2015`, delta `-0.0273`

## Top 15 local ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.000954`, |coef| `0.000954`
- `lag_00__CT_place_BDOORS`: coefficient `0.000936`, |coef| `0.000936`
- `lag_00__T_damage_last_5s`: coefficient `-0.000836`, |coef| `0.000836`
- `lag_00__damage_diff_last_5s`: coefficient `0.000766`, |coef| `0.000766`
- `lag_00__CT2__alive`: coefficient `0.000763`, |coef| `0.000763`
- `lag_00__CT2__hp`: coefficient `0.000740`, |coef| `0.000740`
- `lag_00__T_velocity_mean`: coefficient `-0.000734`, |coef| `0.000734`
- `lag_00__CT2__armor`: coefficient `0.000713`, |coef| `0.000713`
- `lag_14__CT2__duck_amount`: coefficient `-0.000708`, |coef| `0.000708`
- `lag_09__T3__is_walking`: coefficient `0.000694`, |coef| `0.000694`
- `lag_00__kill_diff_last_3s`: coefficient `0.000656`, |coef| `0.000656`
- `lag_10__T3__molly`: coefficient `0.000632`, |coef| `0.000632`
- `lag_00__T5__molly`: coefficient `0.000626`, |coef| `0.000626`
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000621`, |coef| `0.000621`
- `lag_15__T3__is_walking`: coefficient `-0.000618`, |coef| `0.000618`

## Top 10 utility ridge features

- `lag_10__T3__molly`: coefficient `0.000632` (raises CT win probability)
- `lag_00__T5__molly`: coefficient `0.000626` (raises CT win probability)
- `lag_07__T_B_site_active_infernos`: coefficient `-0.000615` (lowers CT win probability)
- `lag_07__T_active_infernos`: coefficient `-0.000603` (lowers CT win probability)
- `lag_00__T_flashes_last_5s`: coefficient `-0.000598` (lowers CT win probability)
- `lag_06__T_B_site_active_infernos`: coefficient `-0.000567` (lowers CT win probability)
- `lag_06__T_active_infernos`: coefficient `-0.000539` (lowers CT win probability)
- `lag_10__T_flashes_last_5s`: coefficient `0.000527` (raises CT win probability)
- `lag_09__T3__molly`: coefficient `0.000507` (raises CT win probability)
- `lag_04__T_active_infernos`: coefficient `-0.000496` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.000954` (lowers CT win probability)
- `lag_00__CT_place_BDOORS`: coefficient `0.000936` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.000836` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000766` (raises CT win probability)
- `lag_00__CT2__alive`: coefficient `0.000763` (raises CT win probability)
- `lag_00__CT2__hp`: coefficient `0.000740` (raises CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000734` (lowers CT win probability)
- `lag_00__CT2__armor`: coefficient `0.000713` (raises CT win probability)
- `lag_14__CT2__duck_amount`: coefficient `-0.000708` (lowers CT win probability)
- `lag_09__T3__is_walking`: coefficient `0.000694` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `206639`, seconds `87.50`, LSTM delta `-0.0750`

Top all feature movements:
- `lag_00__CT_place_BDOORS`: contribution `-0.004501`
- `lag_00__T_kills_last_3s`: contribution `-0.003022`
- `lag_14__CT2__duck_amount`: contribution `-0.002253`
- `lag_04__CT4__duck_amount`: contribution `-0.002203`
- `lag_00__T_shots_fired_sum`: contribution `-0.002007`

Top utility-only movements:
- `lag_07__T_B_site_active_infernos`: contribution `-0.001739`
- `lag_10__T3__molly`: contribution `-0.001404`
- `lag_07__T_active_infernos`: contribution `-0.001255`
- `lag_02__T1__smoke`: contribution `-0.001044`

### tick `201071`, seconds `0.50`, LSTM delta `-0.0666`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.002969`
- `lag_01__T_place_TSPAWN`: contribution `-0.002738`
- `lag_00__T_velocity_mean`: contribution `-0.002569`
- `lag_00__CT_velocity_mean`: contribution `-0.001681`
- `lag_01__T_closest_enemy_dist`: contribution `-0.001124`

Top utility-only movements:
- `lag_01__T_flash_inv`: contribution `-0.000903`
- `lag_01__flash_inv_diff`: contribution `-0.000875`
- `lag_01__T2__utility_total`: contribution `-0.000840`
- `lag_01__utility_inv_diff`: contribution `-0.000708`
- `lag_01__T2__flash`: contribution `-0.000666`

### tick `202575`, seconds `24.00`, LSTM delta `-0.0379`

Top all feature movements:
- `lag_00__CT_place_ARAMP`: contribution `-0.003375`
- `lag_06__CT3__duck_amount`: contribution `-0.001809`
- `lag_09__T3__is_walking`: contribution `+0.001612`
- `lag_00__T4__is_walking`: contribution `-0.001334`
- `lag_05__CT1__is_walking`: contribution `-0.001266`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `205775`, seconds `74.00`, LSTM delta `+0.0348`

Top all feature movements:
- `lag_00__T_flashes_last_5s`: contribution `+0.005419`
- `lag_10__T_flashes_last_5s`: contribution `+0.004772`
- `lag_00__CT_place_BDOORS`: contribution `+0.004501`
- `lag_09__T_place_TUNNELSTAIRS`: contribution `+0.003358`
- `lag_05__T4__duck_amount`: contribution `+0.001603`

Top utility-only movements:
- `lag_00__T_flashes_last_5s`: contribution `+0.005419`
- `lag_10__T_flashes_last_5s`: contribution `+0.004772`

### tick `205455`, seconds `69.00`, LSTM delta `-0.0342`

Top all feature movements:
- `lag_00__T_flashes_last_5s`: contribution `-0.005419`
- `lag_09__T_place_TUNNELSTAIRS`: contribution `-0.003358`
- `lag_05__CT_place_EXTENDEDA`: contribution `-0.001874`
- `lag_02__T4__duck_amount`: contribution `-0.001718`
- `lag_15__T3__is_walking`: contribution `-0.001436`

Top utility-only movements:
- `lag_00__T_flashes_last_5s`: contribution `-0.005419`
