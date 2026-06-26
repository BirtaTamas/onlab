# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-mouz-vs-vitality-bo5-g3-5jFl1QSVPqll-eeCKIE/mouz-vs-vitality-m1-inferno.csv`
- round_num: `8`

## Largest probability jumps

- tick `78060`, seconds `29.50`, LSTM `0.0259`, delta `-0.1220`
- tick `76204`, seconds `0.50`, LSTM `0.1439`, delta `-0.0587`
- tick `78028`, seconds `29.00`, LSTM `0.1479`, delta `+0.0412`
- tick `77804`, seconds `25.50`, LSTM `0.0928`, delta `-0.0344`
- tick `76492`, seconds `5.00`, LSTM `0.1899`, delta `+0.0323`
- tick `77868`, seconds `26.50`, LSTM `0.0651`, delta `-0.0299`
- tick `77996`, seconds `28.50`, LSTM `0.1067`, delta `+0.0238`
- tick `81740`, seconds `87.00`, LSTM `0.0031`, delta `-0.0201`
- tick `76588`, seconds `6.50`, LSTM `0.1490`, delta `-0.0200`
- tick `76876`, seconds `11.00`, LSTM `0.1461`, delta `-0.0184`

## Top 15 local ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.001663`, |coef| `0.001663`
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.000709`, |coef| `0.000709`
- `lag_00__CT_flashes_last_5s`: coefficient `-0.000681`, |coef| `0.000681`
- `lag_04__CT3__shots_fired`: coefficient `-0.000543`, |coef| `0.000543`
- `lag_00__T_place_LOWERMID`: coefficient `0.000526`, |coef| `0.000526`
- `lag_08__T_utility_damage_last_5s`: coefficient `-0.000487`, |coef| `0.000487`
- `lag_03__T2__flash_duration`: coefficient `0.000451`, |coef| `0.000451`
- `lag_03__CT3__shots_fired`: coefficient `-0.000442`, |coef| `0.000442`
- `lag_02__T4__shots_fired`: coefficient `0.000437`, |coef| `0.000437`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000410`, |coef| `0.000410`
- `lag_04__T_shots_fired_sum`: coefficient `-0.000400`, |coef| `0.000400`
- `lag_14__T5__duck_amount`: coefficient `0.000391`, |coef| `0.000391`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000389`, |coef| `0.000389`
- `lag_05__T1__flash_duration`: coefficient `-0.000387`, |coef| `0.000387`
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000383`, |coef| `0.000383`

## Top 10 utility ridge features

- `lag_00__T_utility_damage_last_5s`: coefficient `-0.000709` (lowers CT win probability)
- `lag_00__CT_flashes_last_5s`: coefficient `-0.000681` (lowers CT win probability)
- `lag_08__T_utility_damage_last_5s`: coefficient `-0.000487` (lowers CT win probability)
- `lag_03__T2__flash_duration`: coefficient `0.000451` (raises CT win probability)
- `lag_05__T1__flash_duration`: coefficient `-0.000387` (lowers CT win probability)
- `lag_01__T_utility_damage_last_5s`: coefficient `-0.000367` (lowers CT win probability)
- `lag_05__CT3__flash_duration`: coefficient `-0.000362` (lowers CT win probability)
- `lag_07__T_active_infernos`: coefficient `-0.000344` (lowers CT win probability)
- `lag_05__T2__flash_duration`: coefficient `-0.000331` (lowers CT win probability)
- `lag_06__T_utility_damage_last_5s`: coefficient `-0.000328` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.001663` (raises CT win probability)
- `lag_04__CT3__shots_fired`: coefficient `-0.000543` (lowers CT win probability)
- `lag_00__T_place_LOWERMID`: coefficient `0.000526` (raises CT win probability)
- `lag_03__CT3__shots_fired`: coefficient `-0.000442` (lowers CT win probability)
- `lag_02__T4__shots_fired`: coefficient `0.000437` (raises CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000410` (lowers CT win probability)
- `lag_04__T_shots_fired_sum`: coefficient `-0.000400` (lowers CT win probability)
- `lag_14__T5__duck_amount`: coefficient `0.000391` (raises CT win probability)
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000389` (lowers CT win probability)
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000383` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `78060`, seconds `29.50`, LSTM delta `-0.1220`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.032356`
- `lag_04__T_shots_fired_sum`: contribution `-0.002401`
- `lag_03__T2__flash_duration`: contribution `-0.002381`
- `lag_08__T_utility_damage_last_5s`: contribution `-0.002363`
- `lag_05__T1__flash_duration`: contribution `-0.002235`

Top utility-only movements:
- `lag_03__T2__flash_duration`: contribution `-0.002381`
- `lag_08__T_utility_damage_last_5s`: contribution `-0.002363`
- `lag_05__T1__flash_duration`: contribution `-0.002235`
- `lag_05__CT3__flash_duration`: contribution `-0.002012`
- `lag_06__T_utility_damage_last_5s`: contribution `-0.001965`

### tick `76204`, seconds `0.50`, LSTM delta `-0.0587`

Top all feature movements:
- `lag_00__CT_flashes_last_5s`: contribution `-0.007492`
- `lag_01__T_place_TSPAWN`: contribution `-0.001814`
- `lag_01__CT_place_CTSPAWN`: contribution `-0.001718`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.001577`
- `lag_01__T_closest_enemy_dist`: contribution `-0.001566`

Top utility-only movements:
- `lag_00__CT_flashes_last_5s`: contribution `-0.007492`
- `lag_01__molly_inv_diff`: contribution `-0.000629`
- `lag_01__T_smoke_inv`: contribution `-0.000526`
- `lag_01__utility_inv_diff`: contribution `-0.000466`
- `lag_01__flash_inv_diff`: contribution `-0.000363`

### tick `78028`, seconds `29.00`, LSTM delta `+0.0412`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.008089`
- `lag_03__CT3__shots_fired`: contribution `-0.001591`
- `lag_14__T5__duck_amount`: contribution `+0.001483`
- `lag_02__CT3__shots_fired`: contribution `-0.001370`
- `lag_02__T4__shots_fired`: contribution `+0.001349`

Top utility-only movements:
- `lag_04__CT3__flash_duration`: contribution `+0.001098`
- `lag_04__T1__flash_duration`: contribution `+0.001046`
- `lag_04__T2__flash_duration`: contribution `+0.000873`
- `lag_04__T_flash_duration_sum`: contribution `+0.000813`
- `lag_05__T_utility_damage_last_5s`: contribution `+0.000778`

### tick `77804`, seconds `25.50`, LSTM delta `-0.0344`

Top all feature movements:
- `lag_00__T_utility_damage_last_5s`: contribution `-0.003443`
- `lag_04__T_place_BRIDGE`: contribution `-0.001073`
- `lag_08__T_utility_damage_last_5s`: contribution `-0.001042`
- `lag_14__T_place_SECONDMID`: contribution `-0.000992`
- `lag_14__T_place_LOWERMID`: contribution `-0.000923`

Top utility-only movements:
- `lag_00__T_utility_damage_last_5s`: contribution `-0.003443`
- `lag_08__T_utility_damage_last_5s`: contribution `-0.001042`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.000785`
- `lag_07__T_active_infernos`: contribution `-0.000715`
- `lag_07__T_A_site_active_infernos`: contribution `-0.000681`

### tick `76492`, seconds `5.00`, LSTM delta `+0.0323`

Top all feature movements:
- `lag_00__T_place_LOWERMID`: contribution `+0.003497`
- `lag_01__T_place_LOWERMID`: contribution `+0.002263`
- `lag_06__CT_place_LIBRARY`: contribution `+0.001869`
- `lag_09__CT_flashes_last_5s`: contribution `+0.001835`
- `lag_01__CT_place_LIBRARY`: contribution `+0.001402`

Top utility-only movements:
- `lag_09__CT_flashes_last_5s`: contribution `+0.001835`
- `lag_10__CT4__smoke`: contribution `+0.000352`
- `lag_00__CT4__smoke`: contribution `+0.000342`
- `lag_02__CT4__smoke`: contribution `+0.000338`
- `lag_08__CT3__smoke`: contribution `+0.000319`
