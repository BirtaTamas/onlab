# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-fnatic-vs-legacy-bo3-XoJZ8zL16kSaGnHRZrLL4s/legacy-vs-fnatic-m1-ancient.csv`
- round_num: `3`

## Largest probability jumps

- tick `27136`, seconds `101.00`, LSTM `0.7638`, delta `+0.2460`
- tick `27392`, seconds `105.00`, LSTM `0.8889`, delta `+0.2231`
- tick `23584`, seconds `45.50`, LSTM `0.5490`, delta `-0.1626`
- tick `21216`, seconds `8.50`, LSTM `0.3989`, delta `-0.1242`
- tick `21536`, seconds `13.50`, LSTM `0.5079`, delta `+0.1199`
- tick `23456`, seconds `43.50`, LSTM `0.6724`, delta `+0.1082`
- tick `25088`, seconds `69.00`, LSTM `0.4429`, delta `-0.0804`
- tick `27328`, seconds `104.00`, LSTM `0.6688`, delta `-0.0694`
- tick `21248`, seconds `9.00`, LSTM `0.3323`, delta `-0.0666`
- tick `23488`, seconds `44.00`, LSTM `0.7374`, delta `+0.0651`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002776`, |coef| `0.002776`
- `lag_00__CT_kills_last_3s`: coefficient `0.002099`, |coef| `0.002099`
- `lag_13__CT_place_TSIDELOWER`: coefficient `-0.001858`, |coef| `0.001858`
- `lag_00__damage_diff_last_5s`: coefficient `0.001839`, |coef| `0.001839`
- `lag_12__CT5__flash_duration`: coefficient `0.001767`, |coef| `0.001767`
- `lag_00__T_bomb_zone_count`: coefficient `-0.001596`, |coef| `0.001596`
- `lag_01__CT_place_MAINHALL`: coefficient `0.001565`, |coef| `0.001565`
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.001532`, |coef| `0.001532`
- `lag_06__CT5__flash_duration`: coefficient `-0.001450`, |coef| `0.001450`
- `lag_00__CT_place_TSIDELOWER`: coefficient `-0.001389`, |coef| `0.001389`
- `lag_04__T_bomb_zone_count`: coefficient `0.001371`, |coef| `0.001371`
- `lag_10__CT_place_TSIDEUPPER`: coefficient `-0.001367`, |coef| `0.001367`
- `lag_00__T_kills_last_3s`: coefficient `-0.001351`, |coef| `0.001351`
- `lag_13__CT3__is_walking`: coefficient `0.001348`, |coef| `0.001348`
- `lag_07__CT_place_MAINHALL`: coefficient `-0.001326`, |coef| `0.001326`

## Top 10 utility ridge features

- `lag_12__CT5__flash_duration`: coefficient `0.001767` (raises CT win probability)
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.001532` (lowers CT win probability)
- `lag_06__CT5__flash_duration`: coefficient `-0.001450` (lowers CT win probability)
- `lag_02__T_flashes_last_5s`: coefficient `-0.001247` (lowers CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000935` (raises CT win probability)
- `lag_14__CT5__flash_duration`: coefficient `0.000922` (raises CT win probability)
- `lag_12__CT_flash_duration_sum`: coefficient `0.000891` (raises CT win probability)
- `lag_07__CT5__flash_duration`: coefficient `-0.000848` (lowers CT win probability)
- `lag_11__CT5__flash_duration`: coefficient `0.000836` (raises CT win probability)
- `lag_14__CT3__molly`: coefficient `-0.000814` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002776` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002099` (raises CT win probability)
- `lag_13__CT_place_TSIDELOWER`: coefficient `-0.001858` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001839` (raises CT win probability)
- `lag_00__T_bomb_zone_count`: coefficient `-0.001596` (lowers CT win probability)
- `lag_01__CT_place_MAINHALL`: coefficient `0.001565` (raises CT win probability)
- `lag_00__CT_place_TSIDELOWER`: coefficient `-0.001389` (lowers CT win probability)
- `lag_04__T_bomb_zone_count`: coefficient `0.001371` (raises CT win probability)
- `lag_10__CT_place_TSIDEUPPER`: coefficient `-0.001367` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001351` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `27136`, seconds `101.00`, LSTM delta `+0.2460`

Top all feature movements:
- `lag_05__CT_place_TSIDELOWER`: contribution `+0.015793`
- `lag_12__CT5__flash_duration`: contribution `+0.013568`
- `lag_10__CT_place_TSIDEUPPER`: contribution `+0.010273`
- `lag_00__T_bomb_zone_count`: contribution `+0.009293`
- `lag_04__T_bomb_zone_count`: contribution `+0.007980`

Top utility-only movements:
- `lag_12__CT5__flash_duration`: contribution `+0.013568`
- `lag_12__CT_flash_duration_sum`: contribution `+0.003157`

### tick `27392`, seconds `105.00`, LSTM delta `+0.2231`

Top all feature movements:
- `lag_13__CT_place_TSIDELOWER`: contribution `+0.025241`
- `lag_00__CT_place_TSIDELOWER`: contribution `+0.018874`
- `lag_03__CT_place_TSIDELOWER`: contribution `+0.014127`
- `lag_06__CT5__flash_duration`: contribution `+0.011136`
- `lag_12__T_bomb_zone_count`: contribution `+0.006872`

Top utility-only movements:
- `lag_06__CT5__flash_duration`: contribution `+0.011136`
- `lag_06__CT_flash_duration_sum`: contribution `+0.002746`

### tick `23584`, seconds `45.50`, LSTM delta `-0.1626`

Top all feature movements:
- `lag_01__CT_place_MAINHALL`: contribution `-0.012951`
- `lag_07__CT_place_MAINHALL`: contribution `-0.010975`
- `lag_00__kill_diff_last_3s`: contribution `-0.006681`
- `lag_00__T_kills_last_3s`: contribution `-0.004279`
- `lag_01__T5__duck_amount`: contribution `-0.004182`

Top utility-only movements:
- `lag_05__T_B_site_active_infernos`: contribution `-0.001842`

### tick `21216`, seconds `8.50`, LSTM delta `-0.1242`

Top all feature movements:
- `lag_00__T_utility_damage_last_5s`: contribution `-0.017058`
- `lag_14__CT_place_UNKNOWN`: contribution `-0.012224`
- `lag_02__T_flashes_last_5s`: contribution `-0.011299`
- `lag_15__CT_place_UNKNOWN`: contribution `-0.010041`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.006584`

Top utility-only movements:
- `lag_00__T_utility_damage_last_5s`: contribution `-0.017058`
- `lag_02__T_flashes_last_5s`: contribution `-0.011299`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.006584`
- `lag_01__T2__flash_duration`: contribution `-0.003725`
- `lag_01__CT4__flash_duration`: contribution `-0.003715`

### tick `21536`, seconds `13.50`, LSTM delta `+0.1199`

Top all feature movements:
- `lag_00__T_utility_damage_last_5s`: contribution `+0.017058`
- `lag_02__T_flashes_last_5s`: contribution `+0.011299`
- `lag_10__T_utility_damage_last_5s`: contribution `+0.007675`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.006584`
- `lag_12__T_flashes_last_5s`: contribution `+0.005087`

Top utility-only movements:
- `lag_00__T_utility_damage_last_5s`: contribution `+0.017058`
- `lag_02__T_flashes_last_5s`: contribution `+0.011299`
- `lag_10__T_utility_damage_last_5s`: contribution `+0.007675`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.006584`
- `lag_12__T_flashes_last_5s`: contribution `+0.005087`
