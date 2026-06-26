# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-lynn-vision-bo3-6KVULP2-Gxo12lI67V9ZfV/chinggis-warriors-vs-lynn-vision-m3-ancient.csv`
- round_num: `6`

## Largest probability jumps

- tick `42801`, seconds `13.00`, LSTM `0.8121`, delta `+0.3548`
- tick `45489`, seconds `55.00`, LSTM `0.8155`, delta `+0.2181`
- tick `44977`, seconds `47.00`, LSTM `0.6594`, delta `-0.1894`
- tick `45233`, seconds `51.00`, LSTM `0.6378`, delta `-0.0684`
- tick `42833`, seconds `13.50`, LSTM `0.8606`, delta `+0.0485`
- tick `44945`, seconds `46.50`, LSTM `0.8488`, delta `+0.0467`
- tick `43761`, seconds `28.00`, LSTM `0.8336`, delta `+0.0383`
- tick `45265`, seconds `51.50`, LSTM `0.6047`, delta `-0.0332`
- tick `44593`, seconds `41.00`, LSTM `0.8309`, delta `+0.0324`
- tick `48913`, seconds `108.50`, LSTM `0.8489`, delta `-0.0320`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002773`, |coef| `0.002773`
- `lag_09__CT_place_TOPOFMID`: coefficient `-0.002515`, |coef| `0.002515`
- `lag_00__CT_kills_last_3s`: coefficient `0.002488`, |coef| `0.002488`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002437`, |coef| `0.002437`
- `lag_08__CT5__flash_duration`: coefficient `-0.002152`, |coef| `0.002152`
- `lag_09__CT_place_MIDDLE`: coefficient `0.002117`, |coef| `0.002117`
- `lag_10__CT1__duck_amount`: coefficient `-0.002028`, |coef| `0.002028`
- `lag_08__CT3__flash_duration`: coefficient `-0.001868`, |coef| `0.001868`
- `lag_00__CT3__duck_amount`: coefficient `0.001816`, |coef| `0.001816`
- `lag_15__T_shots_fired_sum`: coefficient `-0.001777`, |coef| `0.001777`
- `lag_02__T1__flash_duration`: coefficient `0.001752`, |coef| `0.001752`
- `lag_08__CT_flash_duration_sum`: coefficient `-0.001685`, |coef| `0.001685`
- `lag_12__CT_place_HOUSE`: coefficient `-0.001656`, |coef| `0.001656`
- `lag_02__T_flashed_players`: coefficient `0.001619`, |coef| `0.001619`
- `lag_08__T_utility_damage_last_5s`: coefficient `0.001594`, |coef| `0.001594`

## Top 10 utility ridge features

- `lag_08__CT5__flash_duration`: coefficient `-0.002152` (lowers CT win probability)
- `lag_08__CT3__flash_duration`: coefficient `-0.001868` (lowers CT win probability)
- `lag_02__T1__flash_duration`: coefficient `0.001752` (raises CT win probability)
- `lag_08__CT_flash_duration_sum`: coefficient `-0.001685` (lowers CT win probability)
- `lag_08__T_utility_damage_last_5s`: coefficient `0.001594` (raises CT win probability)
- `lag_08__T_B_site_active_infernos`: coefficient `-0.001460` (lowers CT win probability)
- `lag_07__T_B_site_active_infernos`: coefficient `-0.001287` (lowers CT win probability)
- `lag_04__CT3__flash_duration`: coefficient `-0.001267` (lowers CT win probability)
- `lag_08__T_A_site_active_infernos`: coefficient `-0.001182` (lowers CT win probability)
- `lag_04__CT_flash_duration_sum`: coefficient `-0.001170` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002773` (raises CT win probability)
- `lag_09__CT_place_TOPOFMID`: coefficient `-0.002515` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002488` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002437` (raises CT win probability)
- `lag_09__CT_place_MIDDLE`: coefficient `0.002117` (raises CT win probability)
- `lag_10__CT1__duck_amount`: coefficient `-0.002028` (lowers CT win probability)
- `lag_00__CT3__duck_amount`: coefficient `0.001816` (raises CT win probability)
- `lag_15__T_shots_fired_sum`: coefficient `-0.001777` (lowers CT win probability)
- `lag_12__CT_place_HOUSE`: coefficient `-0.001656` (lowers CT win probability)
- `lag_02__T_flashed_players`: coefficient `0.001619` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `42801`, seconds `13.00`, LSTM delta `+0.3548`

Top all feature movements:
- `lag_09__CT_place_TOPOFMID`: contribution `+0.027380`
- `lag_00__CT_shots_fired_sum`: contribution `+0.016933`
- `lag_09__CT_place_MIDDLE`: contribution `+0.016660`
- `lag_00__CT_kills_last_3s`: contribution `+0.014364`
- `lag_00__kill_diff_last_3s`: contribution `+0.013351`

Top utility-only movements:
- `lag_08__T_utility_damage_last_5s`: contribution `+0.012972`
- `lag_02__T1__flash_duration`: contribution `+0.011870`
- `lag_02__T_flash_duration_sum`: contribution `+0.005326`
- `lag_06__T_utility_damage_last_5s`: contribution `+0.004824`
- `lag_03__T2__flash_duration`: contribution `+0.004346`

### tick `45489`, seconds `55.00`, LSTM delta `+0.2181`

Top all feature movements:
- `lag_08__CT5__flash_duration`: contribution `+0.013744`
- `lag_08__CT3__flash_duration`: contribution `+0.012271`
- `lag_08__CT_flash_duration_sum`: contribution `+0.009916`
- `lag_15__T_shots_fired_sum`: contribution `+0.009325`
- `lag_10__CT1__duck_amount`: contribution `+0.007738`

Top utility-only movements:
- `lag_08__CT5__flash_duration`: contribution `+0.013744`
- `lag_08__CT3__flash_duration`: contribution `+0.012271`
- `lag_08__CT_flash_duration_sum`: contribution `+0.009916`
- `lag_08__T_B_site_active_infernos`: contribution `+0.004129`
- `lag_08__T_A_site_active_infernos`: contribution `+0.003518`

### tick `44977`, seconds `47.00`, LSTM delta `-0.1894`

Top all feature movements:
- `lag_04__CT_flashed_players`: contribution `-0.008614`
- `lag_04__CT3__flash_duration`: contribution `-0.008323`
- `lag_04__CT_flash_duration_sum`: contribution `-0.007677`
- `lag_04__CT5__flash_duration`: contribution `-0.007028`
- `lag_00__kill_diff_last_3s`: contribution `-0.006675`

Top utility-only movements:
- `lag_04__CT3__flash_duration`: contribution `-0.008323`
- `lag_04__CT_flash_duration_sum`: contribution `-0.007677`
- `lag_04__CT5__flash_duration`: contribution `-0.007028`
- `lag_07__T_B_site_active_infernos`: contribution `-0.003639`
- `lag_07__T_A_site_active_infernos`: contribution `-0.002880`

### tick `45233`, seconds `51.00`, LSTM delta `-0.0684`

Top all feature movements:
- `lag_12__CT_flash_duration_sum`: contribution `-0.005843`
- `lag_12__CT3__flash_duration`: contribution `-0.005058`
- `lag_12__CT5__flash_duration`: contribution `-0.004772`
- `lag_12__CT_flashed_players`: contribution `-0.003334`
- `lag_00__T4__is_walking`: contribution `-0.003058`

Top utility-only movements:
- `lag_12__CT_flash_duration_sum`: contribution `-0.005843`
- `lag_12__CT3__flash_duration`: contribution `-0.005058`
- `lag_12__CT5__flash_duration`: contribution `-0.004772`
- `lag_00__CT3__flash_duration`: contribution `-0.002842`

### tick `42833`, seconds `13.50`, LSTM delta `+0.0485`

Top all feature movements:
- `lag_01__CT_shots_fired_sum`: contribution `+0.006996`
- `lag_13__CT_place_HOUSE`: contribution `+0.006139`
- `lag_12__CT_place_HOUSE`: contribution `+0.005849`
- `lag_10__CT_place_MIDDLE`: contribution `+0.005171`
- `lag_01__CT_kills_last_3s`: contribution `+0.004270`

Top utility-only movements:
- `lag_03__T_flash_duration_sum`: contribution `+0.002098`
