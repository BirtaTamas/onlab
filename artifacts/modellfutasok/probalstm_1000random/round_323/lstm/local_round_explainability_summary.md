# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-vitality-vs-astralis-bo3-Zley6FZuKcttfrliAqsvWJ/astralis-vs-vitality-m1-inferno.csv`
- round_num: `16`

## Largest probability jumps

- tick `116931`, seconds `34.50`, LSTM `0.5573`, delta `-0.2182`
- tick `120355`, seconds `88.00`, LSTM `0.8064`, delta `+0.1983`
- tick `116419`, seconds `26.50`, LSTM `0.7291`, delta `+0.1363`
- tick `120387`, seconds `88.50`, LSTM `0.9275`, delta `+0.1211`
- tick `115427`, seconds `11.00`, LSTM `0.5613`, delta `+0.0647`
- tick `115459`, seconds `11.50`, LSTM `0.6099`, delta `+0.0486`
- tick `117219`, seconds `39.00`, LSTM `0.6173`, delta `+0.0444`
- tick `116547`, seconds `28.50`, LSTM `0.7436`, delta `+0.0435`
- tick `117411`, seconds `42.00`, LSTM `0.6070`, delta `-0.0384`
- tick `117443`, seconds `42.50`, LSTM `0.6437`, delta `+0.0368`

## Top 15 local ridge features

- `lag_00__T4__flash_duration`: coefficient `0.004620`, |coef| `0.004620`
- `lag_00__CT_kills_last_3s`: coefficient `0.003178`, |coef| `0.003178`
- `lag_00__kill_diff_last_3s`: coefficient `0.003037`, |coef| `0.003037`
- `lag_00__CT1__flash_duration`: coefficient `0.002355`, |coef| `0.002355`
- `lag_00__T_flash_duration_sum`: coefficient `0.002328`, |coef| `0.002328`
- `lag_00__T3__flash_duration`: coefficient `0.002167`, |coef| `0.002167`
- `lag_01__T4__flash_duration`: coefficient `0.002033`, |coef| `0.002033`
- `lag_00__damage_diff_last_5s`: coefficient `0.001864`, |coef| `0.001864`
- `lag_07__CT_place_QUAD`: coefficient `0.001823`, |coef| `0.001823`
- `lag_12__CT5__is_walking`: coefficient `0.001707`, |coef| `0.001707`
- `lag_00__CT_damage_last_5s`: coefficient `0.001617`, |coef| `0.001617`
- `lag_09__T4__is_walking`: coefficient `-0.001616`, |coef| `0.001616`
- `lag_00__T3__alive`: coefficient `-0.001549`, |coef| `0.001549`
- `lag_08__T3__is_walking`: coefficient `0.001450`, |coef| `0.001450`
- `lag_00__T3__smoke`: coefficient `-0.001392`, |coef| `0.001392`

## Top 10 utility ridge features

- `lag_00__T4__flash_duration`: coefficient `0.004620` (raises CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `0.002355` (raises CT win probability)
- `lag_00__T_flash_duration_sum`: coefficient `0.002328` (raises CT win probability)
- `lag_00__T3__flash_duration`: coefficient `0.002167` (raises CT win probability)
- `lag_01__T4__flash_duration`: coefficient `0.002033` (raises CT win probability)
- `lag_00__T3__smoke`: coefficient `-0.001392` (lowers CT win probability)
- `lag_04__CT1__flash_duration`: coefficient `-0.001349` (lowers CT win probability)
- `lag_01__CT1__flash_duration`: coefficient `0.001228` (raises CT win probability)
- `lag_04__T4__smoke`: coefficient `-0.001192` (lowers CT win probability)
- `lag_12__T3__flash_duration`: coefficient `-0.001115` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.003178` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003037` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001864` (raises CT win probability)
- `lag_07__CT_place_QUAD`: coefficient `0.001823` (raises CT win probability)
- `lag_12__CT5__is_walking`: coefficient `0.001707` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001617` (raises CT win probability)
- `lag_09__T4__is_walking`: coefficient `-0.001616` (lowers CT win probability)
- `lag_00__T3__alive`: coefficient `-0.001549` (lowers CT win probability)
- `lag_08__T3__is_walking`: coefficient `0.001450` (raises CT win probability)
- `lag_03__CT_place_RUINS`: coefficient `0.001376` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `116931`, seconds `34.50`, LSTM delta `-0.2182`

Top all feature movements:
- `lag_00__T4__flash_duration`: contribution `-0.028830`
- `lag_00__T_flash_duration_sum`: contribution `-0.016636`
- `lag_07__CT_place_QUAD`: contribution `-0.014364`
- `lag_00__T3__flash_duration`: contribution `-0.009707`
- `lag_04__CT1__flash_duration`: contribution `-0.009679`

Top utility-only movements:
- `lag_00__T4__flash_duration`: contribution `-0.028830`
- `lag_00__T_flash_duration_sum`: contribution `-0.016636`
- `lag_00__T3__flash_duration`: contribution `-0.009707`
- `lag_04__CT1__flash_duration`: contribution `-0.009679`
- `lag_12__T_flash_duration_sum`: contribution `-0.007990`

### tick `120355`, seconds `88.00`, LSTM delta `+0.1983`

Top all feature movements:
- `lag_00__T4__flash_duration`: contribution `+0.035931`
- `lag_00__CT1__flash_duration`: contribution `+0.014608`
- `lag_00__CT_kills_last_3s`: contribution `+0.009175`
- `lag_00__T_flash_duration_sum`: contribution `+0.008839`
- `lag_00__kill_diff_last_3s`: contribution `+0.007311`

Top utility-only movements:
- `lag_00__T4__flash_duration`: contribution `+0.035931`
- `lag_00__CT1__flash_duration`: contribution `+0.014608`
- `lag_00__T_flash_duration_sum`: contribution `+0.008839`
- `lag_00__T3__flash_duration`: contribution `+0.006831`
- `lag_00__CT_flash_duration_sum`: contribution `+0.003062`

### tick `116419`, seconds `26.50`, LSTM delta `+0.1363`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.009175`
- `lag_00__kill_diff_last_3s`: contribution `+0.007311`
- `lag_03__CT_place_QUAD`: contribution `+0.006910`
- `lag_09__CT_utility_damage_last_5s`: contribution `+0.004409`
- `lag_04__CT_flashed_players`: contribution `+0.004386`

Top utility-only movements:
- `lag_09__CT_utility_damage_last_5s`: contribution `+0.004409`
- `lag_09__utility_damage_diff_last_5s`: contribution `+0.002711`
- `lag_12__CT_A_site_active_infernos`: contribution `+0.001851`

### tick `120387`, seconds `88.50`, LSTM delta `+0.1211`

Top all feature movements:
- `lag_01__T4__flash_duration`: contribution `+0.015811`
- `lag_00__CT_kills_last_3s`: contribution `+0.009175`
- `lag_01__CT1__flash_duration`: contribution `+0.007618`
- `lag_00__kill_diff_last_3s`: contribution `+0.007311`
- `lag_00__CT_shots_fired_sum`: contribution `+0.004395`

Top utility-only movements:
- `lag_01__T4__flash_duration`: contribution `+0.015811`
- `lag_01__CT1__flash_duration`: contribution `+0.007618`
- `lag_01__T3__flash_duration`: contribution `+0.002848`
- `lag_00__T2__smoke`: contribution `+0.002126`
- `lag_01__T_flash_duration_sum`: contribution `+0.002070`

### tick `115427`, seconds `11.00`, LSTM delta `+0.0647`

Top all feature movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.009045`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.007669`
- `lag_00__CT_flash_duration_sum`: contribution `+0.007531`
- `lag_00__damage_diff_last_5s`: contribution `+0.004921`
- `lag_03__CT_place_RUINS`: contribution `+0.004806`

Top utility-only movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.009045`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.007669`
- `lag_00__CT_flash_duration_sum`: contribution `+0.007531`
- `lag_00__T3__flash_duration`: contribution `+0.004387`
- `lag_00__T_flash_duration_sum`: contribution `+0.001964`
