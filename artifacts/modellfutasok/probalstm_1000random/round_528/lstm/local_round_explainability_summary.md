# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-3dmax-vs-faze-bo3-oPmZd_fSjJ93cG46OkNLxM/3dmax-vs-faze-m3-dust2.csv`
- round_num: `8`

## Largest probability jumps

- tick `51134`, seconds `76.50`, LSTM `0.4938`, delta `-0.3355`
- tick `51518`, seconds `82.50`, LSTM `0.1810`, delta `-0.3180`
- tick `53278`, seconds `110.00`, LSTM `0.6762`, delta `+0.2189`
- tick `53118`, seconds `107.50`, LSTM `0.3969`, delta `+0.2132`
- tick `51006`, seconds `74.50`, LSTM `0.7557`, delta `+0.1980`
- tick `52926`, seconds `104.50`, LSTM `0.1957`, delta `+0.1320`
- tick `52862`, seconds `103.50`, LSTM `0.0768`, delta `-0.1280`
- tick `51486`, seconds `82.00`, LSTM `0.4990`, delta `+0.1251`
- tick `51646`, seconds `84.50`, LSTM `0.0273`, delta `-0.1051`
- tick `51742`, seconds `86.00`, LSTM `0.1246`, delta `+0.1018`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004603`, |coef| `0.004603`
- `lag_00__T_kills_last_3s`: coefficient `-0.004041`, |coef| `0.004041`
- `lag_00__CT_defusing_count`: coefficient `0.003833`, |coef| `0.003833`
- `lag_04__T_place_BDOORS`: coefficient `0.003103`, |coef| `0.003103`
- `lag_12__CT_place_HOLE`: coefficient `0.003061`, |coef| `0.003061`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.002943`, |coef| `0.002943`
- `lag_00__damage_diff_last_5s`: coefficient `0.002913`, |coef| `0.002913`
- `lag_13__CT4__shots_fired`: coefficient `0.002873`, |coef| `0.002873`
- `lag_01__T3__duck_amount`: coefficient `0.002529`, |coef| `0.002529`
- `lag_05__T_flash_alpha_mean`: coefficient `-0.002406`, |coef| `0.002406`
- `lag_13__CT_shots_fired_sum`: coefficient `0.002379`, |coef| `0.002379`
- `lag_05__T_duck_amount_mean`: coefficient `-0.002357`, |coef| `0.002357`
- `lag_00__T_damage_last_5s`: coefficient `-0.002301`, |coef| `0.002301`
- `lag_06__T_place_BDOORS`: coefficient `-0.002275`, |coef| `0.002275`
- `lag_15__CT2__is_walking`: coefficient `0.002271`, |coef| `0.002271`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.002943` (lowers CT win probability)
- `lag_05__T_flash_alpha_mean`: coefficient `-0.002406` (lowers CT win probability)
- `lag_01__T4__flash_duration`: coefficient `-0.002221` (lowers CT win probability)
- `lag_03__T_flash_alpha_mean`: coefficient `-0.001333` (lowers CT win probability)
- `lag_02__T_flash_alpha_mean`: coefficient `-0.001317` (lowers CT win probability)
- `lag_02__T4__flash_duration`: coefficient `-0.001236` (lowers CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `-0.001183` (lowers CT win probability)
- `lag_00__CT1__molly`: coefficient `0.001168` (raises CT win probability)
- `lag_04__T_B_site_active_infernos`: coefficient `-0.001095` (lowers CT win probability)
- `lag_07__T1__molly`: coefficient `0.001017` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004603` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.004041` (lowers CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.003833` (raises CT win probability)
- `lag_04__T_place_BDOORS`: coefficient `0.003103` (raises CT win probability)
- `lag_12__CT_place_HOLE`: coefficient `0.003061` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002913` (raises CT win probability)
- `lag_13__CT4__shots_fired`: coefficient `0.002873` (raises CT win probability)
- `lag_01__T3__duck_amount`: coefficient `0.002529` (raises CT win probability)
- `lag_13__CT_shots_fired_sum`: coefficient `0.002379` (raises CT win probability)
- `lag_05__T_duck_amount_mean`: coefficient `-0.002357` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `51134`, seconds `76.50`, LSTM delta `-0.3355`

Top all feature movements:
- `lag_04__T_place_BDOORS`: contribution `-0.038811`
- `lag_13__CT_shots_fired_sum`: contribution `-0.033054`
- `lag_13__CT4__shots_fired`: contribution `-0.030959`
- `lag_06__T_place_BDOORS`: contribution `-0.028459`
- `lag_00__kill_diff_last_3s`: contribution `-0.022158`

Top utility-only movements:
- `lag_06__T_active_infernos`: contribution `-0.003752`

### tick `51518`, seconds `82.50`, LSTM delta `-0.3180`

Top all feature movements:
- `lag_12__CT_place_HOLE`: contribution `-0.034178`
- `lag_00__T_kills_last_3s`: contribution `-0.012802`
- `lag_01__T4__flash_duration`: contribution `-0.011211`
- `lag_00__kill_diff_last_3s`: contribution `-0.011079`
- `lag_01__CT1__is_scoped`: contribution `-0.008462`

Top utility-only movements:
- `lag_01__T4__flash_duration`: contribution `-0.011211`

### tick `53278`, seconds `110.00`, LSTM delta `+0.2189`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `+0.037161`
- `lag_05__T_flash_alpha_mean`: contribution `+0.014599`
- `lag_05__T_duck_amount_mean`: contribution `+0.013706`
- `lag_03__CT_place_BDOORS`: contribution `+0.007928`
- `lag_11__CT_shots_fired_sum`: contribution `+0.007868`

Top utility-only movements:
- `lag_05__T_flash_alpha_mean`: contribution `+0.014599`

### tick `53118`, seconds `107.50`, LSTM delta `+0.2132`

Top all feature movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.017854`
- `lag_05__T_duck_amount_mean`: contribution `+0.011430`
- `lag_08__CT_place_BDOORS`: contribution `+0.010574`
- `lag_04__T3__shots_fired`: contribution `+0.009451`
- `lag_04__T_shots_fired_sum`: contribution `+0.009384`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.017854`

### tick `51006`, seconds `74.50`, LSTM delta `+0.1980`

Top all feature movements:
- `lag_02__T_place_BDOORS`: contribution `+0.014908`
- `lag_04__CT_place_HOLE`: contribution `+0.014422`
- `lag_00__T_place_BDOORS`: contribution `+0.011482`
- `lag_00__kill_diff_last_3s`: contribution `+0.011079`
- `lag_13__T5__shots_fired`: contribution `+0.010219`

Top utility-only movements:
- No utility movement among the top local contributors.
