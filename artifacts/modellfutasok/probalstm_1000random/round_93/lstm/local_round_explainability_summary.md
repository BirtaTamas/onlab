# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-the-mongolz-vs-g2-bo3-_aqP5h00uQDg161T2kCLGM/the-mongolz-vs-g2-m2-dust2.csv`
- round_num: `10`

## Largest probability jumps

- tick `86400`, seconds `88.00`, LSTM `0.8480`, delta `+0.1658`
- tick `86432`, seconds `88.50`, LSTM `0.9166`, delta `+0.0686`
- tick `84992`, seconds `66.00`, LSTM `0.6641`, delta `+0.0491`
- tick `86240`, seconds `85.50`, LSTM `0.7053`, delta `+0.0452`
- tick `84480`, seconds `58.00`, LSTM `0.6742`, delta `+0.0413`
- tick `82752`, seconds `31.00`, LSTM `0.6053`, delta `-0.0371`
- tick `81792`, seconds `16.00`, LSTM `0.6276`, delta `+0.0366`
- tick `86496`, seconds `89.50`, LSTM `0.9611`, delta `+0.0361`
- tick `86080`, seconds `83.00`, LSTM `0.6590`, delta `+0.0354`
- tick `82784`, seconds `31.50`, LSTM `0.5717`, delta `-0.0336`

## Top 15 local ridge features

- `lag_01__T_place_BDOORS`: coefficient `0.001309`, |coef| `0.001309`
- `lag_00__T_place_MIDDOORS`: coefficient `-0.001217`, |coef| `0.001217`
- `lag_11__CT5__flash_duration`: coefficient `0.001024`, |coef| `0.001024`
- `lag_11__T1__flash_duration`: coefficient `0.000895`, |coef| `0.000895`
- `lag_00__T2__duck_amount`: coefficient `-0.000869`, |coef| `0.000869`
- `lag_11__T_flash_duration_sum`: coefficient `0.000856`, |coef| `0.000856`
- `lag_11__T_flashed_players`: coefficient `0.000848`, |coef| `0.000848`
- `lag_09__T_place_MIDDOORS`: coefficient `0.000805`, |coef| `0.000805`
- `lag_01__T_place_MIDDOORS`: coefficient `-0.000784`, |coef| `0.000784`
- `lag_03__CT3__is_walking`: coefficient `-0.000724`, |coef| `0.000724`
- `lag_11__T2__flash_duration`: coefficient `0.000719`, |coef| `0.000719`
- `lag_00__CT_kills_last_3s`: coefficient `0.000702`, |coef| `0.000702`
- `lag_00__CT3__is_walking`: coefficient `-0.000702`, |coef| `0.000702`
- `lag_00__CT4__is_scoped`: coefficient `-0.000699`, |coef| `0.000699`
- `lag_00__CT_damage_last_5s`: coefficient `0.000680`, |coef| `0.000680`

## Top 10 utility ridge features

- `lag_11__CT5__flash_duration`: coefficient `0.001024` (raises CT win probability)
- `lag_11__T1__flash_duration`: coefficient `0.000895` (raises CT win probability)
- `lag_11__T_flash_duration_sum`: coefficient `0.000856` (raises CT win probability)
- `lag_11__T2__flash_duration`: coefficient `0.000719` (raises CT win probability)
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.000580` (raises CT win probability)
- `lag_12__CT5__flash_duration`: coefficient `0.000551` (raises CT win probability)
- `lag_12__T1__flash_duration`: coefficient `0.000503` (raises CT win probability)
- `lag_02__T4__flash_duration`: coefficient `0.000498` (raises CT win probability)
- `lag_02__CT_A_site_active_infernos`: coefficient `0.000494` (raises CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000475` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_01__T_place_BDOORS`: coefficient `0.001309` (raises CT win probability)
- `lag_00__T_place_MIDDOORS`: coefficient `-0.001217` (lowers CT win probability)
- `lag_00__T2__duck_amount`: coefficient `-0.000869` (lowers CT win probability)
- `lag_11__T_flashed_players`: coefficient `0.000848` (raises CT win probability)
- `lag_09__T_place_MIDDOORS`: coefficient `0.000805` (raises CT win probability)
- `lag_01__T_place_MIDDOORS`: coefficient `-0.000784` (lowers CT win probability)
- `lag_03__CT3__is_walking`: coefficient `-0.000724` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000702` (raises CT win probability)
- `lag_00__CT3__is_walking`: coefficient `-0.000702` (lowers CT win probability)
- `lag_00__CT4__is_scoped`: coefficient `-0.000699` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `86400`, seconds `88.00`, LSTM delta `+0.1658`

Top all feature movements:
- `lag_01__T_place_BDOORS`: contribution `+0.016369`
- `lag_11__CT5__flash_duration`: contribution `+0.007498`
- `lag_11__T_flashed_players`: contribution `+0.006548`
- `lag_11__T1__flash_duration`: contribution `+0.006127`
- `lag_11__T_flash_duration_sum`: contribution `+0.005823`

Top utility-only movements:
- `lag_11__CT5__flash_duration`: contribution `+0.007498`
- `lag_11__T1__flash_duration`: contribution `+0.006127`
- `lag_11__T_flash_duration_sum`: contribution `+0.005823`
- `lag_11__T2__flash_duration`: contribution `+0.003901`
- `lag_02__CT_A_site_active_infernos`: contribution `+0.001744`

### tick `86432`, seconds `88.50`, LSTM delta `+0.0686`

Top all feature movements:
- `lag_02__T_place_BDOORS`: contribution `+0.007520`
- `lag_00__T_place_MIDDOORS`: contribution `+0.005171`
- `lag_12__CT5__flash_duration`: contribution `+0.004032`
- `lag_12__T_flashed_players`: contribution `+0.003702`
- `lag_01__CT_place_ARAMP`: contribution `-0.003658`

Top utility-only movements:
- `lag_12__CT5__flash_duration`: contribution `+0.004032`
- `lag_12__T1__flash_duration`: contribution `+0.003443`
- `lag_12__T_flash_duration_sum`: contribution `+0.003200`
- `lag_12__T2__flash_duration`: contribution `+0.002279`
- `lag_00__T1__flash_duration`: contribution `+0.001961`

### tick `84992`, seconds `66.00`, LSTM delta `+0.0491`

Top all feature movements:
- `lag_00__T_place_TUNNELSTAIRS`: contribution `+0.003437`
- `lag_09__CT_place_ARAMP`: contribution `+0.003193`
- `lag_13__T_place_LOWERTUNNEL`: contribution `+0.002500`
- `lag_14__CT_place_EXTENDEDA`: contribution `+0.002311`
- `lag_00__T5__duck_amount`: contribution `+0.002171`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `86240`, seconds `85.50`, LSTM delta `+0.0452`

Top all feature movements:
- `lag_09__T_place_MIDDOORS`: contribution `+0.003423`
- `lag_06__CT5__flash_duration`: contribution `+0.003170`
- `lag_06__T1__flash_duration`: contribution `+0.002743`
- `lag_06__T_flashed_players`: contribution `+0.001750`
- `lag_03__CT3__is_walking`: contribution `+0.001729`

Top utility-only movements:
- `lag_06__CT5__flash_duration`: contribution `+0.003170`
- `lag_06__T1__flash_duration`: contribution `+0.002743`
- `lag_10__CT_utility_damage_last_5s`: contribution `+0.001664`
- `lag_06__T_flash_duration_sum`: contribution `+0.001288`
- `lag_10__utility_damage_diff_last_5s`: contribution `+0.001121`

### tick `84480`, seconds `58.00`, LSTM delta `+0.0413`

Top all feature movements:
- `lag_14__T_place_CATWALK`: contribution `+0.002623`
- `lag_00__CT4__is_scoped`: contribution `+0.002381`
- `lag_15__T1__is_scoped`: contribution `+0.002088`
- `lag_01__CT1__duck_amount`: contribution `+0.001979`
- `lag_02__T1__duck_amount`: contribution `-0.001868`

Top utility-only movements:
- `lag_06__CT_utility_damage_last_5s`: contribution `+0.000890`
