# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-vitality-vs-the-mongolz-bo3-vhm_UWcBfYfYcOeLh9JIDA/vitality-vs-the-mongolz-m1-mirage.csv`
- round_num: `1`

## Largest probability jumps

- tick `19759`, seconds `39.50`, LSTM `0.4607`, delta `-0.2251`
- tick `22287`, seconds `79.00`, LSTM `0.7250`, delta `+0.2149`
- tick `18991`, seconds `27.50`, LSTM `0.8217`, delta `+0.1485`
- tick `19727`, seconds `39.00`, LSTM `0.6858`, delta `+0.1473`
- tick `19663`, seconds `38.00`, LSTM `0.5559`, delta `-0.1317`
- tick `18863`, seconds `25.50`, LSTM `0.6342`, delta `+0.1272`
- tick `22607`, seconds `84.00`, LSTM `0.8806`, delta `+0.1251`
- tick `19087`, seconds `29.00`, LSTM `0.7504`, delta `-0.1132`
- tick `20239`, seconds `47.00`, LSTM `0.6178`, delta `+0.1109`
- tick `19631`, seconds `37.50`, LSTM `0.6877`, delta `-0.0989`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003486`, |coef| `0.003486`
- `lag_00__CT_defusing_count`: coefficient `0.003399`, |coef| `0.003399`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.002713`, |coef| `0.002713`
- `lag_00__damage_diff_last_5s`: coefficient `0.002643`, |coef| `0.002643`
- `lag_12__CT_place_SHOP`: coefficient `0.002530`, |coef| `0.002530`
- `lag_00__CT_kills_last_3s`: coefficient `0.002491`, |coef| `0.002491`
- `lag_10__T_flash_alpha_mean`: coefficient `-0.002284`, |coef| `0.002284`
- `lag_00__CT_velocity_mean`: coefficient `-0.002249`, |coef| `0.002249`
- `lag_13__T_bomb_zone_count`: coefficient `-0.002135`, |coef| `0.002135`
- `lag_15__T3__flash_duration`: coefficient `0.002109`, |coef| `0.002109`
- `lag_00__CT_place_JUNGLE`: coefficient `0.002031`, |coef| `0.002031`
- `lag_06__T_bomb_zone_count`: coefficient `-0.001905`, |coef| `0.001905`
- `lag_00__T_kills_last_3s`: coefficient `-0.001856`, |coef| `0.001856`
- `lag_13__T_velocity_mean`: coefficient `0.001827`, |coef| `0.001827`
- `lag_08__T_place_TRAMP`: coefficient `0.001638`, |coef| `0.001638`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.002713` (lowers CT win probability)
- `lag_10__T_flash_alpha_mean`: coefficient `-0.002284` (lowers CT win probability)
- `lag_15__T3__flash_duration`: coefficient `0.002109` (raises CT win probability)
- `lag_14__T3__flash_duration`: coefficient `0.001601` (raises CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `-0.001558` (lowers CT win probability)
- `lag_07__T4__flash_duration`: coefficient `0.001168` (raises CT win probability)
- `lag_07__T_flash_duration_sum`: coefficient `0.001117` (raises CT win probability)
- `lag_15__T_flash_duration_sum`: coefficient `0.000962` (raises CT win probability)
- `lag_11__T_flash_alpha_mean`: coefficient `-0.000932` (lowers CT win probability)
- `lag_03__T2__flash_duration`: coefficient `0.000918` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003486` (raises CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.003399` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002643` (raises CT win probability)
- `lag_12__CT_place_SHOP`: coefficient `0.002530` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002491` (raises CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.002249` (lowers CT win probability)
- `lag_13__T_bomb_zone_count`: coefficient `-0.002135` (lowers CT win probability)
- `lag_00__CT_place_JUNGLE`: coefficient `0.002031` (raises CT win probability)
- `lag_06__T_bomb_zone_count`: coefficient `-0.001905` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001856` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `19759`, seconds `39.50`, LSTM delta `-0.2251`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.008392`
- `lag_04__CT_place_JUNGLE`: contribution `-0.006647`
- `lag_00__T_kills_last_3s`: contribution `-0.005879`
- `lag_00__damage_diff_last_5s`: contribution `-0.005307`
- `lag_12__T5__duck_amount`: contribution `-0.004769`

Top utility-only movements:
- `lag_01__T_flash_alpha_mean`: contribution `-0.003152`

### tick `22287`, seconds `79.00`, LSTM delta `+0.2149`

Top all feature movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.016458`
- `lag_12__CT_place_SHOP`: contribution `+0.012692`
- `lag_13__T_bomb_zone_count`: contribution `+0.012429`
- `lag_00__kill_diff_last_3s`: contribution `+0.008392`
- `lag_13__T_duck_amount_mean`: contribution `+0.007960`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.016458`

### tick `18991`, seconds `27.50`, LSTM delta `+0.1485`

Top all feature movements:
- `lag_07__T_flash_duration_sum`: contribution `+0.010283`
- `lag_07__T4__flash_duration`: contribution `+0.008896`
- `lag_00__kill_diff_last_3s`: contribution `+0.008392`
- `lag_00__CT_kills_last_3s`: contribution `+0.007191`
- `lag_07__T_flashed_players`: contribution `+0.006758`

Top utility-only movements:
- `lag_07__T_flash_duration_sum`: contribution `+0.010283`
- `lag_07__T4__flash_duration`: contribution `+0.008896`
- `lag_07__T3__flash_duration`: contribution `+0.005150`
- `lag_04__T4__flash_duration`: contribution `+0.005099`
- `lag_07__T2__flash_duration`: contribution `+0.004816`

### tick `19727`, seconds `39.00`, LSTM delta `+0.1473`

Top all feature movements:
- `lag_00__CT_place_JUNGLE`: contribution `+0.013032`
- `lag_00__kill_diff_last_3s`: contribution `+0.008392`
- `lag_00__CT_kills_last_3s`: contribution `+0.007191`
- `lag_00__T_flash_alpha_mean`: contribution `-0.005486`
- `lag_12__T5__duck_amount`: contribution `+0.004769`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `-0.005486`

### tick `19663`, seconds `38.00`, LSTM delta `-0.1317`

Top all feature movements:
- `lag_15__T3__flash_duration`: contribution `-0.015125`
- `lag_12__CT_place_SHOP`: contribution `-0.012692`
- `lag_00__kill_diff_last_3s`: contribution `-0.008392`
- `lag_00__T_kills_last_3s`: contribution `-0.005879`
- `lag_08__T_place_TRAMP`: contribution `-0.004795`

Top utility-only movements:
- `lag_15__T3__flash_duration`: contribution `-0.015125`
- `lag_15__T_flash_duration_sum`: contribution `-0.002873`
