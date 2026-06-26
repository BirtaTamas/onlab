# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-spirit-vs-the-huns-bo3-TWIJIxJZifB3vPv3OUvjVr/spirit-vs-the-huns-m2-dust2.csv`
- round_num: `11`

## Largest probability jumps

- tick `90846`, seconds `26.50`, LSTM `0.3552`, delta `-0.1669`
- tick `90494`, seconds `21.00`, LSTM `0.4879`, delta `+0.1402`
- tick `90974`, seconds `28.50`, LSTM `0.0549`, delta `-0.1166`
- tick `90942`, seconds `28.00`, LSTM `0.1715`, delta `-0.0770`
- tick `90910`, seconds `27.50`, LSTM `0.2485`, delta `-0.0608`
- tick `91454`, seconds `36.00`, LSTM `0.0174`, delta `-0.0464`
- tick `90878`, seconds `27.00`, LSTM `0.3093`, delta `-0.0460`
- tick `90334`, seconds `18.50`, LSTM `0.3944`, delta `+0.0392`
- tick `89406`, seconds `4.00`, LSTM `0.4578`, delta `+0.0342`
- tick `89726`, seconds `9.00`, LSTM `0.3873`, delta `-0.0332`

## Top 15 local ridge features

- `lag_05__T4__flash_duration`: coefficient `0.001488`, |coef| `0.001488`
- `lag_02__T_place_BDOORS`: coefficient `-0.001227`, |coef| `0.001227`
- `lag_02__T4__flash_duration`: coefficient `0.001204`, |coef| `0.001204`
- `lag_10__CT_place_EXTENDEDA`: coefficient `0.001167`, |coef| `0.001167`
- `lag_05__T_flash_duration_sum`: coefficient `0.001128`, |coef| `0.001128`
- `lag_15__CT5__flash_duration`: coefficient `-0.001126`, |coef| `0.001126`
- `lag_00__T_kills_last_3s`: coefficient `-0.001109`, |coef| `0.001109`
- `lag_04__CT5__flash_duration`: coefficient `0.001105`, |coef| `0.001105`
- `lag_00__kill_diff_last_3s`: coefficient `0.001105`, |coef| `0.001105`
- `lag_00__T_place_BDOORS`: coefficient `-0.001100`, |coef| `0.001100`
- `lag_01__damage_diff_last_5s`: coefficient `0.001015`, |coef| `0.001015`
- `lag_04__CT_flashed_players`: coefficient `0.001014`, |coef| `0.001014`
- `lag_13__CT2__flash_duration`: coefficient `-0.001011`, |coef| `0.001011`
- `lag_05__T_flashed_players`: coefficient `0.001005`, |coef| `0.001005`
- `lag_01__T_place_BDOORS`: coefficient `-0.000977`, |coef| `0.000977`

## Top 10 utility ridge features

- `lag_05__T4__flash_duration`: coefficient `0.001488` (raises CT win probability)
- `lag_02__T4__flash_duration`: coefficient `0.001204` (raises CT win probability)
- `lag_05__T_flash_duration_sum`: coefficient `0.001128` (raises CT win probability)
- `lag_15__CT5__flash_duration`: coefficient `-0.001126` (lowers CT win probability)
- `lag_04__CT5__flash_duration`: coefficient `0.001105` (raises CT win probability)
- `lag_13__CT2__flash_duration`: coefficient `-0.001011` (lowers CT win probability)
- `lag_05__T3__flash_duration`: coefficient `0.000974` (raises CT win probability)
- `lag_06__T4__flash_duration`: coefficient `0.000969` (raises CT win probability)
- `lag_02__CT2__flash_duration`: coefficient `0.000934` (raises CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `0.000932` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_02__T_place_BDOORS`: coefficient `-0.001227` (lowers CT win probability)
- `lag_10__CT_place_EXTENDEDA`: coefficient `0.001167` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001109` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001105` (raises CT win probability)
- `lag_00__T_place_BDOORS`: coefficient `-0.001100` (lowers CT win probability)
- `lag_01__damage_diff_last_5s`: coefficient `0.001015` (raises CT win probability)
- `lag_04__CT_flashed_players`: coefficient `0.001014` (raises CT win probability)
- `lag_05__T_flashed_players`: coefficient `0.001005` (raises CT win probability)
- `lag_01__T_place_BDOORS`: coefficient `-0.000977` (lowers CT win probability)
- `lag_05__CT_place_LONGDOORS`: coefficient `0.000968` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `90846`, seconds `26.50`, LSTM delta `-0.1669`

Top all feature movements:
- `lag_02__T4__flash_duration`: contribution `-0.008714`
- `lag_15__CT5__flash_duration`: contribution `-0.007054`
- `lag_10__CT_place_EXTENDEDA`: contribution `-0.006550`
- `lag_13__CT2__flash_duration`: contribution `-0.005035`
- `lag_00__CT2__flash_duration`: contribution `-0.004880`

Top utility-only movements:
- `lag_02__T4__flash_duration`: contribution `-0.008714`
- `lag_15__CT5__flash_duration`: contribution `-0.007054`
- `lag_13__CT2__flash_duration`: contribution `-0.005035`
- `lag_00__CT2__flash_duration`: contribution `-0.004880`
- `lag_03__T3__flash_duration`: contribution `-0.003028`

### tick `90494`, seconds `21.00`, LSTM delta `+0.1402`

Top all feature movements:
- `lag_05__T4__flash_duration`: contribution `+0.011277`
- `lag_05__T_flash_duration_sum`: contribution `+0.008324`
- `lag_05__T_flashed_players`: contribution `+0.007753`
- `lag_05__T3__flash_duration`: contribution `+0.006933`
- `lag_04__CT5__flash_duration`: contribution `+0.006923`

Top utility-only movements:
- `lag_05__T4__flash_duration`: contribution `+0.011277`
- `lag_05__T_flash_duration_sum`: contribution `+0.008324`
- `lag_05__T3__flash_duration`: contribution `+0.006933`
- `lag_04__CT5__flash_duration`: contribution `+0.006923`
- `lag_02__CT2__flash_duration`: contribution `+0.004652`

### tick `90974`, seconds `28.50`, LSTM delta `-0.1166`

Top all feature movements:
- `lag_02__T_place_BDOORS`: contribution `-0.015344`
- `lag_06__T4__flash_duration`: contribution `-0.007010`
- `lag_02__CT_shots_fired_sum`: contribution `-0.006131`
- `lag_00__T_kills_last_3s`: contribution `-0.003515`
- `lag_03__CT2__duck_amount`: contribution `-0.003413`

Top utility-only movements:
- `lag_06__T4__flash_duration`: contribution `-0.007010`
- `lag_04__CT2__flash_duration`: contribution `-0.002989`
- `lag_07__T3__flash_duration`: contribution `-0.002444`
- `lag_06__T_flash_duration_sum`: contribution `-0.001956`
- `lag_04__CT_flash_duration_sum`: contribution `-0.001901`

### tick `90942`, seconds `28.00`, LSTM delta `-0.0770`

Top all feature movements:
- `lag_01__T_place_BDOORS`: contribution `-0.012226`
- `lag_05__T4__flash_duration`: contribution `-0.010766`
- `lag_03__CT2__flash_duration`: contribution `-0.003452`
- `lag_05__T_flash_duration_sum`: contribution `-0.003320`
- `lag_02__CT_shots_fired_sum`: contribution `-0.003065`

Top utility-only movements:
- `lag_05__T4__flash_duration`: contribution `-0.010766`
- `lag_03__CT2__flash_duration`: contribution `-0.003452`
- `lag_05__T_flash_duration_sum`: contribution `-0.003320`
- `lag_06__T3__flash_duration`: contribution `-0.002717`
- `lag_13__CT5__flash_duration`: contribution `-0.001293`

### tick `90910`, seconds `27.50`, LSTM delta `-0.0608`

Top all feature movements:
- `lag_00__T_place_BDOORS`: contribution `-0.013761`
- `lag_02__CT2__flash_duration`: contribution `-0.004894`
- `lag_05__T3__flash_duration`: contribution `-0.004533`
- `lag_12__CT_place_EXTENDEDA`: contribution `-0.003484`
- `lag_04__T4__flash_duration`: contribution `-0.002900`

Top utility-only movements:
- `lag_02__CT2__flash_duration`: contribution `-0.004894`
- `lag_05__T3__flash_duration`: contribution `-0.004533`
- `lag_04__T4__flash_duration`: contribution `-0.002900`
- `lag_15__CT5__flash_duration`: contribution `+0.002357`
- `lag_05__T_flash_duration_sum`: contribution `-0.002189`
