# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-heroic-vs-natus-vincere-bo3-P_vZ7pAIyzYcLTUjDHhSUR/heroic-vs-natus-vincere-m2-ancient.csv`
- round_num: `2`

## Largest probability jumps

- tick `6757`, seconds `10.00`, LSTM `0.0574`, delta `-0.1472`
- tick `6149`, seconds `0.50`, LSTM `0.2639`, delta `-0.0805`
- tick `6309`, seconds `3.00`, LSTM `0.2198`, delta `-0.0385`
- tick `6565`, seconds `7.00`, LSTM `0.1664`, delta `+0.0339`
- tick `6597`, seconds `7.50`, LSTM `0.1982`, delta `+0.0317`
- tick `6693`, seconds `9.00`, LSTM `0.1759`, delta `-0.0301`
- tick `6725`, seconds `9.50`, LSTM `0.2046`, delta `+0.0287`
- tick `6341`, seconds `3.50`, LSTM `0.1951`, delta `-0.0247`
- tick `6373`, seconds `4.00`, LSTM `0.1738`, delta `-0.0213`
- tick `6213`, seconds `1.50`, LSTM `0.2486`, delta `-0.0172`

## Top 15 local ridge features

- `lag_00__CT_place_UNKNOWN`: coefficient `0.002217`, |coef| `0.002217`
- `lag_14__T_place_TUNNEL`: coefficient `-0.001575`, |coef| `0.001575`
- `lag_03__CT4__flash_duration`: coefficient `-0.001213`, |coef| `0.001213`
- `lag_01__CT_place_UNKNOWN`: coefficient `-0.001089`, |coef| `0.001089`
- `lag_13__CT_place_HOUSE`: coefficient `-0.001038`, |coef| `0.001038`
- `lag_07__T_place_WATER`: coefficient `0.000793`, |coef| `0.000793`
- `lag_02__CT_place_UNKNOWN`: coefficient `0.000735`, |coef| `0.000735`
- `lag_04__CT_place_UNKNOWN`: coefficient `0.000723`, |coef| `0.000723`
- `lag_01__CT4__flash_duration`: coefficient `-0.000719`, |coef| `0.000719`
- `lag_09__T_place_WATER`: coefficient `-0.000633`, |coef| `0.000633`
- `lag_07__CT_place_HOUSE`: coefficient `0.000606`, |coef| `0.000606`
- `lag_00__T5__has_bomb`: coefficient `0.000584`, |coef| `0.000584`
- `lag_09__T_place_TUNNEL`: coefficient `0.000581`, |coef| `0.000581`
- `lag_07__CT_place_ALLEY`: coefficient `-0.000574`, |coef| `0.000574`
- `lag_00__CT4__alive`: coefficient `0.000551`, |coef| `0.000551`

## Top 10 utility ridge features

- `lag_03__CT4__flash_duration`: coefficient `-0.001213` (lowers CT win probability)
- `lag_01__CT4__flash_duration`: coefficient `-0.000719` (lowers CT win probability)
- `lag_00__CT4__smoke`: coefficient `0.000491` (raises CT win probability)
- `lag_04__CT4__flash_duration`: coefficient `-0.000474` (lowers CT win probability)
- `lag_06__CT4__flash_duration`: coefficient `-0.000462` (lowers CT win probability)
- `lag_12__CT4__flash_duration`: coefficient `-0.000443` (lowers CT win probability)
- `lag_05__CT4__flash_duration`: coefficient `-0.000441` (lowers CT win probability)
- `lag_10__CT4__flash_duration`: coefficient `-0.000415` (lowers CT win probability)
- `lag_01__CT_flash_alpha_mean`: coefficient `0.000401` (raises CT win probability)
- `lag_11__CT4__flash_duration`: coefficient `-0.000382` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_UNKNOWN`: coefficient `0.002217` (raises CT win probability)
- `lag_14__T_place_TUNNEL`: coefficient `-0.001575` (lowers CT win probability)
- `lag_01__CT_place_UNKNOWN`: coefficient `-0.001089` (lowers CT win probability)
- `lag_13__CT_place_HOUSE`: coefficient `-0.001038` (lowers CT win probability)
- `lag_07__T_place_WATER`: coefficient `0.000793` (raises CT win probability)
- `lag_02__CT_place_UNKNOWN`: coefficient `0.000735` (raises CT win probability)
- `lag_04__CT_place_UNKNOWN`: coefficient `0.000723` (raises CT win probability)
- `lag_09__T_place_WATER`: coefficient `-0.000633` (lowers CT win probability)
- `lag_07__CT_place_HOUSE`: coefficient `0.000606` (raises CT win probability)
- `lag_00__T5__has_bomb`: coefficient `0.000584` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `6757`, seconds `10.00`, LSTM delta `-0.1472`

Top all feature movements:
- `lag_14__T_place_TUNNEL`: contribution `-0.019136`
- `lag_13__CT_place_HOUSE`: contribution `-0.007335`
- `lag_03__CT4__flash_duration`: contribution `-0.007167`
- `lag_07__T_place_WATER`: contribution `-0.004526`
- `lag_09__T_place_WATER`: contribution `-0.003614`

Top utility-only movements:
- `lag_03__CT4__flash_duration`: contribution `-0.007167`
- `lag_00__CT4__flash_duration`: contribution `-0.001483`

### tick `6149`, seconds `0.50`, LSTM delta `-0.0805`

Top all feature movements:
- `lag_01__CT_place_UNKNOWN`: contribution `-0.038220`
- `lag_00__T_velocity_mean`: contribution `-0.001192`
- `lag_01__T_closest_enemy_dist`: contribution `-0.000872`
- `lag_01__T_place_TSPAWN`: contribution `-0.000792`
- `lag_00__CT_velocity_mean`: contribution `-0.000761`

Top utility-only movements:
- `lag_00__CT1__smoke`: contribution `-0.000675`
- `lag_01__CT_flash_alpha_mean`: contribution `-0.000523`
- `lag_01__CT_smoke_inv`: contribution `-0.000364`

### tick `6309`, seconds `3.00`, LSTM delta `-0.0385`

Top all feature movements:
- `lag_00__CT_place_UNKNOWN`: contribution `-0.015570`
- `lag_04__CT_place_UNKNOWN`: contribution `-0.015231`
- `lag_02__CT_place_UNKNOWN`: contribution `-0.005159`
- `lag_06__CT_place_UNKNOWN`: contribution `-0.003556`
- `lag_00__T_place_TUNNEL`: contribution `+0.001962`

Top utility-only movements:
- `lag_06__T5__flash`: contribution `+0.000496`
- `lag_06__CT_flash_alpha_mean`: contribution `-0.000424`
- `lag_01__CT5__smoke`: contribution `+0.000234`
- `lag_06__CT3__smoke`: contribution `-0.000211`
- `lag_04__T5__smoke`: contribution `+0.000197`

### tick `6565`, seconds `7.00`, LSTM delta `+0.0339`

Top all feature movements:
- `lag_14__CT_place_UNKNOWN`: contribution `+0.017145`
- `lag_12__CT_place_UNKNOWN`: contribution `+0.004957`
- `lag_07__CT_place_HOUSE`: contribution `+0.004283`
- `lag_08__T_place_TUNNEL`: contribution `+0.004281`
- `lag_08__CT_place_UNKNOWN`: contribution `+0.001873`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `6597`, seconds `7.50`, LSTM delta `+0.0317`

Top all feature movements:
- `lag_13__CT_place_UNKNOWN`: contribution `+0.008710`
- `lag_09__T_place_TUNNEL`: contribution `+0.007062`
- `lag_15__CT_place_UNKNOWN`: contribution `+0.006579`
- `lag_04__T_place_WATER`: contribution `+0.002665`
- `lag_07__CT_place_HOUSE`: contribution `+0.002142`

Top utility-only movements:
- `lag_15__CT5__smoke`: contribution `+0.000484`
