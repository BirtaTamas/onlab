# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-legacy-bo3-ryWGopRV1OfbL288nR6Rql/falcons-vs-legacy-m1-inferno.csv`
- round_num: `10`

## Largest probability jumps

- tick `72796`, seconds `59.50`, LSTM `0.6643`, delta `-0.2521`
- tick `72412`, seconds `53.50`, LSTM `0.9138`, delta `+0.2130`
- tick `73148`, seconds `65.00`, LSTM `0.9290`, delta `+0.2075`
- tick `69628`, seconds `10.00`, LSTM `0.6675`, delta `+0.1065`
- tick `70076`, seconds `17.00`, LSTM `0.6914`, delta `+0.0476`
- tick `72252`, seconds `51.00`, LSTM `0.6325`, delta `+0.0411`
- tick `72476`, seconds `54.50`, LSTM `0.9703`, delta `+0.0367`
- tick `69724`, seconds `11.50`, LSTM `0.6686`, delta `-0.0325`
- tick `71580`, seconds `40.50`, LSTM `0.6748`, delta `+0.0307`
- tick `72284`, seconds `51.50`, LSTM `0.6600`, delta `+0.0275`

## Top 15 local ridge features

- `lag_00__damage_diff_last_5s`: coefficient `0.002513`, |coef| `0.002513`
- `lag_00__kill_diff_last_3s`: coefficient `0.002329`, |coef| `0.002329`
- `lag_00__CT_kills_last_3s`: coefficient `0.002253`, |coef| `0.002253`
- `lag_04__CT1__flash_duration`: coefficient `-0.002192`, |coef| `0.002192`
- `lag_00__CT_damage_last_5s`: coefficient `0.002154`, |coef| `0.002154`
- `lag_09__CT_shots_fired_sum`: coefficient `0.002065`, |coef| `0.002065`
- `lag_00__T_place_ARCH`: coefficient `-0.001953`, |coef| `0.001953`
- `lag_03__T_place_ARCH`: coefficient `0.001949`, |coef| `0.001949`
- `lag_05__CT_place_BALCONY`: coefficient `-0.001888`, |coef| `0.001888`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001777`, |coef| `0.001777`
- `lag_01__T_flashed_players`: coefficient `0.001744`, |coef| `0.001744`
- `lag_09__CT1__shots_fired`: coefficient `0.001722`, |coef| `0.001722`
- `lag_05__CT_place_PIT`: coefficient `0.001715`, |coef| `0.001715`
- `lag_12__CT_place_BANANA`: coefficient `-0.001664`, |coef| `0.001664`
- `lag_01__T4__flash_duration`: coefficient `0.001663`, |coef| `0.001663`

## Top 10 utility ridge features

- `lag_04__CT1__flash_duration`: coefficient `-0.002192` (lowers CT win probability)
- `lag_01__T4__flash_duration`: coefficient `0.001663` (raises CT win probability)
- `lag_10__T4__flash_duration`: coefficient `0.001251` (raises CT win probability)
- `lag_13__T4__flash_duration`: coefficient `-0.001181` (lowers CT win probability)
- `lag_08__T_A_site_active_infernos`: coefficient `-0.001045` (lowers CT win probability)
- `lag_01__T_flash_duration_sum`: coefficient `0.001044` (raises CT win probability)
- `lag_02__T3__molly`: coefficient `-0.000874` (lowers CT win probability)
- `lag_15__CT5__flash_duration`: coefficient `-0.000872` (lowers CT win probability)
- `lag_04__CT_flash_duration_sum`: coefficient `-0.000862` (lowers CT win probability)
- `lag_12__CT1__flash_duration`: coefficient `0.000788` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__damage_diff_last_5s`: coefficient `0.002513` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002329` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002253` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002154` (raises CT win probability)
- `lag_09__CT_shots_fired_sum`: coefficient `0.002065` (raises CT win probability)
- `lag_00__T_place_ARCH`: coefficient `-0.001953` (lowers CT win probability)
- `lag_03__T_place_ARCH`: coefficient `0.001949` (raises CT win probability)
- `lag_05__CT_place_BALCONY`: coefficient `-0.001888` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001777` (raises CT win probability)
- `lag_01__T_flashed_players`: coefficient `0.001744` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `72796`, seconds `59.50`, LSTM delta `-0.2521`

Top all feature movements:
- `lag_09__CT_shots_fired_sum`: contribution `-0.020083`
- `lag_09__CT1__shots_fired`: contribution `-0.012735`
- `lag_00__damage_diff_last_5s`: contribution `-0.009243`
- `lag_10__T4__flash_duration`: contribution `-0.006866`
- `lag_13__T4__flash_duration`: contribution `-0.006482`

Top utility-only movements:
- `lag_10__T4__flash_duration`: contribution `-0.006866`
- `lag_13__T4__flash_duration`: contribution `-0.006482`

### tick `72412`, seconds `53.50`, LSTM delta `+0.2130`

Top all feature movements:
- `lag_04__CT1__flash_duration`: contribution `+0.016206`
- `lag_05__CT_place_BALCONY`: contribution `+0.012118`
- `lag_01__T_flashed_players`: contribution `+0.010098`
- `lag_01__T4__flash_duration`: contribution `+0.009127`
- `lag_05__CT_place_PIT`: contribution `+0.007384`

Top utility-only movements:
- `lag_04__CT1__flash_duration`: contribution `+0.016206`
- `lag_01__T4__flash_duration`: contribution `+0.009127`
- `lag_01__T_flash_duration_sum`: contribution `+0.003611`
- `lag_04__CT_flash_duration_sum`: contribution `+0.002862`

### tick `73148`, seconds `65.00`, LSTM delta `+0.2075`

Top all feature movements:
- `lag_00__T_place_ARCH`: contribution `+0.018167`
- `lag_03__T_place_ARCH`: contribution `+0.018129`
- `lag_00__CT_kills_last_3s`: contribution `+0.006506`
- `lag_00__CT_shots_fired_sum`: contribution `+0.006172`
- `lag_00__kill_diff_last_3s`: contribution `+0.005606`

Top utility-only movements:
- `lag_08__T_A_site_active_infernos`: contribution `+0.003112`

### tick `69628`, seconds `10.00`, LSTM delta `+0.1065`

Top all feature movements:
- `lag_06__T_place_LOWERMID`: contribution `+0.009554`
- `lag_09__T_place_LOWERMID`: contribution `+0.006869`
- `lag_00__CT_kills_last_3s`: contribution `+0.006506`
- `lag_00__kill_diff_last_3s`: contribution `+0.005606`
- `lag_10__T_place_LOWERMID`: contribution `+0.005042`

Top utility-only movements:
- `lag_02__T3__molly`: contribution `+0.001941`

### tick `70076`, seconds `17.00`, LSTM delta `+0.0476`

Top all feature movements:
- `lag_07__T_shots_fired_sum`: contribution `+0.004247`
- `lag_00__CT_place_ARCH`: contribution `+0.004239`
- `lag_00__damage_diff_last_5s`: contribution `+0.002892`
- `lag_07__T1__shots_fired`: contribution `+0.002782`
- `lag_09__T_shots_fired_sum`: contribution `-0.002590`

Top utility-only movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.001926`
- `lag_12__T_utility_damage_last_5s`: contribution `+0.001436`
