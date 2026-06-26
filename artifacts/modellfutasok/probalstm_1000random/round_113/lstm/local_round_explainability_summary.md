# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-tyloo-vs-falcons-bo3-MBKGKnSCeuy54EHzS5mmW8/tyloo-vs-falcons-m2-ancient.csv`
- round_num: `2`

## Largest probability jumps

- tick `8043`, seconds `8.50`, LSTM `0.1720`, delta `-0.0764`
- tick `8459`, seconds `15.00`, LSTM `0.1068`, delta `-0.0709`
- tick `8523`, seconds `16.00`, LSTM `0.0257`, delta `-0.0585`
- tick `7851`, seconds `5.50`, LSTM `0.2359`, delta `-0.0476`
- tick `8331`, seconds `13.00`, LSTM `0.1625`, delta `+0.0416`
- tick `8363`, seconds `13.50`, LSTM `0.1892`, delta `+0.0267`
- tick `7531`, seconds `0.50`, LSTM `0.2292`, delta `-0.0260`
- tick `8075`, seconds `9.00`, LSTM `0.1467`, delta `-0.0252`
- tick `8491`, seconds `15.50`, LSTM `0.0841`, delta `-0.0227`
- tick `7659`, seconds `2.50`, LSTM `0.2660`, delta `+0.0199`

## Top 15 local ridge features

- `lag_00__CT_place_UNKNOWN`: coefficient `0.001099`, |coef| `0.001099`
- `lag_15__CT_place_UNKNOWN`: coefficient `0.000861`, |coef| `0.000861`
- `lag_03__T3__shots_fired`: coefficient `-0.000815`, |coef| `0.000815`
- `lag_06__CT_flashes_last_5s`: coefficient `0.000765`, |coef| `0.000765`
- `lag_02__T3__shots_fired`: coefficient `-0.000758`, |coef| `0.000758`
- `lag_01__T3__shots_fired`: coefficient `-0.000754`, |coef| `0.000754`
- `lag_05__T3__shots_fired`: coefficient `-0.000679`, |coef| `0.000679`
- `lag_00__T3__shots_fired`: coefficient `-0.000605`, |coef| `0.000605`
- `lag_12__T3__shots_fired`: coefficient `-0.000591`, |coef| `0.000591`
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.000584`, |coef| `0.000584`
- `lag_13__T3__shots_fired`: coefficient `-0.000563`, |coef| `0.000563`
- `lag_01__CT_place_UNKNOWN`: coefficient `-0.000559`, |coef| `0.000559`
- `lag_04__T3__shots_fired`: coefficient `-0.000552`, |coef| `0.000552`
- `lag_11__CT_place_UNKNOWN`: coefficient `-0.000494`, |coef| `0.000494`
- `lag_01__CT_place_TOPOFMID`: coefficient `0.000488`, |coef| `0.000488`

## Top 10 utility ridge features

- `lag_06__CT_flashes_last_5s`: coefficient `0.000765` (raises CT win probability)
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.000584` (lowers CT win probability)
- `lag_04__CT_flashes_last_5s`: coefficient `0.000428` (raises CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000369` (raises CT win probability)
- `lag_01__CT_flashes_last_5s`: coefficient `0.000333` (raises CT win probability)
- `lag_13__T_utility_damage_last_5s`: coefficient `-0.000321` (lowers CT win probability)
- `lag_10__T3__flash_duration`: coefficient `-0.000315` (lowers CT win probability)
- `lag_02__CT_flashes_last_5s`: coefficient `0.000312` (raises CT win probability)
- `lag_15__CT_flashes_last_5s`: coefficient `-0.000305` (lowers CT win probability)
- `lag_06__CT3__flash_duration`: coefficient `-0.000304` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_UNKNOWN`: coefficient `0.001099` (raises CT win probability)
- `lag_15__CT_place_UNKNOWN`: coefficient `0.000861` (raises CT win probability)
- `lag_03__T3__shots_fired`: coefficient `-0.000815` (lowers CT win probability)
- `lag_02__T3__shots_fired`: coefficient `-0.000758` (lowers CT win probability)
- `lag_01__T3__shots_fired`: coefficient `-0.000754` (lowers CT win probability)
- `lag_05__T3__shots_fired`: coefficient `-0.000679` (lowers CT win probability)
- `lag_00__T3__shots_fired`: coefficient `-0.000605` (lowers CT win probability)
- `lag_12__T3__shots_fired`: coefficient `-0.000591` (lowers CT win probability)
- `lag_13__T3__shots_fired`: coefficient `-0.000563` (lowers CT win probability)
- `lag_01__CT_place_UNKNOWN`: coefficient `-0.000559` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `8043`, seconds `8.50`, LSTM delta `-0.0764`

Top all feature movements:
- `lag_15__CT_place_UNKNOWN`: contribution `-0.018144`
- `lag_06__CT_flashes_last_5s`: contribution `-0.008416`
- `lag_14__CT_place_UNKNOWN`: contribution `-0.004440`
- `lag_08__CT_place_HOUSE`: contribution `-0.004155`
- `lag_00__T_utility_damage_last_5s`: contribution `-0.003751`

Top utility-only movements:
- `lag_06__CT_flashes_last_5s`: contribution `-0.008416`
- `lag_00__T_utility_damage_last_5s`: contribution `-0.003751`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.001500`

### tick `8459`, seconds `15.00`, LSTM delta `-0.0709`

Top all feature movements:
- `lag_01__T3__shots_fired`: contribution `-0.002739`
- `lag_10__T3__flash_duration`: contribution `-0.002249`
- `lag_13__T_utility_damage_last_5s`: contribution `-0.002063`
- `lag_02__T3__shots_fired`: contribution `-0.001837`
- `lag_03__T_utility_damage_last_5s`: contribution `-0.001813`

Top utility-only movements:
- `lag_10__T3__flash_duration`: contribution `-0.002249`
- `lag_13__T_utility_damage_last_5s`: contribution `-0.002063`
- `lag_03__T_utility_damage_last_5s`: contribution `-0.001813`
- `lag_06__CT3__flash_duration`: contribution `-0.001761`
- `lag_00__T3__flash_duration`: contribution `-0.001489`

### tick `8523`, seconds `16.00`, LSTM delta `-0.0585`

Top all feature movements:
- `lag_03__T3__shots_fired`: contribution `-0.002960`
- `lag_11__T3__shots_fired`: contribution `+0.002306`
- `lag_12__T3__flash_duration`: contribution `-0.001886`
- `lag_02__T3__shots_fired`: contribution `-0.001837`
- `lag_12__T3__shots_fired`: contribution `-0.001789`

Top utility-only movements:
- `lag_12__T3__flash_duration`: contribution `-0.001886`
- `lag_15__T_utility_damage_last_5s`: contribution `-0.001680`
- `lag_02__T3__flash_duration`: contribution `-0.001633`
- `lag_08__CT3__flash_duration`: contribution `-0.001519`
- `lag_05__T_utility_damage_last_5s`: contribution `-0.001358`

### tick `7851`, seconds `5.50`, LSTM delta `-0.0476`

Top all feature movements:
- `lag_11__CT_place_UNKNOWN`: contribution `-0.017323`
- `lag_09__CT_place_UNKNOWN`: contribution `-0.006146`
- `lag_08__CT_place_UNKNOWN`: contribution `-0.006113`
- `lag_00__CT_flashes_last_5s`: contribution `-0.002621`
- `lag_07__T_place_TUNNEL`: contribution `+0.001500`

Top utility-only movements:
- `lag_00__CT_flashes_last_5s`: contribution `-0.002621`
- `lag_10__CT_flashes_last_5s`: contribution `-0.001101`
- `lag_11__CT_flash_alpha_mean`: contribution `-0.000328`

### tick `8331`, seconds `13.00`, LSTM delta `+0.0416`

Top all feature movements:
- `lag_05__T3__shots_fired`: contribution `+0.003703`
- `lag_15__CT_flashes_last_5s`: contribution `+0.003352`
- `lag_06__T3__flash_duration`: contribution `+0.001724`
- `lag_05__T_shots_fired_sum`: contribution `+0.001620`
- `lag_11__CT_place_HOUSE`: contribution `+0.001486`

Top utility-only movements:
- `lag_15__CT_flashes_last_5s`: contribution `+0.003352`
- `lag_06__T3__flash_duration`: contribution `+0.001724`
- `lag_02__CT3__flash_duration`: contribution `+0.001484`
- `lag_09__T_utility_damage_last_5s`: contribution `+0.001482`
