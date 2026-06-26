# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-faze-vs-virtuspro-bo3-YDlVsCnS6YPgcr85bBYoPq/faze-vs-virtus-pro-m3-inferno.csv`
- round_num: `12`

## Largest probability jumps

- tick `109200`, seconds `69.50`, LSTM `0.8202`, delta `+0.0986`
- tick `109168`, seconds `69.00`, LSTM `0.7216`, delta `+0.0475`
- tick `109328`, seconds `71.50`, LSTM `0.9350`, delta `+0.0457`
- tick `109296`, seconds `71.00`, LSTM `0.8893`, delta `+0.0401`
- tick `106800`, seconds `32.00`, LSTM `0.7211`, delta `+0.0397`
- tick `109520`, seconds `74.50`, LSTM `0.9649`, delta `+0.0383`
- tick `108976`, seconds `66.00`, LSTM `0.7035`, delta `+0.0374`
- tick `105488`, seconds `11.50`, LSTM `0.6712`, delta `-0.0315`
- tick `109936`, seconds `81.00`, LSTM `0.9635`, delta `+0.0311`
- tick `105648`, seconds `14.00`, LSTM `0.6646`, delta `+0.0286`

## Top 15 local ridge features

- `lag_00__CT_place_BALCONY`: coefficient `-0.001170`, |coef| `0.001170`
- `lag_05__T_flash_duration_sum`: coefficient `0.000875`, |coef| `0.000875`
- `lag_07__CT_place_BALCONY`: coefficient `-0.000837`, |coef| `0.000837`
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.000822`, |coef| `0.000822`
- `lag_05__T1__flash_duration`: coefficient `0.000719`, |coef| `0.000719`
- `lag_02__T5__flash_duration`: coefficient `0.000709`, |coef| `0.000709`
- `lag_08__CT_place_ARCH`: coefficient `0.000683`, |coef| `0.000683`
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000674`, |coef| `0.000674`
- `lag_04__T1__flash_duration`: coefficient `0.000632`, |coef| `0.000632`
- `lag_05__T3__flash_duration`: coefficient `0.000618`, |coef| `0.000618`
- `lag_04__T_flash_duration_sum`: coefficient `0.000610`, |coef| `0.000610`
- `lag_00__damage_diff_last_5s`: coefficient `0.000600`, |coef| `0.000600`
- `lag_00__CT_damage_last_5s`: coefficient `0.000580`, |coef| `0.000580`
- `lag_07__CT_place_ARCH`: coefficient `0.000569`, |coef| `0.000569`
- `lag_05__CT_place_ARCH`: coefficient `0.000564`, |coef| `0.000564`

## Top 10 utility ridge features

- `lag_05__T_flash_duration_sum`: coefficient `0.000875` (raises CT win probability)
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.000822` (raises CT win probability)
- `lag_05__T1__flash_duration`: coefficient `0.000719` (raises CT win probability)
- `lag_02__T5__flash_duration`: coefficient `0.000709` (raises CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000674` (raises CT win probability)
- `lag_04__T1__flash_duration`: coefficient `0.000632` (raises CT win probability)
- `lag_05__T3__flash_duration`: coefficient `0.000618` (raises CT win probability)
- `lag_04__T_flash_duration_sum`: coefficient `0.000610` (raises CT win probability)
- `lag_00__T4__flash_duration`: coefficient `-0.000555` (lowers CT win probability)
- `lag_05__T4__flash_duration`: coefficient `0.000545` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_BALCONY`: coefficient `-0.001170` (lowers CT win probability)
- `lag_07__CT_place_BALCONY`: coefficient `-0.000837` (lowers CT win probability)
- `lag_08__CT_place_ARCH`: coefficient `0.000683` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000600` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.000580` (raises CT win probability)
- `lag_07__CT_place_ARCH`: coefficient `0.000569` (raises CT win probability)
- `lag_05__CT_place_ARCH`: coefficient `0.000564` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000561` (raises CT win probability)
- `lag_05__T_flashed_players`: coefficient `0.000556` (raises CT win probability)
- `lag_04__CT_place_RUINS`: coefficient `0.000550` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `109200`, seconds `69.50`, LSTM delta `+0.0986`

Top all feature movements:
- `lag_05__T_flash_duration_sum`: contribution `+0.007484`
- `lag_07__CT_place_BALCONY`: contribution `+0.005375`
- `lag_05__T1__flash_duration`: contribution `+0.005196`
- `lag_02__T5__flash_duration`: contribution `+0.004581`
- `lag_05__T_flashed_players`: contribution `+0.004295`

Top utility-only movements:
- `lag_05__T_flash_duration_sum`: contribution `+0.007484`
- `lag_05__T1__flash_duration`: contribution `+0.005196`
- `lag_02__T5__flash_duration`: contribution `+0.004581`
- `lag_05__T3__flash_duration`: contribution `+0.004217`
- `lag_00__T4__flash_duration`: contribution `+0.003006`

### tick `109168`, seconds `69.00`, LSTM delta `+0.0475`

Top all feature movements:
- `lag_04__T_flash_duration_sum`: contribution `+0.005219`
- `lag_04__T1__flash_duration`: contribution `+0.004564`
- `lag_04__T_flashed_players`: contribution `+0.004100`
- `lag_01__T5__flash_duration`: contribution `+0.002948`
- `lag_06__CT_place_BALCONY`: contribution `+0.002807`

Top utility-only movements:
- `lag_04__T_flash_duration_sum`: contribution `+0.005219`
- `lag_04__T1__flash_duration`: contribution `+0.004564`
- `lag_01__T5__flash_duration`: contribution `+0.002948`
- `lag_04__T3__flash_duration`: contribution `+0.002749`
- `lag_04__T4__flash_duration`: contribution `+0.001336`

### tick `109328`, seconds `71.50`, LSTM delta `+0.0457`

Top all feature movements:
- `lag_09__T_flash_duration_sum`: contribution `+0.003148`
- `lag_09__T3__flash_duration`: contribution `+0.002629`
- `lag_11__CT_place_BALCONY`: contribution `+0.002557`
- `lag_09__T1__flash_duration`: contribution `+0.002068`
- `lag_09__T_flashed_players`: contribution `+0.002011`

Top utility-only movements:
- `lag_09__T_flash_duration_sum`: contribution `+0.003148`
- `lag_09__T3__flash_duration`: contribution `+0.002629`
- `lag_09__T1__flash_duration`: contribution `+0.002068`
- `lag_04__T4__flash_duration`: contribution `-0.001354`
- `lag_04__T_flash_duration_sum`: contribution `-0.001345`

### tick `109296`, seconds `71.00`, LSTM delta `+0.0401`

Top all feature movements:
- `lag_08__T_flash_duration_sum`: contribution `+0.004058`
- `lag_08__T3__flash_duration`: contribution `+0.002867`
- `lag_05__T5__flash_duration`: contribution `+0.002600`
- `lag_08__T1__flash_duration`: contribution `+0.002563`
- `lag_10__CT_place_BALCONY`: contribution `+0.002346`

Top utility-only movements:
- `lag_08__T_flash_duration_sum`: contribution `+0.004058`
- `lag_08__T3__flash_duration`: contribution `+0.002867`
- `lag_05__T5__flash_duration`: contribution `+0.002600`
- `lag_08__T1__flash_duration`: contribution `+0.002563`
- `lag_05__T_flash_duration_sum`: contribution `+0.002227`

### tick `106800`, seconds `32.00`, LSTM delta `+0.0397`

Top all feature movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.005881`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.003957`
- `lag_04__CT_place_ARCH`: contribution `+0.002027`
- `lag_04__CT_place_RUINS`: contribution `+0.001922`
- `lag_00__T3__duck_amount`: contribution `-0.001464`

Top utility-only movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.005881`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.003957`
