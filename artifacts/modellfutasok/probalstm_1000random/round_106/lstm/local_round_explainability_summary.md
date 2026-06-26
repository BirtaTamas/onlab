# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-gamerlegion-vs-complexity-bo3-A8nOd44IyEYHGVOxrkExMv/gamerlegion-vs-complexity-m1-inferno.csv`
- round_num: `3`

## Largest probability jumps

- tick `22422`, seconds `87.50`, LSTM `0.8402`, delta `+0.2087`
- tick `19510`, seconds `42.00`, LSTM `0.5782`, delta `-0.1655`
- tick `22582`, seconds `90.00`, LSTM `0.8987`, delta `+0.1267`
- tick `22486`, seconds `88.50`, LSTM `0.7876`, delta `-0.1181`
- tick `19606`, seconds `43.50`, LSTM `0.4232`, delta `-0.0913`
- tick `19670`, seconds `44.50`, LSTM `0.5289`, delta `+0.0803`
- tick `21974`, seconds `80.50`, LSTM `0.5448`, delta `+0.0784`
- tick `19478`, seconds `41.50`, LSTM `0.7437`, delta `+0.0704`
- tick `22454`, seconds `88.00`, LSTM `0.9057`, delta `+0.0655`
- tick `22166`, seconds `83.50`, LSTM `0.6085`, delta `+0.0567`

## Top 15 local ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.001840`, |coef| `0.001840`
- `lag_09__T3__flash_duration`: coefficient `0.001522`, |coef| `0.001522`
- `lag_09__T4__flash_duration`: coefficient `0.001468`, |coef| `0.001468`
- `lag_09__T_flash_duration_sum`: coefficient `0.001451`, |coef| `0.001451`
- `lag_09__T5__flash_duration`: coefficient `0.001419`, |coef| `0.001419`
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.001199`, |coef| `0.001199`
- `lag_00__kill_diff_last_3s`: coefficient `0.001196`, |coef| `0.001196`
- `lag_15__CT_shots_fired_sum`: coefficient `0.001086`, |coef| `0.001086`
- `lag_00__CT_kills_last_3s`: coefficient `0.001002`, |coef| `0.001002`
- `lag_08__T1__flash_duration`: coefficient `-0.001001`, |coef| `0.001001`
- `lag_10__T_flash_duration_sum`: coefficient `0.000995`, |coef| `0.000995`
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000983`, |coef| `0.000983`
- `lag_08__CT_place_TOPOFMID`: coefficient `0.000973`, |coef| `0.000973`
- `lag_04__T3__shots_fired`: coefficient `0.000940`, |coef| `0.000940`
- `lag_08__CT_place_ARCH`: coefficient `-0.000933`, |coef| `0.000933`

## Top 10 utility ridge features

- `lag_09__T3__flash_duration`: coefficient `0.001522` (raises CT win probability)
- `lag_09__T4__flash_duration`: coefficient `0.001468` (raises CT win probability)
- `lag_09__T_flash_duration_sum`: coefficient `0.001451` (raises CT win probability)
- `lag_09__T5__flash_duration`: coefficient `0.001419` (raises CT win probability)
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.001199` (raises CT win probability)
- `lag_08__T1__flash_duration`: coefficient `-0.001001` (lowers CT win probability)
- `lag_10__T_flash_duration_sum`: coefficient `0.000995` (raises CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000983` (raises CT win probability)
- `lag_09__T2__flash_duration`: coefficient `-0.000818` (lowers CT win probability)
- `lag_14__T_flash_duration_sum`: coefficient `0.000810` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.001840` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001196` (raises CT win probability)
- `lag_15__CT_shots_fired_sum`: coefficient `0.001086` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001002` (raises CT win probability)
- `lag_08__CT_place_TOPOFMID`: coefficient `0.000973` (raises CT win probability)
- `lag_04__T3__shots_fired`: coefficient `0.000940` (raises CT win probability)
- `lag_08__CT_place_ARCH`: coefficient `-0.000933` (lowers CT win probability)
- `lag_08__T3__is_walking`: coefficient `0.000873` (raises CT win probability)
- `lag_03__T4__duck_amount`: coefficient `-0.000871` (lowers CT win probability)
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.000839` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `22422`, seconds `87.50`, LSTM delta `+0.2087`

Top all feature movements:
- `lag_09__T_flash_duration_sum`: contribution `+0.016777`
- `lag_09__T4__flash_duration`: contribution `+0.011724`
- `lag_09__T5__flash_duration`: contribution `+0.010848`
- `lag_09__T3__flash_duration`: contribution `+0.009057`
- `lag_08__T1__flash_duration`: contribution `+0.006567`

Top utility-only movements:
- `lag_09__T_flash_duration_sum`: contribution `+0.016777`
- `lag_09__T4__flash_duration`: contribution `+0.011724`
- `lag_09__T5__flash_duration`: contribution `+0.010848`
- `lag_09__T3__flash_duration`: contribution `+0.009057`
- `lag_08__T1__flash_duration`: contribution `+0.006567`

### tick `19510`, seconds `42.00`, LSTM delta `-0.1655`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.015171`
- `lag_09__T3__flash_duration`: contribution `-0.012057`
- `lag_01__CT_utility_damage_last_5s`: contribution `-0.008399`
- `lag_08__CT_place_ARCH`: contribution `-0.007616`
- `lag_08__CT_place_TOPOFMID`: contribution `-0.007059`

Top utility-only movements:
- `lag_09__T3__flash_duration`: contribution `-0.012057`
- `lag_01__CT_utility_damage_last_5s`: contribution `-0.008399`
- `lag_01__utility_damage_diff_last_5s`: contribution `-0.005554`
- `lag_09__T_flash_duration_sum`: contribution `-0.004791`
- `lag_10__T_flash_duration_sum`: contribution `-0.003070`

### tick `22582`, seconds `90.00`, LSTM delta `+0.1267`

Top all feature movements:
- `lag_14__T_flash_duration_sum`: contribution `+0.009366`
- `lag_14__T4__flash_duration`: contribution `+0.005288`
- `lag_14__T5__flash_duration`: contribution `+0.004934`
- `lag_14__T_flashed_players`: contribution `+0.004888`
- `lag_14__T1__flash_duration`: contribution `+0.004584`

Top utility-only movements:
- `lag_14__T_flash_duration_sum`: contribution `+0.009366`
- `lag_14__T4__flash_duration`: contribution `+0.005288`
- `lag_14__T5__flash_duration`: contribution `+0.004934`
- `lag_14__T1__flash_duration`: contribution `+0.004584`
- `lag_13__T1__flash_duration`: contribution `+0.002473`

### tick `22486`, seconds `88.50`, LSTM delta `-0.1181`

Top all feature movements:
- `lag_15__CT_shots_fired_sum`: contribution `-0.006788`
- `lag_04__T3__shots_fired`: contribution `-0.004556`
- `lag_02__T_bomb_zone_count`: contribution `-0.004472`
- `lag_04__T5__flash_duration`: contribution `-0.004353`
- `lag_15__CT3__shots_fired`: contribution `-0.003683`

Top utility-only movements:
- `lag_04__T5__flash_duration`: contribution `-0.004353`
- `lag_00__T3__flash_duration`: contribution `-0.003034`
- `lag_10__T_flash_duration_sum`: contribution `-0.002718`
- `lag_04__T_flash_duration_sum`: contribution `-0.002319`
- `lag_11__T1__flash_duration`: contribution `+0.002298`

### tick `19606`, seconds `43.50`, LSTM delta `-0.0913`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.023446`
- `lag_00__CT_utility_damage_last_5s`: contribution `-0.005146`
- `lag_02__T_shots_fired_sum`: contribution `-0.004616`
- `lag_14__T1__flash_duration`: contribution `-0.004556`
- `lag_11__CT_place_TOPOFMID`: contribution `-0.003467`

Top utility-only movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `-0.005146`
- `lag_14__T1__flash_duration`: contribution `-0.004556`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.003462`
- `lag_08__T1__flash_duration`: contribution `-0.002331`
- `lag_14__T_flash_duration_sum`: contribution `-0.002199`
