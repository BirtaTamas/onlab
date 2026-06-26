# Local Round Explainability

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-falcons-vs-heroic-bo3-ReZhZ3UThZvWjRyUeuYiIR/falcons-vs-heroic-m3-dust2.csv`
- round_num: `8`

## Largest probability jumps

- tick `60779`, seconds `78.50`, LSTM `0.1506`, delta `-0.3759`
- tick `60427`, seconds `73.00`, LSTM `0.3644`, delta `-0.2623`
- tick `60683`, seconds `77.00`, LSTM `0.4313`, delta `+0.2616`
- tick `60747`, seconds `78.00`, LSTM `0.5265`, delta `+0.1227`
- tick `60459`, seconds `73.50`, LSTM `0.2509`, delta `-0.1135`
- tick `58187`, seconds `38.00`, LSTM `0.6216`, delta `+0.0566`
- tick `60331`, seconds `71.50`, LSTM `0.6104`, delta `+0.0545`
- tick `60811`, seconds `79.00`, LSTM `0.0988`, delta `-0.0518`
- tick `60875`, seconds `80.00`, LSTM `0.0126`, delta `-0.0452`
- tick `60843`, seconds `79.50`, LSTM `0.0578`, delta `-0.0410`

## Top 15 local ridge features

- `lag_03__T_place_SHORTSTAIRS`: coefficient `-0.002547`, |coef| `0.002547`
- `lag_11__CT2__flash_duration`: coefficient `-0.002325`, |coef| `0.002325`
- `lag_03__T1__flash_duration`: coefficient `-0.002164`, |coef| `0.002164`
- `lag_00__T4__shots_fired`: coefficient `0.002001`, |coef| `0.002001`
- `lag_03__T_place_EXTENDEDA`: coefficient `0.001977`, |coef| `0.001977`
- `lag_07__T_place_LONGA`: coefficient `-0.001886`, |coef| `0.001886`
- `lag_00__kill_diff_last_3s`: coefficient `0.001751`, |coef| `0.001751`
- `lag_04__CT2__is_scoped`: coefficient `0.001712`, |coef| `0.001712`
- `lag_03__CT_place_ARAMP`: coefficient `0.001689`, |coef| `0.001689`
- `lag_00__T_kills_last_3s`: coefficient `-0.001661`, |coef| `0.001661`
- `lag_07__T_place_EXTENDEDA`: coefficient `-0.001646`, |coef| `0.001646`
- `lag_07__CT2__is_scoped`: coefficient `-0.001632`, |coef| `0.001632`
- `lag_14__T1__flash_duration`: coefficient `-0.001603`, |coef| `0.001603`
- `lag_00__damage_diff_last_5s`: coefficient `0.001506`, |coef| `0.001506`
- `lag_03__T_flash_duration_sum`: coefficient `-0.001498`, |coef| `0.001498`

## Top 10 utility ridge features

- `lag_11__CT2__flash_duration`: coefficient `-0.002325` (lowers CT win probability)
- `lag_03__T1__flash_duration`: coefficient `-0.002164` (lowers CT win probability)
- `lag_14__T1__flash_duration`: coefficient `-0.001603` (lowers CT win probability)
- `lag_03__T_flash_duration_sum`: coefficient `-0.001498` (lowers CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `-0.001484` (lowers CT win probability)
- `lag_10__CT4__flash_duration`: coefficient `-0.001435` (lowers CT win probability)
- `lag_14__T4__flash_duration`: coefficient `-0.001420` (lowers CT win probability)
- `lag_03__T4__flash_duration`: coefficient `-0.001375` (lowers CT win probability)
- `lag_14__T_flash_duration_sum`: coefficient `-0.001270` (lowers CT win probability)
- `lag_10__T3__flash_duration`: coefficient `-0.001223` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_03__T_place_SHORTSTAIRS`: coefficient `-0.002547` (lowers CT win probability)
- `lag_00__T4__shots_fired`: coefficient `0.002001` (raises CT win probability)
- `lag_03__T_place_EXTENDEDA`: coefficient `0.001977` (raises CT win probability)
- `lag_07__T_place_LONGA`: coefficient `-0.001886` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001751` (raises CT win probability)
- `lag_04__CT2__is_scoped`: coefficient `0.001712` (raises CT win probability)
- `lag_03__CT_place_ARAMP`: coefficient `0.001689` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001661` (lowers CT win probability)
- `lag_07__T_place_EXTENDEDA`: coefficient `-0.001646` (lowers CT win probability)
- `lag_07__CT2__is_scoped`: coefficient `-0.001632` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `60779`, seconds `78.50`, LSTM delta `-0.3759`

Top all feature movements:
- `lag_00__T4__shots_fired`: contribution `-0.016069`
- `lag_14__T1__flash_duration`: contribution `-0.012545`
- `lag_04__CT2__is_scoped`: contribution `-0.010480`
- `lag_07__CT2__is_scoped`: contribution `-0.009986`
- `lag_03__T_place_EXTENDEDA`: contribution `-0.009800`

Top utility-only movements:
- `lag_14__T1__flash_duration`: contribution `-0.012545`
- `lag_14__T4__flash_duration`: contribution `-0.009249`
- `lag_10__T3__flash_duration`: contribution `-0.008073`
- `lag_14__T_flash_duration_sum`: contribution `-0.007501`
- `lag_10__CT1__flash_duration`: contribution `-0.007451`

### tick `60427`, seconds `73.00`, LSTM delta `-0.2623`

Top all feature movements:
- `lag_03__T1__flash_duration`: contribution `-0.016936`
- `lag_11__CT2__flash_duration`: contribution `-0.012493`
- `lag_03__T_place_SHORTSTAIRS`: contribution `-0.010704`
- `lag_03__CT_place_ARAMP`: contribution `-0.010520`
- `lag_03__T4__flash_duration`: contribution `-0.008960`

Top utility-only movements:
- `lag_03__T1__flash_duration`: contribution `-0.016936`
- `lag_11__CT2__flash_duration`: contribution `-0.012493`
- `lag_03__T4__flash_duration`: contribution `-0.008960`
- `lag_03__T_flash_duration_sum`: contribution `-0.008847`
- `lag_03__CT2__flash_duration`: contribution `-0.005826`

### tick `60683`, seconds `77.00`, LSTM delta `+0.2616`

Top all feature movements:
- `lag_11__CT2__flash_duration`: contribution `+0.011594`
- `lag_03__T_place_SHORTSTAIRS`: contribution `+0.010704`
- `lag_04__CT2__is_scoped`: contribution `+0.010480`
- `lag_03__T_place_EXTENDEDA`: contribution `+0.009800`
- `lag_11__T1__flash_duration`: contribution `+0.009392`

Top utility-only movements:
- `lag_11__CT2__flash_duration`: contribution `+0.011594`
- `lag_11__T1__flash_duration`: contribution `+0.009392`
- `lag_07__CT1__flash_duration`: contribution `+0.006317`
- `lag_11__T4__flash_duration`: contribution `+0.006018`
- `lag_07__T3__flash_duration`: contribution `+0.005729`

### tick `60747`, seconds `78.00`, LSTM delta `+0.1227`

Top all feature movements:
- `lag_03__T_place_SHORTSTAIRS`: contribution `+0.010704`
- `lag_03__T_place_EXTENDEDA`: contribution `+0.009800`
- `lag_01__CT2__is_scoped`: contribution `+0.007396`
- `lag_00__T4__shots_fired`: contribution `+0.006181`
- `lag_02__CT2__is_scoped`: contribution `+0.005535`

Top utility-only movements:
- `lag_09__T3__flash_duration`: contribution `+0.002387`

### tick `60459`, seconds `73.50`, LSTM delta `-0.1135`

Top all feature movements:
- `lag_03__T_place_SHORTSTAIRS`: contribution `-0.010704`
- `lag_04__T1__flash_duration`: contribution `-0.008298`
- `lag_00__CT4__flash_duration`: contribution `-0.007496`
- `lag_04__T4__flash_duration`: contribution `-0.005607`
- `lag_00__T3__flash_duration`: contribution `-0.005564`

Top utility-only movements:
- `lag_04__T1__flash_duration`: contribution `-0.008298`
- `lag_00__CT4__flash_duration`: contribution `-0.007496`
- `lag_04__T4__flash_duration`: contribution `-0.005607`
- `lag_00__T3__flash_duration`: contribution `-0.005564`
- `lag_04__T_flash_duration_sum`: contribution `-0.005134`
