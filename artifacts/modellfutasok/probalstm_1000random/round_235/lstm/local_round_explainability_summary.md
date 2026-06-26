# Local Round Explainability

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-falcons-vs-g2-bo3-Xf_UKx9fB2btv0Vy_VBVUC/falcons-vs-g2-m3-mirage.csv`
- round_num: `17`

## Largest probability jumps

- tick `124831`, seconds `47.00`, LSTM `0.1985`, delta `-0.2900`
- tick `124575`, seconds `43.00`, LSTM `0.4287`, delta `-0.1949`
- tick `122623`, seconds `12.50`, LSTM `0.6080`, delta `+0.1170`
- tick `124735`, seconds `45.50`, LSTM `0.3985`, delta `+0.0762`
- tick `124767`, seconds `46.00`, LSTM `0.4573`, delta `+0.0588`
- tick `124607`, seconds `43.50`, LSTM `0.3802`, delta `-0.0484`
- tick `124863`, seconds `47.50`, LSTM `0.1514`, delta `-0.0471`
- tick `124639`, seconds `44.00`, LSTM `0.3417`, delta `-0.0385`
- tick `124511`, seconds `42.00`, LSTM `0.6253`, delta `+0.0333`
- tick `122591`, seconds `12.00`, LSTM `0.4910`, delta `+0.0316`

## Top 15 local ridge features

- `lag_07__T_place_PALACEINTERIOR`: coefficient `0.001740`, |coef| `0.001740`
- `lag_03__T4__flash_duration`: coefficient `-0.001727`, |coef| `0.001727`
- `lag_12__CT_place_TRUCK`: coefficient `0.001651`, |coef| `0.001651`
- `lag_10__CT2__flash_duration`: coefficient `-0.001631`, |coef| `0.001631`
- `lag_00__CT_place_JUNGLE`: coefficient `0.001614`, |coef| `0.001614`
- `lag_03__CT_place_STAIRS`: coefficient `-0.001522`, |coef| `0.001522`
- `lag_02__CT2__flash_duration`: coefficient `-0.001495`, |coef| `0.001495`
- `lag_11__CT_place_STAIRS`: coefficient `-0.001476`, |coef| `0.001476`
- `lag_00__T_kills_last_3s`: coefficient `-0.001371`, |coef| `0.001371`
- `lag_00__kill_diff_last_3s`: coefficient `0.001283`, |coef| `0.001283`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001260`, |coef| `0.001260`
- `lag_10__CT_flash_duration_sum`: coefficient `-0.001248`, |coef| `0.001248`
- `lag_10__CT3__flash_duration`: coefficient `-0.001244`, |coef| `0.001244`
- `lag_01__CT2__shots_fired`: coefficient `-0.001242`, |coef| `0.001242`
- `lag_02__CT3__flash_duration`: coefficient `-0.001228`, |coef| `0.001228`

## Top 10 utility ridge features

- `lag_03__T4__flash_duration`: coefficient `-0.001727` (lowers CT win probability)
- `lag_10__CT2__flash_duration`: coefficient `-0.001631` (lowers CT win probability)
- `lag_02__CT2__flash_duration`: coefficient `-0.001495` (lowers CT win probability)
- `lag_10__CT_flash_duration_sum`: coefficient `-0.001248` (lowers CT win probability)
- `lag_10__CT3__flash_duration`: coefficient `-0.001244` (lowers CT win probability)
- `lag_02__CT3__flash_duration`: coefficient `-0.001228` (lowers CT win probability)
- `lag_02__CT_flash_duration_sum`: coefficient `-0.001211` (lowers CT win probability)
- `lag_00__T4__flash_duration`: coefficient `0.001202` (raises CT win probability)
- `lag_03__T1__flash_duration`: coefficient `-0.001067` (lowers CT win probability)
- `lag_10__T_utility_damage_last_5s`: coefficient `-0.001063` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_07__T_place_PALACEINTERIOR`: coefficient `0.001740` (raises CT win probability)
- `lag_12__CT_place_TRUCK`: coefficient `0.001651` (raises CT win probability)
- `lag_00__CT_place_JUNGLE`: coefficient `0.001614` (raises CT win probability)
- `lag_03__CT_place_STAIRS`: coefficient `-0.001522` (lowers CT win probability)
- `lag_11__CT_place_STAIRS`: coefficient `-0.001476` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001371` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001283` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001260` (raises CT win probability)
- `lag_01__CT2__shots_fired`: coefficient `-0.001242` (lowers CT win probability)
- `lag_02__CT_place_TRUCK`: coefficient `0.001226` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `124831`, seconds `47.00`, LSTM delta `-0.2900`

Top all feature movements:
- `lag_03__T4__flash_duration`: contribution `-0.013672`
- `lag_10__CT2__flash_duration`: contribution `-0.012481`
- `lag_11__CT_place_STAIRS`: contribution `-0.011487`
- `lag_00__CT_place_JUNGLE`: contribution `-0.010353`
- `lag_10__CT_flash_duration_sum`: contribution `-0.007490`

Top utility-only movements:
- `lag_03__T4__flash_duration`: contribution `-0.013672`
- `lag_10__CT2__flash_duration`: contribution `-0.012481`
- `lag_10__CT_flash_duration_sum`: contribution `-0.007490`
- `lag_10__CT3__flash_duration`: contribution `-0.007019`
- `lag_03__T1__flash_duration`: contribution `-0.005696`

### tick `124575`, seconds `43.00`, LSTM delta `-0.1949`

Top all feature movements:
- `lag_03__CT_place_STAIRS`: contribution `-0.011843`
- `lag_02__CT2__flash_duration`: contribution `-0.011440`
- `lag_12__CT_place_TRUCK`: contribution `-0.010647`
- `lag_02__CT_flash_duration_sum`: contribution `-0.007263`
- `lag_02__CT3__flash_duration`: contribution `-0.006926`

Top utility-only movements:
- `lag_02__CT2__flash_duration`: contribution `-0.011440`
- `lag_02__CT_flash_duration_sum`: contribution `-0.007263`
- `lag_02__CT3__flash_duration`: contribution `-0.006926`
- `lag_10__T_utility_damage_last_5s`: contribution `-0.005918`
- `lag_10__utility_damage_diff_last_5s`: contribution `-0.002381`

### tick `122623`, seconds `12.50`, LSTM delta `+0.1170`

Top all feature movements:
- `lag_02__CT_place_LADDER`: contribution `+0.011895`
- `lag_06__CT_place_LADDER`: contribution `+0.009483`
- `lag_02__CT_place_TRUCK`: contribution `+0.007907`
- `lag_00__CT_place_SNIPERSNEST`: contribution `+0.005702`
- `lag_09__CT_place_SNIPERSNEST`: contribution `+0.003789`

Top utility-only movements:
- `lag_10__T2__flash_duration`: contribution `+0.001989`
- `lag_02__T2__flash_duration`: contribution `+0.001879`

### tick `124735`, seconds `45.50`, LSTM delta `+0.0762`

Top all feature movements:
- `lag_00__T4__flash_duration`: contribution `+0.009519`
- `lag_08__CT_place_STAIRS`: contribution `+0.007977`
- `lag_07__CT2__flash_duration`: contribution `+0.006638`
- `lag_07__CT_flash_duration_sum`: contribution `+0.003960`
- `lag_07__CT3__flash_duration`: contribution `+0.003355`

Top utility-only movements:
- `lag_00__T4__flash_duration`: contribution `+0.009519`
- `lag_07__CT2__flash_duration`: contribution `+0.006638`
- `lag_07__CT_flash_duration_sum`: contribution `+0.003960`
- `lag_07__CT3__flash_duration`: contribution `+0.003355`
- `lag_00__T1__flash_duration`: contribution `+0.003074`

### tick `124767`, seconds `46.00`, LSTM delta `+0.0588`

Top all feature movements:
- `lag_08__CT2__flash_duration`: contribution `+0.006649`
- `lag_07__T_place_PALACEINTERIOR`: contribution `+0.005836`
- `lag_15__CT_place_JUNGLE`: contribution `+0.005456`
- `lag_01__T4__flash_duration`: contribution `+0.004951`
- `lag_00__T_kills_last_3s`: contribution `+0.004344`

Top utility-only movements:
- `lag_08__CT2__flash_duration`: contribution `+0.006649`
- `lag_01__T4__flash_duration`: contribution `+0.004951`
- `lag_08__CT_flash_duration_sum`: contribution `+0.002640`
- `lag_01__T1__flash_duration`: contribution `+0.001619`
