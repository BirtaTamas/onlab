# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-fnatic-vs-legacy-bo3-XoJZ8zL16kSaGnHRZrLL4s/legacy-vs-fnatic-m1-ancient.csv`
- round_num: `8`

## Largest probability jumps

- tick `58773`, seconds `29.00`, LSTM `0.1136`, delta `-0.4135`
- tick `58805`, seconds `29.50`, LSTM `0.4081`, delta `+0.2946`
- tick `58677`, seconds `27.50`, LSTM `0.5234`, delta `-0.2900`
- tick `59765`, seconds `44.50`, LSTM `0.7539`, delta `+0.2292`
- tick `62229`, seconds `83.00`, LSTM `0.7216`, delta `+0.1894`
- tick `62485`, seconds `87.00`, LSTM `0.9204`, delta `+0.1424`
- tick `58325`, seconds `22.00`, LSTM `0.8233`, delta `+0.1035`
- tick `59797`, seconds `45.00`, LSTM `0.6863`, delta `-0.0676`
- tick `62197`, seconds `82.50`, LSTM `0.5322`, delta `+0.0592`
- tick `58549`, seconds `25.50`, LSTM `0.8425`, delta `-0.0572`

## Top 15 local ridge features

- `lag_00__CT_defusing_count`: coefficient `0.006193`, |coef| `0.006193`
- `lag_00__damage_diff_last_5s`: coefficient `0.003743`, |coef| `0.003743`
- `lag_00__kill_diff_last_3s`: coefficient `0.003076`, |coef| `0.003076`
- `lag_01__CT_defusing_count`: coefficient `0.003045`, |coef| `0.003045`
- `lag_00__CT_kills_last_3s`: coefficient `0.002985`, |coef| `0.002985`
- `lag_05__CT1__is_walking`: coefficient `-0.002464`, |coef| `0.002464`
- `lag_03__CT_defusing_count`: coefficient `0.002460`, |coef| `0.002460`
- `lag_08__CT_defusing_count`: coefficient `0.002433`, |coef| `0.002433`
- `lag_08__CT4__flash_duration`: coefficient `-0.002359`, |coef| `0.002359`
- `lag_02__T_velocity_mean`: coefficient `0.002325`, |coef| `0.002325`
- `lag_04__CT_defusing_count`: coefficient `0.002298`, |coef| `0.002298`
- `lag_00__T1__duck_amount`: coefficient `-0.002256`, |coef| `0.002256`
- `lag_05__CT_defusing_count`: coefficient `0.002239`, |coef| `0.002239`
- `lag_10__CT3__is_walking`: coefficient `-0.002076`, |coef| `0.002076`
- `lag_02__CT_defusing_count`: coefficient `0.002061`, |coef| `0.002061`

## Top 10 utility ridge features

- `lag_08__CT4__flash_duration`: coefficient `-0.002359` (lowers CT win probability)
- `lag_10__T5__flash_duration`: coefficient `-0.001917` (lowers CT win probability)
- `lag_11__CT4__flash_duration`: coefficient `-0.001701` (lowers CT win probability)
- `lag_00__T2__flash_duration`: coefficient `0.001697` (raises CT win probability)
- `lag_07__T5__flash_duration`: coefficient `-0.001659` (lowers CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.001521` (lowers CT win probability)
- `lag_10__T3__flash_duration`: coefficient `-0.001518` (lowers CT win probability)
- `lag_15__T2__flash_duration`: coefficient `-0.001484` (lowers CT win probability)
- `lag_09__CT3__smoke`: coefficient `-0.001459` (lowers CT win probability)
- `lag_07__CT4__flash_duration`: coefficient `0.001423` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_defusing_count`: coefficient `0.006193` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003743` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003076` (raises CT win probability)
- `lag_01__CT_defusing_count`: coefficient `0.003045` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002985` (raises CT win probability)
- `lag_05__CT1__is_walking`: coefficient `-0.002464` (lowers CT win probability)
- `lag_03__CT_defusing_count`: coefficient `0.002460` (raises CT win probability)
- `lag_08__CT_defusing_count`: coefficient `0.002433` (raises CT win probability)
- `lag_02__T_velocity_mean`: coefficient `0.002325` (raises CT win probability)
- `lag_04__CT_defusing_count`: coefficient `0.002298` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `58773`, seconds `29.00`, LSTM delta `-0.4135`

Top all feature movements:
- `lag_00__CT_place_TSIDEUPPER`: contribution `-0.013472`
- `lag_10__T5__flash_duration`: contribution `-0.013148`
- `lag_00__damage_diff_last_5s`: contribution `-0.011146`
- `lag_11__CT4__flash_duration`: contribution `-0.010783`
- `lag_00__T1__duck_amount`: contribution `-0.008835`

Top utility-only movements:
- `lag_10__T5__flash_duration`: contribution `-0.013148`
- `lag_11__CT4__flash_duration`: contribution `-0.010783`
- `lag_07__CT4__flash_duration`: contribution `-0.008749`
- `lag_00__T2__flash_duration`: contribution `-0.008114`
- `lag_15__T2__flash_duration`: contribution `-0.007936`

### tick `58805`, seconds `29.50`, LSTM delta `+0.2946`

Top all feature movements:
- `lag_08__CT4__flash_duration`: contribution `+0.014504`
- `lag_00__T1__duck_amount`: contribution `+0.008835`
- `lag_00__CT_kills_last_3s`: contribution `+0.008619`
- `lag_00__kill_diff_last_3s`: contribution `+0.007404`
- `lag_00__CT_shots_fired_sum`: contribution `+0.006764`

Top utility-only movements:
- `lag_08__CT4__flash_duration`: contribution `+0.014504`
- `lag_12__CT4__flash_duration`: contribution `+0.006396`
- `lag_11__T3__flash_duration`: contribution `+0.004343`

### tick `58677`, seconds `27.50`, LSTM delta `-0.2900`

Top all feature movements:
- `lag_08__CT4__flash_duration`: contribution `-0.014955`
- `lag_08__CT_place_TSIDEUPPER`: contribution `-0.012337`
- `lag_07__T5__flash_duration`: contribution `-0.011378`
- `lag_00__kill_diff_last_3s`: contribution `-0.007404`
- `lag_04__CT4__flash_duration`: contribution `-0.006765`

Top utility-only movements:
- `lag_08__CT4__flash_duration`: contribution `-0.014955`
- `lag_07__T5__flash_duration`: contribution `-0.011378`
- `lag_04__CT4__flash_duration`: contribution `-0.006765`
- `lag_07__T3__flash_duration`: contribution `-0.006547`
- `lag_12__T2__flash_duration`: contribution `-0.006097`

### tick `59765`, seconds `44.50`, LSTM delta `+0.2292`

Top all feature movements:
- `lag_01__T_place_SIDEHALL`: contribution `+0.009966`
- `lag_00__CT_kills_last_3s`: contribution `+0.008619`
- `lag_00__damage_diff_last_5s`: contribution `+0.008360`
- `lag_00__kill_diff_last_3s`: contribution `+0.007404`
- `lag_05__CT1__is_walking`: contribution `+0.005752`

Top utility-only movements:
- `lag_00__T1__molly`: contribution `+0.002683`

### tick `62229`, seconds `83.00`, LSTM delta `+0.1894`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `+0.060038`
- `lag_02__T_velocity_mean`: contribution `+0.019729`
- `lag_04__T_duck_amount_mean`: contribution `+0.006102`
- `lag_10__CT3__is_walking`: contribution `+0.004956`
- `lag_01__CT_duck_amount_mean`: contribution `+0.004788`

Top utility-only movements:
- `lag_09__CT3__smoke`: contribution `+0.003227`
