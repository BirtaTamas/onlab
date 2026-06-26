# Local Round Explainability

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-falcons-vs-g2-bo3-Xf_UKx9fB2btv0Vy_VBVUC/falcons-vs-g2-m3-mirage.csv`
- round_num: `7`

## Largest probability jumps

- tick `48561`, seconds `104.50`, LSTM `0.5030`, delta `+0.3947`
- tick `48145`, seconds `98.00`, LSTM `0.0950`, delta `-0.2810`
- tick `46257`, seconds `68.50`, LSTM `0.5430`, delta `-0.2000`
- tick `48657`, seconds `106.00`, LSTM `0.4618`, delta `-0.1115`
- tick `44433`, seconds `40.00`, LSTM `0.6018`, delta `+0.0820`
- tick `48465`, seconds `103.00`, LSTM `0.0989`, delta `+0.0816`
- tick `48625`, seconds `105.50`, LSTM `0.5733`, delta `+0.0804`
- tick `48753`, seconds `107.50`, LSTM `0.5945`, delta `+0.0787`
- tick `44657`, seconds `43.50`, LSTM `0.5948`, delta `+0.0704`
- tick `48689`, seconds `106.50`, LSTM `0.5188`, delta `+0.0570`

## Top 15 local ridge features

- `lag_15__T_place_UNDERPASS`: coefficient `0.003527`, |coef| `0.003527`
- `lag_09__T_place_TRUCK`: coefficient `-0.003403`, |coef| `0.003403`
- `lag_10__T_place_TRUCK`: coefficient `0.003149`, |coef| `0.003149`
- `lag_00__kill_diff_last_3s`: coefficient `0.003106`, |coef| `0.003106`
- `lag_00__T_kills_last_3s`: coefficient `-0.003069`, |coef| `0.003069`
- `lag_00__CT_place_TRUCK`: coefficient `0.002885`, |coef| `0.002885`
- `lag_00__CT_duck_amount_mean`: coefficient `0.002698`, |coef| `0.002698`
- `lag_00__damage_diff_last_5s`: coefficient `0.002550`, |coef| `0.002550`
- `lag_00__T_damage_last_5s`: coefficient `-0.002513`, |coef| `0.002513`
- `lag_00__CT2__duck_amount`: coefficient `0.002391`, |coef| `0.002391`
- `lag_06__CT_place_STAIRS`: coefficient `0.002197`, |coef| `0.002197`
- `lag_03__T4__duck_amount`: coefficient `0.002189`, |coef| `0.002189`
- `lag_15__T2__is_scoped`: coefficient `0.002143`, |coef| `0.002143`
- `lag_03__T_bomb_zone_count`: coefficient `-0.002139`, |coef| `0.002139`
- `lag_00__CT3__duck_amount`: coefficient `0.002091`, |coef| `0.002091`

## Top 10 utility ridge features

- `lag_00__CT2__flash`: coefficient `0.001442` (raises CT win probability)
- `lag_03__T3__flash`: coefficient `-0.001092` (lowers CT win probability)
- `lag_00__T4__flash`: coefficient `-0.001068` (lowers CT win probability)
- `lag_04__CT3__molly`: coefficient `0.001002` (raises CT win probability)
- `lag_09__CT3__smoke`: coefficient `0.000858` (raises CT win probability)
- `lag_00__flash_inv_diff`: coefficient `0.000781` (raises CT win probability)
- `lag_00__CT2__utility_total`: coefficient `0.000726` (raises CT win probability)
- `lag_01__CT2__flash`: coefficient `0.000683` (raises CT win probability)
- `lag_00__CT4__flash`: coefficient `0.000665` (raises CT win probability)
- `lag_13__CT2__flash`: coefficient `-0.000641` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_15__T_place_UNDERPASS`: coefficient `0.003527` (raises CT win probability)
- `lag_09__T_place_TRUCK`: coefficient `-0.003403` (lowers CT win probability)
- `lag_10__T_place_TRUCK`: coefficient `0.003149` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003106` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003069` (lowers CT win probability)
- `lag_00__CT_place_TRUCK`: coefficient `0.002885` (raises CT win probability)
- `lag_00__CT_duck_amount_mean`: coefficient `0.002698` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002550` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002513` (lowers CT win probability)
- `lag_00__CT2__duck_amount`: coefficient `0.002391` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `48561`, seconds `104.50`, LSTM delta `+0.3947`

Top all feature movements:
- `lag_09__T_place_TRUCK`: contribution `+0.059092`
- `lag_10__T_place_TRUCK`: contribution `+0.054685`
- `lag_09__T2__is_scoped`: contribution `+0.014186`
- `lag_03__T_bomb_zone_count`: contribution `+0.012450`
- `lag_02__T2__is_scoped`: contribution `+0.010831`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `48145`, seconds `98.00`, LSTM delta `-0.2810`

Top all feature movements:
- `lag_00__CT_place_TRUCK`: contribution `-0.018612`
- `lag_15__T_place_UNDERPASS`: contribution `-0.013815`
- `lag_00__T_kills_last_3s`: contribution `-0.009724`
- `lag_00__CT2__duck_amount`: contribution `-0.009110`
- `lag_03__T4__duck_amount`: contribution `-0.008094`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `46257`, seconds `68.50`, LSTM delta `-0.2000`

Top all feature movements:
- `lag_06__CT_place_STAIRS`: contribution `-0.017099`
- `lag_04__CT_place_JUNGLE`: contribution `-0.010197`
- `lag_00__T_kills_last_3s`: contribution `-0.009724`
- `lag_12__CT_place_SNIPERSNEST`: contribution `-0.009585`
- `lag_00__T2__is_scoped`: contribution `-0.008808`

Top utility-only movements:
- `lag_04__CT3__molly`: contribution `-0.002473`

### tick `48657`, seconds `106.00`, LSTM delta `-0.1115`

Top all feature movements:
- `lag_12__T_place_TRUCK`: contribution `-0.031474`
- `lag_13__T_place_TRUCK`: contribution `-0.018911`
- `lag_05__T2__is_scoped`: contribution `-0.011368`
- `lag_01__T2__is_scoped`: contribution `-0.010124`
- `lag_12__T2__is_scoped`: contribution `+0.008859`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `44433`, seconds `40.00`, LSTM delta `+0.0820`

Top all feature movements:
- `lag_00__CT2__duck_amount`: contribution `-0.008663`
- `lag_00__CT3__duck_amount`: contribution `+0.007782`
- `lag_00__kill_diff_last_3s`: contribution `+0.007475`
- `lag_00__CT3__is_walking`: contribution `+0.004618`
- `lag_00__T_shots_fired_sum`: contribution `+0.004321`

Top utility-only movements:
- No utility movement among the top local contributors.
