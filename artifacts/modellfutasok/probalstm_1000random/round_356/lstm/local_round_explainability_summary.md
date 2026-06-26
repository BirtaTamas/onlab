# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-imperial-vs-legacy-bo3-GRvbnL5Q4zT_JzAd-0AXgo/imperial-vs-legacy-m3-mirage.csv`
- round_num: `14`

## Largest probability jumps

- tick `122103`, seconds `137.50`, LSTM `0.7352`, delta `+0.4218`
- tick `118647`, seconds `83.50`, LSTM `0.2272`, delta `-0.4020`
- tick `117783`, seconds `70.00`, LSTM `0.5591`, delta `+0.2771`
- tick `118359`, seconds `79.00`, LSTM `0.3308`, delta `+0.2701`
- tick `117847`, seconds `71.00`, LSTM `0.2519`, delta `-0.2687`
- tick `116759`, seconds `54.00`, LSTM `0.2343`, delta `-0.2172`
- tick `118423`, seconds `80.00`, LSTM `0.5155`, delta `+0.1884`
- tick `122263`, seconds `140.00`, LSTM `0.8845`, delta `+0.1679`
- tick `118007`, seconds `73.50`, LSTM `0.0945`, delta `-0.1615`
- tick `117719`, seconds `69.00`, LSTM `0.1965`, delta `-0.1483`

## Top 15 local ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.013786`, |coef| `0.013786`
- `lag_00__CT_shots_fired_sum`: coefficient `0.011490`, |coef| `0.011490`
- `lag_00__kill_diff_last_3s`: coefficient `0.008847`, |coef| `0.008847`
- `lag_00__CT_kills_last_3s`: coefficient `0.007375`, |coef| `0.007375`
- `lag_00__CT_defusing_count`: coefficient `0.006523`, |coef| `0.006523`
- `lag_00__T1__alive`: coefficient `-0.005502`, |coef| `0.005502`
- `lag_14__T_place_STAIRS`: coefficient `0.005186`, |coef| `0.005186`
- `lag_11__CT5__smoke`: coefficient `0.005142`, |coef| `0.005142`
- `lag_12__CT5__has_defuser`: coefficient `0.004859`, |coef| `0.004859`
- `lag_00__T_closest_enemy_dist`: coefficient `-0.004826`, |coef| `0.004826`
- `lag_00__T1__has_helmet`: coefficient `-0.004734`, |coef| `0.004734`
- `lag_00__CT5__shots_fired`: coefficient `0.004683`, |coef| `0.004683`
- `lag_01__T_flash_alpha_mean`: coefficient `-0.004659`, |coef| `0.004659`
- `lag_00__T_place_BOMBSITEA`: coefficient `-0.004519`, |coef| `0.004519`
- `lag_00__T_macro_A`: coefficient `-0.004519`, |coef| `0.004519`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.013786` (lowers CT win probability)
- `lag_11__CT5__smoke`: coefficient `0.005142` (raises CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `-0.004659` (lowers CT win probability)
- `lag_05__T_flash_alpha_mean`: coefficient `-0.004058` (lowers CT win probability)
- `lag_10__CT5__smoke`: coefficient `0.003078` (raises CT win probability)
- `lag_04__T_flash_alpha_mean`: coefficient `-0.002731` (lowers CT win probability)
- `lag_11__CT5__utility_total`: coefficient `0.002199` (raises CT win probability)
- `lag_12__CT5__smoke`: coefficient `0.001779` (raises CT win probability)
- `lag_00__CT_smokes_last_5s`: coefficient `0.001647` (raises CT win probability)
- `lag_11__CT_smoke_inv`: coefficient `0.001646` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.011490` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.008847` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.007375` (raises CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.006523` (raises CT win probability)
- `lag_00__T1__alive`: coefficient `-0.005502` (lowers CT win probability)
- `lag_14__T_place_STAIRS`: coefficient `0.005186` (raises CT win probability)
- `lag_12__CT5__has_defuser`: coefficient `0.004859` (raises CT win probability)
- `lag_00__T_closest_enemy_dist`: coefficient `-0.004826` (lowers CT win probability)
- `lag_00__T1__has_helmet`: coefficient `-0.004734` (lowers CT win probability)
- `lag_00__CT5__shots_fired`: coefficient `0.004683` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `122103`, seconds `137.50`, LSTM delta `+0.4218`

Top all feature movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.083645`
- `lag_00__CT_shots_fired_sum`: contribution `+0.047896`
- `lag_00__kill_diff_last_3s`: contribution `+0.021294`
- `lag_00__CT_kills_last_3s`: contribution `+0.021293`
- `lag_00__CT5__shots_fired`: contribution `+0.014859`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.083645`
- `lag_11__CT5__smoke`: contribution `+0.011279`

### tick `118647`, seconds `83.50`, LSTM delta `-0.4020`

Top all feature movements:
- `lag_14__T_place_STAIRS`: contribution `-0.099277`
- `lag_00__CT_shots_fired_sum`: contribution `-0.071845`
- `lag_00__kill_diff_last_3s`: contribution `-0.021294`
- `lag_11__T_bomb_zone_count`: contribution `-0.013630`
- `lag_00__T_kills_last_3s`: contribution `-0.011252`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `117783`, seconds `70.00`, LSTM delta `+0.2771`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.055879`
- `lag_00__T_place_STAIRS`: contribution `+0.050971`
- `lag_00__kill_diff_last_3s`: contribution `+0.021294`
- `lag_00__CT_kills_last_3s`: contribution `+0.021293`
- `lag_00__T_shots_fired_sum`: contribution `+0.017622`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `118359`, seconds `79.00`, LSTM delta `+0.2701`

Top all feature movements:
- `lag_05__T_place_STAIRS`: contribution `+0.077627`
- `lag_10__T_place_JUNGLE`: contribution `+0.022048`
- `lag_00__kill_diff_last_3s`: contribution `+0.021294`
- `lag_00__CT_kills_last_3s`: contribution `+0.021293`
- `lag_07__CT5__is_walking`: contribution `+0.009062`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `117847`, seconds `71.00`, LSTM delta `-0.2687`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.159655`
- `lag_00__kill_diff_last_3s`: contribution `-0.021294`
- `lag_00__T1__shots_fired`: contribution `+0.012545`
- `lag_00__T_kills_last_3s`: contribution `-0.011252`
- `lag_04__T_place_JUNGLE`: contribution `-0.009232`

Top utility-only movements:
- `lag_07__T4__flash_duration`: contribution `-0.006871`
