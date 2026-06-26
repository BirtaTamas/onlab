# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-wildcard-vs-legacy-bo3-NvI4DRplwm0O-zy6YVkFbj/wildcard-vs-legacy-m2-nuke.csv`
- round_num: `8`

## Largest probability jumps

- tick `66681`, seconds `53.50`, LSTM `0.6705`, delta `+0.1483`
- tick `66201`, seconds `46.00`, LSTM `0.5366`, delta `-0.1464`
- tick `69817`, seconds `102.50`, LSTM `0.9324`, delta `+0.1308`
- tick `66009`, seconds `43.00`, LSTM `0.6415`, delta `+0.1020`
- tick `66425`, seconds `49.50`, LSTM `0.6093`, delta `+0.0861`
- tick `65689`, seconds `38.00`, LSTM `0.6555`, delta `-0.0780`
- tick `69049`, seconds `90.50`, LSTM `0.7947`, delta `+0.0685`
- tick `66521`, seconds `51.00`, LSTM `0.5398`, delta `-0.0639`
- tick `66713`, seconds `54.00`, LSTM `0.7201`, delta `+0.0496`
- tick `66041`, seconds `43.50`, LSTM `0.6846`, delta `+0.0432`

## Top 15 local ridge features

- `lag_00__T3__is_scoped`: coefficient `0.002893`, |coef| `0.002893`
- `lag_04__T_place_HUT`: coefficient `0.001853`, |coef| `0.001853`
- `lag_00__kill_diff_last_3s`: coefficient `0.001673`, |coef| `0.001673`
- `lag_00__damage_diff_last_5s`: coefficient `0.001574`, |coef| `0.001574`
- `lag_00__CT_place_HUT`: coefficient `-0.001563`, |coef| `0.001563`
- `lag_06__CT_place_DECON`: coefficient `-0.001557`, |coef| `0.001557`
- `lag_09__CT_place_CRANE`: coefficient `0.001528`, |coef| `0.001528`
- `lag_00__CT_kills_last_3s`: coefficient `0.001460`, |coef| `0.001460`
- `lag_04__CT_shots_fired_sum`: coefficient `-0.001376`, |coef| `0.001376`
- `lag_10__CT_place_DECON`: coefficient `0.001278`, |coef| `0.001278`
- `lag_03__CT_place_VENTS`: coefficient `0.001257`, |coef| `0.001257`
- `lag_00__T_place_SQUEAKY`: coefficient `-0.001231`, |coef| `0.001231`
- `lag_00__CT_damage_last_5s`: coefficient `0.001082`, |coef| `0.001082`
- `lag_04__CT4__shots_fired`: coefficient `-0.001073`, |coef| `0.001073`
- `lag_09__CT_place_VENTS`: coefficient `0.001061`, |coef| `0.001061`

## Top 10 utility ridge features

- `lag_09__CT1__smoke`: coefficient `0.000543` (raises CT win probability)
- `lag_05__CT_A_site_active_infernos`: coefficient `0.000519` (raises CT win probability)
- `lag_05__CT_B_site_active_infernos`: coefficient `0.000504` (raises CT win probability)
- `lag_09__CT4__flash_duration`: coefficient `-0.000500` (lowers CT win probability)
- `lag_15__CT_A_site_active_infernos`: coefficient `-0.000498` (lowers CT win probability)
- `lag_15__CT_B_site_active_infernos`: coefficient `-0.000485` (lowers CT win probability)
- `lag_05__CT4__flash_duration`: coefficient `0.000480` (raises CT win probability)
- `lag_04__CT4__flash_duration`: coefficient `0.000445` (raises CT win probability)
- `lag_02__CT4__flash_duration`: coefficient `-0.000444` (lowers CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `0.000409` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T3__is_scoped`: coefficient `0.002893` (raises CT win probability)
- `lag_04__T_place_HUT`: coefficient `0.001853` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001673` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001574` (raises CT win probability)
- `lag_00__CT_place_HUT`: coefficient `-0.001563` (lowers CT win probability)
- `lag_06__CT_place_DECON`: coefficient `-0.001557` (lowers CT win probability)
- `lag_09__CT_place_CRANE`: coefficient `0.001528` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001460` (raises CT win probability)
- `lag_04__CT_shots_fired_sum`: coefficient `-0.001376` (lowers CT win probability)
- `lag_10__CT_place_DECON`: coefficient `0.001278` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `66681`, seconds `53.50`, LSTM delta `+0.1483`

Top all feature movements:
- `lag_06__CT_place_DECON`: contribution `+0.024760`
- `lag_10__CT_place_DECON`: contribution `+0.020324`
- `lag_15__CT_place_ADMIN`: contribution `+0.004687`
- `lag_00__CT_kills_last_3s`: contribution `+0.004216`
- `lag_00__kill_diff_last_3s`: contribution `+0.004028`

Top utility-only movements:
- `lag_05__CT_A_site_active_infernos`: contribution `+0.001831`
- `lag_15__CT_A_site_active_infernos`: contribution `+0.001756`
- `lag_05__CT_B_site_active_infernos`: contribution `+0.001733`
- `lag_15__CT_B_site_active_infernos`: contribution `+0.001665`

### tick `66201`, seconds `46.00`, LSTM delta `-0.1464`

Top all feature movements:
- `lag_09__CT_place_CRANE`: contribution `-0.025072`
- `lag_03__CT_place_VENTS`: contribution `-0.010550`
- `lag_00__kill_diff_last_3s`: contribution `-0.008056`
- `lag_12__T_place_CONTROL`: contribution `-0.007044`
- `lag_03__T_place_TROPHY`: contribution `-0.005662`

Top utility-only movements:
- `lag_05__CT4__flash_duration`: contribution `-0.003444`

### tick `69817`, seconds `102.50`, LSTM delta `+0.1308`

Top all feature movements:
- `lag_00__T3__is_scoped`: contribution `+0.018558`
- `lag_04__T_place_HUT`: contribution `+0.017268`
- `lag_04__CT_shots_fired_sum`: contribution `+0.011471`
- `lag_00__T_place_SQUEAKY`: contribution `+0.007662`
- `lag_04__CT4__shots_fired`: contribution `+0.006935`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `66009`, seconds `43.00`, LSTM delta `+0.1020`

Top all feature movements:
- `lag_03__CT_place_CRANE`: contribution `+0.013000`
- `lag_15__CT_place_SECRET`: contribution `+0.007896`
- `lag_12__T_place_CONTROL`: contribution `+0.007044`
- `lag_00__damage_diff_last_5s`: contribution `+0.006145`
- `lag_03__T_place_TROPHY`: contribution `+0.005662`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `66425`, seconds `49.50`, LSTM delta `+0.0861`

Top all feature movements:
- `lag_02__CT_place_DECON`: contribution `+0.010728`
- `lag_00__CT_kills_last_3s`: contribution `+0.004216`
- `lag_00__kill_diff_last_3s`: contribution `+0.004028`
- `lag_13__T_place_CONTROL`: contribution `+0.003362`
- `lag_00__CT_shots_fired_sum`: contribution `+0.003316`

Top utility-only movements:
- `lag_12__CT4__flash_duration`: contribution `+0.001877`
