# Local Round Explainability

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-vitality-vs-g2-bo5-ieXHvClzA7f_aJ_85fPFqK/vitality-vs-g2-m1-dust2.csv`
- round_num: `3`

## Largest probability jumps

- tick `20692`, seconds `96.00`, LSTM `0.4034`, delta `-0.3759`
- tick `20436`, seconds `92.00`, LSTM `0.5497`, delta `+0.3482`
- tick `20116`, seconds `87.00`, LSTM `0.2650`, delta `-0.3366`
- tick `19956`, seconds `84.50`, LSTM `0.7597`, delta `+0.2969`
- tick `21332`, seconds `106.00`, LSTM `0.1427`, delta `-0.2954`
- tick `19860`, seconds `83.00`, LSTM `0.5151`, delta `-0.2075`
- tick `19828`, seconds `82.50`, LSTM `0.7226`, delta `+0.1647`
- tick `20148`, seconds `87.50`, LSTM `0.1646`, delta `-0.1003`
- tick `20468`, seconds `92.50`, LSTM `0.6388`, delta `+0.0891`
- tick `20756`, seconds `97.00`, LSTM `0.3324`, delta `-0.0761`

## Top 15 local ridge features

- `lag_04__CT_mollies_last_5s`: coefficient `0.004229`, |coef| `0.004229`
- `lag_00__CT_shots_fired_sum`: coefficient `0.003853`, |coef| `0.003853`
- `lag_00__kill_diff_last_3s`: coefficient `0.003500`, |coef| `0.003500`
- `lag_12__T4__flash_duration`: coefficient `-0.003002`, |coef| `0.003002`
- `lag_02__CT_mollies_last_5s`: coefficient `0.002975`, |coef| `0.002975`
- `lag_00__CT_place_HOLE`: coefficient `0.002746`, |coef| `0.002746`
- `lag_04__CT_place_OUTSIDELONG`: coefficient `-0.002645`, |coef| `0.002645`
- `lag_01__T_bomb_zone_count`: coefficient `-0.002503`, |coef| `0.002503`
- `lag_00__T_kills_last_3s`: coefficient `-0.002406`, |coef| `0.002406`
- `lag_10__T_bomb_zone_count`: coefficient `-0.002387`, |coef| `0.002387`
- `lag_00__damage_diff_last_5s`: coefficient `0.002154`, |coef| `0.002154`
- `lag_00__CT_kills_last_3s`: coefficient `0.002005`, |coef| `0.002005`
- `lag_01__CT5__duck_amount`: coefficient `0.002005`, |coef| `0.002005`
- `lag_09__CT_place_OUTSIDELONG`: coefficient `0.001983`, |coef| `0.001983`
- `lag_12__T_flash_duration_sum`: coefficient `-0.001978`, |coef| `0.001978`

## Top 10 utility ridge features

- `lag_04__CT_mollies_last_5s`: coefficient `0.004229` (raises CT win probability)
- `lag_12__T4__flash_duration`: coefficient `-0.003002` (lowers CT win probability)
- `lag_02__CT_mollies_last_5s`: coefficient `0.002975` (raises CT win probability)
- `lag_12__T_flash_duration_sum`: coefficient `-0.001978` (lowers CT win probability)
- `lag_07__T_flash_duration_sum`: coefficient `0.001895` (raises CT win probability)
- `lag_04__T1__flash_duration`: coefficient `-0.001818` (lowers CT win probability)
- `lag_07__T1__flash_duration`: coefficient `0.001810` (raises CT win probability)
- `lag_03__T4__flash_duration`: coefficient `0.001792` (raises CT win probability)
- `lag_02__T4__flash_duration`: coefficient `0.001785` (raises CT win probability)
- `lag_10__T_B_site_active_infernos`: coefficient `-0.001784` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.003853` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003500` (raises CT win probability)
- `lag_00__CT_place_HOLE`: coefficient `0.002746` (raises CT win probability)
- `lag_04__CT_place_OUTSIDELONG`: coefficient `-0.002645` (lowers CT win probability)
- `lag_01__T_bomb_zone_count`: coefficient `-0.002503` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002406` (lowers CT win probability)
- `lag_10__T_bomb_zone_count`: coefficient `-0.002387` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002154` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002005` (raises CT win probability)
- `lag_01__CT5__duck_amount`: coefficient `0.002005` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `20692`, seconds `96.00`, LSTM delta `-0.3759`

Top all feature movements:
- `lag_02__CT_mollies_last_5s`: contribution `-0.098615`
- `lag_12__CT_mollies_last_5s`: contribution `-0.046157`
- `lag_13__CT_place_HOLE`: contribution `-0.020450`
- `lag_12__CT_place_OUTSIDELONG`: contribution `-0.018212`
- `lag_02__CT_flashes_last_5s`: contribution `-0.010905`

Top utility-only movements:
- `lag_02__CT_mollies_last_5s`: contribution `-0.098615`
- `lag_12__CT_mollies_last_5s`: contribution `-0.046157`
- `lag_02__CT_flashes_last_5s`: contribution `-0.010905`
- `lag_12__CT_flashes_last_5s`: contribution `-0.005010`

### tick `20436`, seconds `92.00`, LSTM delta `+0.3482`

Top all feature movements:
- `lag_04__CT_mollies_last_5s`: contribution `+0.140204`
- `lag_04__CT_place_OUTSIDELONG`: contribution `+0.026828`
- `lag_09__CT_place_OUTSIDELONG`: contribution `+0.020115`
- `lag_12__T4__flash_duration`: contribution `+0.018001`
- `lag_08__CT_place_HOLE`: contribution `-0.017174`

Top utility-only movements:
- `lag_04__CT_mollies_last_5s`: contribution `+0.140204`
- `lag_12__T4__flash_duration`: contribution `+0.018001`
- `lag_04__CT_flashes_last_5s`: contribution `+0.015509`
- `lag_09__T5__flash_duration`: contribution `+0.012952`
- `lag_12__T_flash_duration_sum`: contribution `+0.004821`

### tick `20116`, seconds `87.00`, LSTM delta `-0.3366`

Top all feature movements:
- `lag_12__T4__flash_duration`: contribution `-0.022939`
- `lag_13__CT_place_HOLE`: contribution `-0.020450`
- `lag_12__T_flash_duration_sum`: contribution `-0.019369`
- `lag_03__T2__is_scoped`: contribution `-0.015855`
- `lag_08__CT_shots_fired_sum`: contribution `-0.015466`

Top utility-only movements:
- `lag_12__T4__flash_duration`: contribution `-0.022939`
- `lag_12__T_flash_duration_sum`: contribution `-0.019369`
- `lag_02__T4__flash_duration`: contribution `-0.010699`
- `lag_09__T1__flash_duration`: contribution `-0.010116`
- `lag_12__T1__flash_duration`: contribution `-0.010058`

### tick `19956`, seconds `84.50`, LSTM delta `+0.2969`

Top all feature movements:
- `lag_03__CT_shots_fired_sum`: contribution `+0.020197`
- `lag_07__T_flash_duration_sum`: contribution `+0.018562`
- `lag_08__CT_place_HOLE`: contribution `+0.017174`
- `lag_04__T1__flash_duration`: contribution `+0.014163`
- `lag_07__T1__flash_duration`: contribution `+0.014096`

Top utility-only movements:
- `lag_07__T_flash_duration_sum`: contribution `+0.018562`
- `lag_04__T1__flash_duration`: contribution `+0.014163`
- `lag_07__T1__flash_duration`: contribution `+0.014096`
- `lag_07__T4__flash_duration`: contribution `+0.010430`
- `lag_07__T5__flash_duration`: contribution `+0.009095`

### tick `21332`, seconds `106.00`, LSTM delta `-0.2954`

Top all feature movements:
- `lag_00__CT_place_HOLE`: contribution `-0.030656`
- `lag_08__CT_place_HOLE`: contribution `-0.017174`
- `lag_01__T_bomb_zone_count`: contribution `-0.014570`
- `lag_10__T_bomb_zone_count`: contribution `-0.013893`
- `lag_00__CT_shots_fired_sum`: contribution `-0.010709`

Top utility-only movements:
- `lag_10__T_B_site_active_infernos`: contribution `-0.005043`
- `lag_05__CT_B_site_active_infernos`: contribution `-0.004737`
