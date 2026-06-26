# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-fluxo-bo3-q_dqfGh9bi4kDnaRAX0wjf/chinggis-warriors-vs-fluxo-m2-mirage.csv`
- round_num: `6`

## Largest probability jumps

- tick `47135`, seconds `50.00`, LSTM `0.1444`, delta `-0.2071`
- tick `47295`, seconds `52.50`, LSTM `0.0504`, delta `-0.1025`
- tick `47487`, seconds `55.50`, LSTM `0.1210`, delta `+0.0809`
- tick `47615`, seconds `57.50`, LSTM `0.0176`, delta `-0.0485`
- tick `44799`, seconds `13.50`, LSTM `0.3972`, delta `+0.0438`
- tick `44319`, seconds `6.00`, LSTM `0.3241`, delta `-0.0412`
- tick `47103`, seconds `49.50`, LSTM `0.3515`, delta `-0.0380`
- tick `47519`, seconds `56.00`, LSTM `0.0857`, delta `-0.0353`
- tick `47039`, seconds `48.50`, LSTM `0.3833`, delta `+0.0325`
- tick `44959`, seconds `16.00`, LSTM `0.4089`, delta `+0.0318`

## Top 15 local ridge features

- `lag_06__CT_flashed_players`: coefficient `0.002543`, |coef| `0.002543`
- `lag_00__T_kills_last_3s`: coefficient `-0.002020`, |coef| `0.002020`
- `lag_08__CT_flashed_players`: coefficient `-0.002016`, |coef| `0.002016`
- `lag_04__T5__is_scoped`: coefficient `-0.001551`, |coef| `0.001551`
- `lag_00__T_damage_last_5s`: coefficient `-0.001480`, |coef| `0.001480`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001437`, |coef| `0.001437`
- `lag_00__CT4__alive`: coefficient `0.001384`, |coef| `0.001384`
- `lag_00__kill_diff_last_3s`: coefficient `0.001380`, |coef| `0.001380`
- `lag_00__CT4__hp`: coefficient `0.001363`, |coef| `0.001363`
- `lag_05__T_place_TRAMP`: coefficient `-0.001334`, |coef| `0.001334`
- `lag_15__CT3__duck_amount`: coefficient `-0.001309`, |coef| `0.001309`
- `lag_13__CT3__duck_amount`: coefficient `-0.001286`, |coef| `0.001286`
- `lag_00__CT4__armor`: coefficient `0.001278`, |coef| `0.001278`
- `lag_00__CT_place_TRUCK`: coefficient `0.001237`, |coef| `0.001237`
- `lag_03__T1__duck_amount`: coefficient `-0.001232`, |coef| `0.001232`

## Top 10 utility ridge features

- `lag_15__T4__smoke`: coefficient `0.001129` (raises CT win probability)
- `lag_04__T_A_site_active_infernos`: coefficient `-0.001069` (lowers CT win probability)
- `lag_08__T2__molly`: coefficient `0.001038` (raises CT win probability)
- `lag_00__CT4__flash`: coefficient `0.000978` (raises CT win probability)
- `lag_02__T3__molly`: coefficient `0.000913` (raises CT win probability)
- `lag_00__T_A_site_active_infernos`: coefficient `-0.000891` (lowers CT win probability)
- `lag_03__T_A_site_active_infernos`: coefficient `0.000860` (raises CT win probability)
- `lag_01__T_A_site_active_infernos`: coefficient `-0.000741` (lowers CT win probability)
- `lag_04__T_active_infernos`: coefficient `-0.000726` (lowers CT win probability)
- `lag_00__T_A_site_active_smokes`: coefficient `-0.000719` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_06__CT_flashed_players`: coefficient `0.002543` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002020` (lowers CT win probability)
- `lag_08__CT_flashed_players`: coefficient `-0.002016` (lowers CT win probability)
- `lag_04__T5__is_scoped`: coefficient `-0.001551` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001480` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001437` (lowers CT win probability)
- `lag_00__CT4__alive`: coefficient `0.001384` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001380` (raises CT win probability)
- `lag_00__CT4__hp`: coefficient `0.001363` (raises CT win probability)
- `lag_05__T_place_TRAMP`: coefficient `-0.001334` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `47135`, seconds `50.00`, LSTM delta `-0.2071`

Top all feature movements:
- `lag_06__CT_flashed_players`: contribution `-0.016705`
- `lag_08__CT_flashed_players`: contribution `-0.013246`
- `lag_04__T5__is_scoped`: contribution `-0.007396`
- `lag_00__T_kills_last_3s`: contribution `-0.006400`
- `lag_00__T_shots_fired_sum`: contribution `-0.005388`

Top utility-only movements:
- `lag_04__T_A_site_active_infernos`: contribution `-0.003181`

### tick `47295`, seconds `52.50`, LSTM delta `-0.1025`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.006400`
- `lag_00__T_shots_fired_sum`: contribution `-0.004310`
- `lag_13__CT_flashed_players`: contribution `-0.003952`
- `lag_15__CT3__duck_amount`: contribution `-0.003786`
- `lag_00__T_damage_last_5s`: contribution `-0.003550`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `47487`, seconds `55.50`, LSTM delta `+0.0809`

Top all feature movements:
- `lag_03__CT_place_STAIRS`: contribution `+0.006646`
- `lag_00__kill_diff_last_3s`: contribution `+0.006641`
- `lag_00__T_kills_last_3s`: contribution `+0.006400`
- `lag_00__CT_place_STAIRS`: contribution `+0.005411`
- `lag_00__T4__duck_amount`: contribution `+0.004160`

Top utility-only movements:
- `lag_01__T_A_site_active_infernos`: contribution `+0.002206`

### tick `47615`, seconds `57.50`, LSTM delta `-0.0485`

Top all feature movements:
- `lag_03__T_place_SCAFFOLDING`: contribution `-0.013755`
- `lag_01__T_place_SCAFFOLDING`: contribution `-0.011538`
- `lag_00__CT_place_TRUCK`: contribution `-0.007977`
- `lag_00__T_kills_last_3s`: contribution `-0.006400`
- `lag_03__CT3__duck_amount`: contribution `+0.003358`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `44799`, seconds `13.50`, LSTM delta `+0.0438`

Top all feature movements:
- `lag_00__CT_place_UNDERPASS`: contribution `+0.004477`
- `lag_11__CT_place_UNDERPASS`: contribution `+0.003498`
- `lag_05__CT4__duck_amount`: contribution `+0.003127`
- `lag_11__CT_place_SHOP`: contribution `+0.002742`
- `lag_12__CT_place_SNIPERSNEST`: contribution `+0.002514`

Top utility-only movements:
- No utility movement among the top local contributors.
