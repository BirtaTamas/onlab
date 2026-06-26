# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-faze-vs-heroic-dust2-PtQF8ASKD1754yZQHk6148/faze-vs-heroic-dust2.csv`
- round_num: `12`

## Largest probability jumps

- tick `90702`, seconds `21.00`, LSTM `0.7007`, delta `+0.1197`
- tick `90830`, seconds `23.00`, LSTM `0.8848`, delta `+0.0759`
- tick `91278`, seconds `30.00`, LSTM `0.9409`, delta `+0.0654`
- tick `90734`, seconds `21.50`, LSTM `0.7486`, delta `+0.0479`
- tick `90094`, seconds `11.50`, LSTM `0.5617`, delta `-0.0474`
- tick `90766`, seconds `22.00`, LSTM `0.7804`, delta `+0.0318`
- tick `91022`, seconds `26.00`, LSTM `0.8742`, delta `-0.0302`
- tick `90798`, seconds `22.50`, LSTM `0.8089`, delta `+0.0285`
- tick `90126`, seconds `12.00`, LSTM `0.5845`, delta `+0.0228`
- tick `91886`, seconds `39.50`, LSTM `0.9821`, delta `+0.0225`

## Top 15 local ridge features

- `lag_01__CT_place_LOWERTUNNEL`: coefficient `-0.001140`, |coef| `0.001140`
- `lag_15__CT_place_HOLE`: coefficient `-0.000992`, |coef| `0.000992`
- `lag_00__CT_kills_last_3s`: coefficient `0.000941`, |coef| `0.000941`
- `lag_10__T4__flash_duration`: coefficient `-0.000812`, |coef| `0.000812`
- `lag_00__CT_damage_last_5s`: coefficient `0.000810`, |coef| `0.000810`
- `lag_03__CT3__flash_duration`: coefficient `-0.000798`, |coef| `0.000798`
- `lag_12__T3__flash_duration`: coefficient `-0.000776`, |coef| `0.000776`
- `lag_02__CT_place_LOWERTUNNEL`: coefficient `-0.000747`, |coef| `0.000747`
- `lag_00__CT_place_LOWERTUNNEL`: coefficient `-0.000689`, |coef| `0.000689`
- `lag_00__kill_diff_last_3s`: coefficient `0.000673`, |coef| `0.000673`
- `lag_04__CT3__flash_duration`: coefficient `-0.000613`, |coef| `0.000613`
- `lag_00__CT5__is_scoped`: coefficient `-0.000607`, |coef| `0.000607`
- `lag_00__damage_diff_last_5s`: coefficient `0.000604`, |coef| `0.000604`
- `lag_03__CT_place_LOWERTUNNEL`: coefficient `-0.000571`, |coef| `0.000571`
- `lag_15__CT_place_BDOORS`: coefficient `0.000544`, |coef| `0.000544`

## Top 10 utility ridge features

- `lag_10__T4__flash_duration`: coefficient `-0.000812` (lowers CT win probability)
- `lag_03__CT3__flash_duration`: coefficient `-0.000798` (lowers CT win probability)
- `lag_12__T3__flash_duration`: coefficient `-0.000776` (lowers CT win probability)
- `lag_04__CT3__flash_duration`: coefficient `-0.000613` (lowers CT win probability)
- `lag_06__CT3__flash_duration`: coefficient `-0.000523` (lowers CT win probability)
- `lag_11__T4__flash_duration`: coefficient `-0.000519` (lowers CT win probability)
- `lag_13__T3__flash_duration`: coefficient `-0.000508` (lowers CT win probability)
- `lag_00__T2__smoke`: coefficient `-0.000474` (lowers CT win probability)
- `lag_12__T_flash_duration_sum`: coefficient `-0.000448` (lowers CT win probability)
- `lag_14__CT_B_site_active_infernos`: coefficient `-0.000435` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_LOWERTUNNEL`: coefficient `-0.001140` (lowers CT win probability)
- `lag_15__CT_place_HOLE`: coefficient `-0.000992` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000941` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.000810` (raises CT win probability)
- `lag_02__CT_place_LOWERTUNNEL`: coefficient `-0.000747` (lowers CT win probability)
- `lag_00__CT_place_LOWERTUNNEL`: coefficient `-0.000689` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000673` (raises CT win probability)
- `lag_00__CT5__is_scoped`: coefficient `-0.000607` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000604` (raises CT win probability)
- `lag_03__CT_place_LOWERTUNNEL`: coefficient `-0.000571` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `90702`, seconds `21.00`, LSTM delta `+0.1197`

Top all feature movements:
- `lag_15__CT_place_HOLE`: contribution `+0.011076`
- `lag_01__CT_place_LOWERTUNNEL`: contribution `+0.008377`
- `lag_12__T3__flash_duration`: contribution `+0.005102`
- `lag_10__T4__flash_duration`: contribution `+0.004256`
- `lag_03__CT3__flash_duration`: contribution `+0.004084`

Top utility-only movements:
- `lag_12__T3__flash_duration`: contribution `+0.005102`
- `lag_10__T4__flash_duration`: contribution `+0.004256`
- `lag_03__CT3__flash_duration`: contribution `+0.004084`
- `lag_14__CT_B_site_active_infernos`: contribution `+0.001495`

### tick `90830`, seconds `23.00`, LSTM delta `+0.0759`

Top all feature movements:
- `lag_01__CT_place_LOWERTUNNEL`: contribution `+0.008377`
- `lag_02__CT_place_LOWERTUNNEL`: contribution `-0.005492`
- `lag_02__CT_place_HOLE`: contribution `+0.004195`
- `lag_00__CT_kills_last_3s`: contribution `+0.002716`
- `lag_05__CT_place_LOWERTUNNEL`: contribution `+0.002672`

Top utility-only movements:
- `lag_07__CT3__flash_duration`: contribution `+0.001637`
- `lag_14__T4__flash_duration`: contribution `+0.001539`

### tick `91278`, seconds `30.00`, LSTM delta `+0.0654`

Top all feature movements:
- `lag_02__CT_place_LOWERTUNNEL`: contribution `+0.005492`
- `lag_00__CT_place_HOLE`: contribution `-0.003693`
- `lag_00__CT_damage_last_5s`: contribution `+0.003145`
- `lag_00__CT_kills_last_3s`: contribution `+0.002716`
- `lag_15__CT_place_LOWERTUNNEL`: contribution `-0.002672`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `90734`, seconds `21.50`, LSTM delta `+0.0479`

Top all feature movements:
- `lag_02__CT_place_LOWERTUNNEL`: contribution `+0.005492`
- `lag_13__T3__flash_duration`: contribution `+0.003342`
- `lag_04__CT3__flash_duration`: contribution `+0.003136`
- `lag_11__T4__flash_duration`: contribution `+0.002723`
- `lag_12__CT4__duck_amount`: contribution `+0.001596`

Top utility-only movements:
- `lag_13__T3__flash_duration`: contribution `+0.003342`
- `lag_04__CT3__flash_duration`: contribution `+0.003136`
- `lag_11__T4__flash_duration`: contribution `+0.002723`
- `lag_15__CT_B_site_active_infernos`: contribution `+0.001168`
- `lag_13__T_flash_duration_sum`: contribution `+0.000846`

### tick `90094`, seconds `11.50`, LSTM delta `-0.0474`

Top all feature movements:
- `lag_00__CT_place_LOWERTUNNEL`: contribution `-0.005063`
- `lag_02__CT_place_HOLE`: contribution `+0.004195`
- `lag_06__CT3__flash_duration`: contribution `-0.002257`
- `lag_00__CT5__is_scoped`: contribution `-0.002169`
- `lag_05__T3__flash_duration`: contribution `-0.002074`

Top utility-only movements:
- `lag_06__CT3__flash_duration`: contribution `-0.002257`
- `lag_05__T3__flash_duration`: contribution `-0.002074`
- `lag_06__T4__flash_duration`: contribution `-0.002016`
