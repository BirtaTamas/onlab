# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-lynn-vision-bo3-6KVULP2-Gxo12lI67V9ZfV/chinggis-warriors-vs-lynn-vision-m3-ancient.csv`
- round_num: `25`

## Largest probability jumps

- tick `208177`, seconds `87.00`, LSTM `0.0527`, delta `-0.3945`
- tick `208017`, seconds `84.50`, LSTM `0.4326`, delta `-0.0472`
- tick `207857`, seconds `82.00`, LSTM `0.4975`, delta `+0.0423`
- tick `208209`, seconds `87.50`, LSTM `0.0131`, delta `-0.0396`
- tick `208753`, seconds `96.00`, LSTM `0.0191`, delta `-0.0335`
- tick `207345`, seconds `74.00`, LSTM `0.4740`, delta `-0.0279`
- tick `208433`, seconds `91.00`, LSTM `0.0328`, delta `+0.0273`
- tick `202673`, seconds `1.00`, LSTM `0.5572`, delta `-0.0260`
- tick `202641`, seconds `0.50`, LSTM `0.5832`, delta `+0.0210`
- tick `208113`, seconds `86.00`, LSTM `0.4366`, delta `+0.0204`

## Top 15 local ridge features

- `lag_10__T_place_SIDEHALL`: coefficient `0.003341`, |coef| `0.003341`
- `lag_08__T_bomb_zone_count`: coefficient `-0.002618`, |coef| `0.002618`
- `lag_14__T_place_SIDEHALL`: coefficient `-0.002445`, |coef| `0.002445`
- `lag_07__T5__is_scoped`: coefficient `-0.002028`, |coef| `0.002028`
- `lag_00__T_place_SIDEHALL`: coefficient `-0.002019`, |coef| `0.002019`
- `lag_00__CT2__flash`: coefficient `0.001993`, |coef| `0.001993`
- `lag_00__T_kills_last_3s`: coefficient `-0.001962`, |coef| `0.001962`
- `lag_07__CT_place_TSIDEUPPER`: coefficient `0.001920`, |coef| `0.001920`
- `lag_07__T_place_MAINHALL`: coefficient `-0.001793`, |coef| `0.001793`
- `lag_05__T_place_MAINHALL`: coefficient `0.001751`, |coef| `0.001751`
- `lag_10__T_place_BOMBSITEA`: coefficient `-0.001688`, |coef| `0.001688`
- `lag_10__T_macro_A`: coefficient `-0.001688`, |coef| `0.001688`
- `lag_06__T4__duck_amount`: coefficient `-0.001684`, |coef| `0.001684`
- `lag_13__CT_place_TSIDEUPPER`: coefficient `-0.001682`, |coef| `0.001682`
- `lag_00__kill_diff_last_3s`: coefficient `0.001641`, |coef| `0.001641`

## Top 10 utility ridge features

- `lag_00__CT2__flash`: coefficient `0.001993` (raises CT win probability)
- `lag_00__CT2__utility_total`: coefficient `0.001305` (raises CT win probability)
- `lag_02__T_A_site_active_infernos`: coefficient `0.001246` (raises CT win probability)
- `lag_00__CT_flash_inv`: coefficient `0.001004` (raises CT win probability)
- `lag_02__T_active_infernos`: coefficient `0.000906` (raises CT win probability)
- `lag_00__T3__flash_duration`: coefficient `-0.000879` (lowers CT win probability)
- `lag_02__active_infernos_total`: coefficient `0.000863` (raises CT win probability)
- `lag_11__CT2__flash`: coefficient `-0.000854` (lowers CT win probability)
- `lag_00__CT_utility_inv`: coefficient `0.000761` (raises CT win probability)
- `lag_11__T_A_site_active_smokes`: coefficient `-0.000704` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_10__T_place_SIDEHALL`: coefficient `0.003341` (raises CT win probability)
- `lag_08__T_bomb_zone_count`: coefficient `-0.002618` (lowers CT win probability)
- `lag_14__T_place_SIDEHALL`: coefficient `-0.002445` (lowers CT win probability)
- `lag_07__T5__is_scoped`: coefficient `-0.002028` (lowers CT win probability)
- `lag_00__T_place_SIDEHALL`: coefficient `-0.002019` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001962` (lowers CT win probability)
- `lag_07__CT_place_TSIDEUPPER`: coefficient `0.001920` (raises CT win probability)
- `lag_07__T_place_MAINHALL`: coefficient `-0.001793` (lowers CT win probability)
- `lag_05__T_place_MAINHALL`: coefficient `0.001751` (raises CT win probability)
- `lag_10__T_place_BOMBSITEA`: coefficient `-0.001688` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `208177`, seconds `87.00`, LSTM delta `-0.3945`

Top all feature movements:
- `lag_10__T_place_SIDEHALL`: contribution `-0.043301`
- `lag_14__T_place_SIDEHALL`: contribution `-0.015849`
- `lag_08__T_bomb_zone_count`: contribution `-0.015238`
- `lag_07__CT_place_TSIDEUPPER`: contribution `-0.014436`
- `lag_13__CT_place_TSIDEUPPER`: contribution `-0.012644`

Top utility-only movements:
- `lag_00__CT2__flash`: contribution `-0.007210`

### tick `208017`, seconds `84.50`, LSTM delta `-0.0472`

Top all feature movements:
- `lag_05__T_place_SIDEHALL`: contribution `-0.005579`
- `lag_00__CT_place_TSIDEUPPER`: contribution `-0.004673`
- `lag_00__CT_place_HOUSE`: contribution `+0.004484`
- `lag_05__T_macro_A`: contribution `-0.004412`
- `lag_05__T_place_BOMBSITEA`: contribution `-0.004412`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `207857`, seconds `82.00`, LSTM delta `+0.0423`

Top all feature movements:
- `lag_00__T_place_SIDEHALL`: contribution `+0.026175`
- `lag_01__CT_place_TSIDEUPPER`: contribution `+0.008999`
- `lag_03__CT_place_TSIDEUPPER`: contribution `-0.002285`
- `lag_04__T_place_SIDEHALL`: contribution `+0.001988`
- `lag_08__CT_damage_last_5s`: contribution `+0.001756`

Top utility-only movements:
- `lag_01__CT2__flash`: contribution `+0.000757`
- `lag_15__T5__smoke`: contribution `+0.000717`

### tick `208209`, seconds `87.50`, LSTM delta `-0.0396`

Top all feature movements:
- `lag_15__T_place_SIDEHALL`: contribution `-0.007281`
- `lag_11__T_place_SIDEHALL`: contribution `-0.006255`
- `lag_00__T_shots_fired_sum`: contribution `+0.005921`
- `lag_08__T_place_MAINHALL`: contribution `+0.003849`
- `lag_08__T5__is_scoped`: contribution `-0.003816`

Top utility-only movements:
- `lag_01__CT2__flash`: contribution `-0.001514`

### tick `208753`, seconds `96.00`, LSTM delta `-0.0335`

Top all feature movements:
- `lag_10__T_place_SIDEHALL`: contribution `-0.021650`
- `lag_14__T_place_SIDEHALL`: contribution `-0.015849`
- `lag_00__T_shots_fired_sum`: contribution `-0.006767`
- `lag_00__T3__flash_duration`: contribution `+0.006595`
- `lag_00__T_kills_last_3s`: contribution `-0.006215`

Top utility-only movements:
- `lag_00__T3__flash_duration`: contribution `+0.006595`
- `lag_13__T3__flash_duration`: contribution `+0.004369`
