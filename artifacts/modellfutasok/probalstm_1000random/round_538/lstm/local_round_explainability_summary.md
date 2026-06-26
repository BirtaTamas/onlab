# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-spirit-vs-heroic-bo3-NSK1XoxyhXK-sIe0204dzp/spirit-vs-heroic-m3-mirage.csv`
- round_num: `4`

## Largest probability jumps

- tick `26113`, seconds `59.50`, LSTM `0.1782`, delta `-0.2411`
- tick `24449`, seconds `33.50`, LSTM `0.4905`, delta `+0.1052`
- tick `24129`, seconds `28.50`, LSTM `0.4781`, delta `-0.0835`
- tick `26177`, seconds `60.50`, LSTM `0.0351`, delta `-0.0814`
- tick `26145`, seconds `60.00`, LSTM `0.1165`, delta `-0.0617`
- tick `23265`, seconds `15.00`, LSTM `0.4103`, delta `+0.0616`
- tick `25505`, seconds `50.00`, LSTM `0.4793`, delta `-0.0603`
- tick `26817`, seconds `70.50`, LSTM `0.1109`, delta `+0.0602`
- tick `26945`, seconds `72.50`, LSTM `0.0174`, delta `-0.0574`
- tick `25537`, seconds `50.50`, LSTM `0.4271`, delta `-0.0522`

## Top 15 local ridge features

- `lag_00__CT_place_SCAFFOLDING`: coefficient `-0.002565`, |coef| `0.002565`
- `lag_15__CT_place_LADDER`: coefficient `-0.001974`, |coef| `0.001974`
- `lag_00__CT_place_LADDER`: coefficient `0.001901`, |coef| `0.001901`
- `lag_11__CT_place_STAIRS`: coefficient `0.001609`, |coef| `0.001609`
- `lag_11__CT_place_TRUCK`: coefficient `0.001413`, |coef| `0.001413`
- `lag_01__CT_place_LADDER`: coefficient `0.001398`, |coef| `0.001398`
- `lag_00__T1__is_scoped`: coefficient `0.001307`, |coef| `0.001307`
- `lag_00__T_kills_last_3s`: coefficient `-0.001175`, |coef| `0.001175`
- `lag_00__CT_place_SHOP`: coefficient `-0.001053`, |coef| `0.001053`
- `lag_13__T_place_UNDERPASS`: coefficient `0.001050`, |coef| `0.001050`
- `lag_09__T_flashed_players`: coefficient `-0.001039`, |coef| `0.001039`
- `lag_03__CT_place_SCAFFOLDING`: coefficient `-0.001017`, |coef| `0.001017`
- `lag_00__kill_diff_last_3s`: coefficient `0.000992`, |coef| `0.000992`
- `lag_11__T_place_CATWALK`: coefficient `-0.000972`, |coef| `0.000972`
- `lag_15__CT_place_CATWALK`: coefficient `0.000928`, |coef| `0.000928`

## Top 10 utility ridge features

- `lag_11__T_A_site_active_infernos`: coefficient `0.000612` (raises CT win probability)
- `lag_05__T1__flash_duration`: coefficient `0.000591` (raises CT win probability)
- `lag_12__CT2__smoke`: coefficient `0.000497` (raises CT win probability)
- `lag_09__T1__flash_duration`: coefficient `-0.000495` (lowers CT win probability)
- `lag_08__CT_B_site_active_smokes`: coefficient `-0.000485` (lowers CT win probability)
- `lag_09__CT_B_site_active_smokes`: coefficient `-0.000479` (lowers CT win probability)
- `lag_11__T_active_infernos`: coefficient `0.000471` (raises CT win probability)
- `lag_14__T4__smoke`: coefficient `0.000457` (raises CT win probability)
- `lag_13__T_B_site_active_infernos`: coefficient `-0.000443` (lowers CT win probability)
- `lag_00__CT_flash_alpha_mean`: coefficient `-0.000436` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_SCAFFOLDING`: coefficient `-0.002565` (lowers CT win probability)
- `lag_15__CT_place_LADDER`: coefficient `-0.001974` (lowers CT win probability)
- `lag_00__CT_place_LADDER`: coefficient `0.001901` (raises CT win probability)
- `lag_11__CT_place_STAIRS`: coefficient `0.001609` (raises CT win probability)
- `lag_11__CT_place_TRUCK`: coefficient `0.001413` (raises CT win probability)
- `lag_01__CT_place_LADDER`: coefficient `0.001398` (raises CT win probability)
- `lag_00__T1__is_scoped`: coefficient `0.001307` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001175` (lowers CT win probability)
- `lag_00__CT_place_SHOP`: coefficient `-0.001053` (lowers CT win probability)
- `lag_13__T_place_UNDERPASS`: coefficient `0.001050` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `26113`, seconds `59.50`, LSTM delta `-0.2411`

Top all feature movements:
- `lag_15__CT_place_LADDER`: contribution `-0.020522`
- `lag_00__CT_place_LADDER`: contribution `-0.019766`
- `lag_11__CT_place_STAIRS`: contribution `-0.012526`
- `lag_11__CT_place_TRUCK`: contribution `-0.009116`
- `lag_09__T_flashed_players`: contribution `-0.008018`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `24449`, seconds `33.50`, LSTM delta `+0.1052`

Top all feature movements:
- `lag_00__CT_place_SCAFFOLDING`: contribution `+0.053523`
- `lag_10__CT_place_SCAFFOLDING`: contribution `+0.014707`
- `lag_09__T1__is_scoped`: contribution `+0.004840`
- `lag_00__CT_place_PALACEINTERIOR`: contribution `+0.003436`
- `lag_01__T1__is_scoped`: contribution `+0.003188`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `24129`, seconds `28.50`, LSTM delta `-0.0835`

Top all feature movements:
- `lag_00__CT_place_SCAFFOLDING`: contribution `-0.053523`
- `lag_10__T1__is_scoped`: contribution `+0.005201`
- `lag_00__CT_place_PALACEINTERIOR`: contribution `-0.003436`
- `lag_13__T1__is_scoped`: contribution `-0.002972`
- `lag_04__T3__duck_amount`: contribution `+0.002631`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `26177`, seconds `60.50`, LSTM delta `-0.0814`

Top all feature movements:
- `lag_02__CT_place_LADDER`: contribution `-0.009253`
- `lag_13__CT_place_TRUCK`: contribution `-0.005896`
- `lag_11__T1__is_scoped`: contribution `-0.004858`
- `lag_09__T_flashed_players`: contribution `+0.004009`
- `lag_11__T_flashed_players`: contribution `-0.003994`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `26145`, seconds `60.00`, LSTM delta `-0.0617`

Top all feature movements:
- `lag_01__CT_place_LADDER`: contribution `-0.014536`
- `lag_12__CT_place_TRUCK`: contribution `-0.005456`
- `lag_12__CT_place_STAIRS`: contribution `-0.005366`
- `lag_10__T1__is_scoped`: contribution `-0.005201`
- `lag_10__T_flashed_players`: contribution `-0.004375`

Top utility-only movements:
- No utility movement among the top local contributors.
