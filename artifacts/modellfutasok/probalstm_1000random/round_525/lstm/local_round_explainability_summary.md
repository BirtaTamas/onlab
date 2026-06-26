# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-faze-vs-virtuspro-bo3-YDlVsCnS6YPgcr85bBYoPq/faze-vs-virtus-pro-m2-mirage.csv`
- round_num: `6`

## Largest probability jumps

- tick `42988`, seconds `51.50`, LSTM `0.5861`, delta `+0.4758`
- tick `43660`, seconds `62.00`, LSTM `0.8623`, delta `+0.2651`
- tick `42028`, seconds `36.50`, LSTM `0.8172`, delta `+0.2364`
- tick `42252`, seconds `40.00`, LSTM `0.5970`, delta `-0.2166`
- tick `42412`, seconds `42.50`, LSTM `0.3408`, delta `-0.1805`
- tick `41580`, seconds `29.50`, LSTM `0.5558`, delta `+0.1500`
- tick `42860`, seconds `49.50`, LSTM `0.1801`, delta `-0.1176`
- tick `43180`, seconds `54.50`, LSTM `0.5420`, delta `-0.0865`
- tick `41388`, seconds `26.50`, LSTM `0.3976`, delta `-0.0712`
- tick `42700`, seconds `47.00`, LSTM `0.2412`, delta `-0.0631`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.006158`, |coef| `0.006158`
- `lag_00__kill_diff_last_3s`: coefficient `0.006042`, |coef| `0.006042`
- `lag_04__CT_place_UNDERPASS`: coefficient `0.005343`, |coef| `0.005343`
- `lag_04__T_bomb_zone_count`: coefficient `-0.004855`, |coef| `0.004855`
- `lag_00__damage_diff_last_5s`: coefficient `0.004176`, |coef| `0.004176`
- `lag_00__CT_damage_last_5s`: coefficient `0.003919`, |coef| `0.003919`
- `lag_07__CT3__is_scoped`: coefficient `-0.003656`, |coef| `0.003656`
- `lag_01__T_utility_damage_last_5s`: coefficient `-0.003349`, |coef| `0.003349`
- `lag_04__CT_place_CATWALK`: coefficient `-0.003198`, |coef| `0.003198`
- `lag_00__T_macro_B`: coefficient `-0.003100`, |coef| `0.003100`
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.003100`, |coef| `0.003100`
- `lag_11__T_utility_damage_last_5s`: coefficient `0.003099`, |coef| `0.003099`
- `lag_12__T_kills_last_3s`: coefficient `-0.002967`, |coef| `0.002967`
- `lag_12__T2__duck_amount`: coefficient `0.002850`, |coef| `0.002850`
- `lag_11__CT3__duck_amount`: coefficient `-0.002837`, |coef| `0.002837`

## Top 10 utility ridge features

- `lag_01__T_utility_damage_last_5s`: coefficient `-0.003349` (lowers CT win probability)
- `lag_11__T_utility_damage_last_5s`: coefficient `0.003099` (raises CT win probability)
- `lag_07__T_B_site_active_smokes`: coefficient `-0.002032` (lowers CT win probability)
- `lag_01__utility_damage_diff_last_5s`: coefficient `0.002012` (raises CT win probability)
- `lag_14__T4__molly`: coefficient `-0.001987` (lowers CT win probability)
- `lag_11__utility_damage_diff_last_5s`: coefficient `-0.001900` (lowers CT win probability)
- `lag_06__T_B_site_active_smokes`: coefficient `-0.001838` (lowers CT win probability)
- `lag_13__T_B_site_active_infernos`: coefficient `0.001597` (raises CT win probability)
- `lag_08__T_utility_damage_last_5s`: coefficient `-0.001529` (lowers CT win probability)
- `lag_07__T_active_smokes`: coefficient `-0.001470` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.006158` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.006042` (raises CT win probability)
- `lag_04__CT_place_UNDERPASS`: coefficient `0.005343` (raises CT win probability)
- `lag_04__T_bomb_zone_count`: coefficient `-0.004855` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.004176` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.003919` (raises CT win probability)
- `lag_07__CT3__is_scoped`: coefficient `-0.003656` (lowers CT win probability)
- `lag_04__CT_place_CATWALK`: coefficient `-0.003198` (lowers CT win probability)
- `lag_00__T_macro_B`: coefficient `-0.003100` (lowers CT win probability)
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.003100` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `42988`, seconds `51.50`, LSTM delta `+0.4758`

Top all feature movements:
- `lag_04__CT_place_UNDERPASS`: contribution `+0.030985`
- `lag_04__T_bomb_zone_count`: contribution `+0.028263`
- `lag_00__CT_kills_last_3s`: contribution `+0.017780`
- `lag_00__kill_diff_last_3s`: contribution `+0.014543`
- `lag_01__T_utility_damage_last_5s`: contribution `+0.014342`

Top utility-only movements:
- `lag_01__T_utility_damage_last_5s`: contribution `+0.014342`
- `lag_11__T_utility_damage_last_5s`: contribution `+0.013273`

### tick `43660`, seconds `62.00`, LSTM delta `+0.2651`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.017780`
- `lag_07__CT3__is_scoped`: contribution `+0.016626`
- `lag_00__kill_diff_last_3s`: contribution `+0.014543`
- `lag_01__T_duck_amount_mean`: contribution `+0.013404`
- `lag_11__CT3__duck_amount`: contribution `+0.010557`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `42028`, seconds `36.50`, LSTM delta `+0.2364`

Top all feature movements:
- `lag_04__CT_place_UNDERPASS`: contribution `+0.030985`
- `lag_05__T_place_TRUCK`: contribution `+0.029601`
- `lag_00__CT_kills_last_3s`: contribution `+0.017780`
- `lag_07__T_place_TRUCK`: contribution `+0.016333`
- `lag_00__kill_diff_last_3s`: contribution `+0.014543`

Top utility-only movements:
- `lag_11__CT2__flash_duration`: contribution `+0.006541`
- `lag_08__T_utility_damage_last_5s`: contribution `+0.005240`
- `lag_04__CT4__flash_duration`: contribution `+0.005232`

### tick `42252`, seconds `40.00`, LSTM delta `-0.2166`

Top all feature movements:
- `lag_12__T_place_TRUCK`: contribution `-0.046617`
- `lag_14__T_place_TRUCK`: contribution `-0.024657`
- `lag_00__kill_diff_last_3s`: contribution `-0.014543`
- `lag_01__CT_kills_last_3s`: contribution `-0.007000`
- `lag_00__CT4__flash_duration`: contribution `-0.005980`

Top utility-only movements:
- `lag_00__CT4__flash_duration`: contribution `-0.005980`
- `lag_11__CT4__flash_duration`: contribution `-0.003449`

### tick `42412`, seconds `42.50`, LSTM delta `-0.1805`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.014543`
- `lag_03__T_bomb_zone_count`: contribution `-0.014187`
- `lag_05__CT_place_UNDERPASS`: contribution `-0.010618`
- `lag_01__T3__duck_amount`: contribution `-0.010468`
- `lag_15__T3__duck_amount`: contribution `-0.007658`

Top utility-only movements:
- `lag_03__CT2__flash_duration`: contribution `-0.006917`
- `lag_05__CT4__flash_duration`: contribution `-0.006875`
