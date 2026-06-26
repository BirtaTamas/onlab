# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-natus-vincere-vs-spirit-bo3-cW0x-KCT4cbPLaZUAvb08Z/natus-vincere-vs-spirit-m2-ancient.csv`
- round_num: `13`

## Largest probability jumps

- tick `107609`, seconds `47.50`, LSTM `0.7884`, delta `+0.2281`
- tick `107513`, seconds `46.00`, LSTM `0.5262`, delta `+0.2265`
- tick `107673`, seconds `48.50`, LSTM `0.8815`, delta `+0.0961`
- tick `107481`, seconds `45.50`, LSTM `0.2998`, delta `-0.0835`
- tick `107801`, seconds `50.50`, LSTM `0.9556`, delta `+0.0525`
- tick `107449`, seconds `45.00`, LSTM `0.3833`, delta `-0.0415`
- tick `107353`, seconds `43.50`, LSTM `0.4560`, delta `-0.0382`
- tick `107385`, seconds `44.00`, LSTM `0.4183`, delta `-0.0378`
- tick `107577`, seconds `47.00`, LSTM `0.5603`, delta `+0.0328`
- tick `105465`, seconds `14.00`, LSTM `0.4400`, delta `-0.0301`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.003392`, |coef| `0.003392`
- `lag_00__kill_diff_last_3s`: coefficient `0.002828`, |coef| `0.002828`
- `lag_02__T2__duck_amount`: coefficient `-0.002735`, |coef| `0.002735`
- `lag_00__T_macro_A`: coefficient `-0.002632`, |coef| `0.002632`
- `lag_00__T_place_BOMBSITEA`: coefficient `-0.002632`, |coef| `0.002632`
- `lag_00__damage_diff_last_5s`: coefficient `0.002536`, |coef| `0.002536`
- `lag_00__CT_damage_last_5s`: coefficient `0.002521`, |coef| `0.002521`
- `lag_03__CT_place_SIDEHALL`: coefficient `-0.002470`, |coef| `0.002470`
- `lag_03__CT_place_HOUSE`: coefficient `-0.002345`, |coef| `0.002345`
- `lag_06__CT_place_HOUSE`: coefficient `-0.002057`, |coef| `0.002057`
- `lag_06__CT_place_SIDEHALL`: coefficient `-0.002049`, |coef| `0.002049`
- `lag_03__T3__duck_amount`: coefficient `-0.002022`, |coef| `0.002022`
- `lag_01__T_place_MAINHALL`: coefficient `-0.002020`, |coef| `0.002020`
- `lag_02__CT_flashed_players`: coefficient `-0.002006`, |coef| `0.002006`
- `lag_07__CT_flashed_players`: coefficient `0.001800`, |coef| `0.001800`

## Top 10 utility ridge features

- `lag_05__T_A_site_active_infernos`: coefficient `0.001577` (raises CT win probability)
- `lag_05__T_B_site_active_infernos`: coefficient `0.001496` (raises CT win probability)
- `lag_08__T_A_site_active_infernos`: coefficient `0.001465` (raises CT win probability)
- `lag_08__T_B_site_active_infernos`: coefficient `0.001389` (raises CT win probability)
- `lag_09__T4__molly`: coefficient `-0.001282` (lowers CT win probability)
- `lag_12__T4__molly`: coefficient `-0.001223` (lowers CT win probability)
- `lag_00__CT1__flash`: coefficient `-0.001127` (lowers CT win probability)
- `lag_05__T_active_infernos`: coefficient `0.001092` (raises CT win probability)
- `lag_01__CT2__flash_duration`: coefficient `-0.001015` (lowers CT win probability)
- `lag_08__T_active_infernos`: coefficient `0.001009` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.003392` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002828` (raises CT win probability)
- `lag_02__T2__duck_amount`: coefficient `-0.002735` (lowers CT win probability)
- `lag_00__T_macro_A`: coefficient `-0.002632` (lowers CT win probability)
- `lag_00__T_place_BOMBSITEA`: coefficient `-0.002632` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002536` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002521` (raises CT win probability)
- `lag_03__CT_place_SIDEHALL`: coefficient `-0.002470` (lowers CT win probability)
- `lag_03__CT_place_HOUSE`: coefficient `-0.002345` (lowers CT win probability)
- `lag_06__CT_place_HOUSE`: coefficient `-0.002057` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `107609`, seconds `47.50`, LSTM delta `+0.2281`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.009794`
- `lag_06__CT_place_SIDEHALL`: contribution `+0.008765`
- `lag_07__CT_flashed_players`: contribution `+0.007884`
- `lag_01__T_place_MAINHALL`: contribution `+0.007293`
- `lag_06__CT_place_HOUSE`: contribution `+0.007267`

Top utility-only movements:
- `lag_08__T_A_site_active_infernos`: contribution `+0.004361`
- `lag_08__T_B_site_active_infernos`: contribution `+0.003927`

### tick `107513`, seconds `46.00`, LSTM delta `+0.2265`

Top all feature movements:
- `lag_03__CT_place_SIDEHALL`: contribution `+0.010564`
- `lag_02__T2__duck_amount`: contribution `+0.010457`
- `lag_00__CT_kills_last_3s`: contribution `+0.009794`
- `lag_03__CT_place_HOUSE`: contribution `+0.008284`
- `lag_01__T_place_MAINHALL`: contribution `+0.007293`

Top utility-only movements:
- `lag_05__T_A_site_active_infernos`: contribution `+0.004695`
- `lag_05__T_B_site_active_infernos`: contribution `+0.004231`

### tick `107673`, seconds `48.50`, LSTM delta `+0.0961`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.009794`
- `lag_06__CT_place_HOUSE`: contribution `+0.007267`
- `lag_00__kill_diff_last_3s`: contribution `+0.006807`
- `lag_00__damage_diff_last_5s`: contribution `+0.005721`
- `lag_00__CT_damage_last_5s`: contribution `+0.005494`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `107481`, seconds `45.50`, LSTM delta `-0.0835`

Top all feature movements:
- `lag_02__T2__duck_amount`: contribution `-0.010457`
- `lag_01__T_place_MAINHALL`: contribution `+0.007293`
- `lag_03__CT_flashed_players`: contribution `-0.006241`
- `lag_04__CT_place_HOUSE`: contribution `-0.006204`
- `lag_03__T3__duck_amount`: contribution `-0.005964`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `107801`, seconds `50.50`, LSTM delta `+0.0525`

Top all feature movements:
- `lag_03__CT_flashed_players`: contribution `-0.006241`
- `lag_04__T_place_MAINHALL`: contribution `-0.005827`
- `lag_03__CT3__flash_duration`: contribution `+0.005584`
- `lag_00__damage_diff_last_5s`: contribution `+0.005493`
- `lag_00__CT_damage_last_5s`: contribution `+0.004615`

Top utility-only movements:
- `lag_03__CT3__flash_duration`: contribution `+0.005584`
- `lag_03__T5__flash_duration`: contribution `+0.003391`
- `lag_03__T1__flash_duration`: contribution `+0.003149`
- `lag_00__T_A_site_active_infernos`: contribution `+0.002636`
- `lag_00__T_B_site_active_infernos`: contribution `+0.002378`
