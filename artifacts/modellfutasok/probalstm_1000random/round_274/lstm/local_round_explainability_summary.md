# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-natus-vincere-vs-tyloo-bo3-u9zlDGjnIy0eSohnO5P-Xx/natus-vincere-vs-tyloo-m3-ancient.csv`
- round_num: `16`

## Largest probability jumps

- tick `115511`, seconds `30.50`, LSTM `0.0204`, delta `-0.0496`
- tick `113591`, seconds `0.50`, LSTM `0.0186`, delta `-0.0289`
- tick `115479`, seconds `30.00`, LSTM `0.0700`, delta `+0.0144`
- tick `115447`, seconds `29.50`, LSTM `0.0556`, delta `+0.0127`
- tick `115543`, seconds `31.00`, LSTM `0.0099`, delta `-0.0105`
- tick `115031`, seconds `23.00`, LSTM `0.0309`, delta `-0.0077`
- tick `114903`, seconds `21.00`, LSTM `0.0342`, delta `+0.0052`
- tick `113623`, seconds `1.00`, LSTM `0.0135`, delta `-0.0051`
- tick `114999`, seconds `22.50`, LSTM `0.0386`, delta `+0.0049`
- tick `115319`, seconds `27.50`, LSTM `0.0427`, delta `+0.0047`

## Top 15 local ridge features

- `lag_00__CT_place_TSIDELOWER`: coefficient `0.000505`, |coef| `0.000505`
- `lag_00__CT_flashed_players`: coefficient `0.000384`, |coef| `0.000384`
- `lag_01__CT_place_ALLEY`: coefficient `-0.000304`, |coef| `0.000304`
- `lag_01__CT_place_TSIDELOWER`: coefficient `-0.000261`, |coef| `0.000261`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000237`, |coef| `0.000237`
- `lag_05__CT5__flash_duration`: coefficient `0.000229`, |coef| `0.000229`
- `lag_02__CT_flashed_players`: coefficient `-0.000220`, |coef| `0.000220`
- `lag_00__CT4__duck_amount`: coefficient `0.000211`, |coef| `0.000211`
- `lag_15__CT5__flash_duration`: coefficient `-0.000211`, |coef| `0.000211`
- `lag_01__CT_place_RAMP`: coefficient `0.000201`, |coef| `0.000201`
- `lag_02__T4__flash_duration`: coefficient `-0.000187`, |coef| `0.000187`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000186`, |coef| `0.000186`
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000185`, |coef| `0.000185`
- `lag_00__CT_place_ALLEY`: coefficient `0.000179`, |coef| `0.000179`
- `lag_01__smoke_inv_diff`: coefficient `0.000175`, |coef| `0.000175`

## Top 10 utility ridge features

- `lag_05__CT5__flash_duration`: coefficient `0.000229` (raises CT win probability)
- `lag_15__CT5__flash_duration`: coefficient `-0.000211` (lowers CT win probability)
- `lag_02__T4__flash_duration`: coefficient `-0.000187` (lowers CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000175` (raises CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000158` (raises CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000157` (raises CT win probability)
- `lag_01__T_B_site_active_smokes`: coefficient `0.000153` (raises CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `0.000152` (raises CT win probability)
- `lag_13__CT_flashes_last_5s`: coefficient `0.000139` (raises CT win probability)
- `lag_01__T_active_smokes`: coefficient `0.000130` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_TSIDELOWER`: coefficient `0.000505` (raises CT win probability)
- `lag_00__CT_flashed_players`: coefficient `0.000384` (raises CT win probability)
- `lag_01__CT_place_ALLEY`: coefficient `-0.000304` (lowers CT win probability)
- `lag_01__CT_place_TSIDELOWER`: coefficient `-0.000261` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000237` (lowers CT win probability)
- `lag_02__CT_flashed_players`: coefficient `-0.000220` (lowers CT win probability)
- `lag_00__CT4__duck_amount`: coefficient `0.000211` (raises CT win probability)
- `lag_01__CT_place_RAMP`: coefficient `0.000201` (raises CT win probability)
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000186` (lowers CT win probability)
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000185` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `115511`, seconds `30.50`, LSTM delta `-0.0496`

Top all feature movements:
- `lag_00__CT_place_TSIDELOWER`: contribution `-0.006854`
- `lag_01__CT_place_TSIDELOWER`: contribution `-0.003552`
- `lag_00__CT_flashed_players`: contribution `-0.003362`
- `lag_02__CT_flashed_players`: contribution `-0.001923`
- `lag_05__CT5__flash_duration`: contribution `-0.001345`

Top utility-only movements:
- `lag_05__CT5__flash_duration`: contribution `-0.001345`
- `lag_15__CT5__flash_duration`: contribution `-0.001238`
- `lag_02__T4__flash_duration`: contribution `-0.000972`
- `lag_00__CT3__flash_duration`: contribution `-0.000494`
- `lag_01__T_B_site_active_smokes`: contribution `-0.000464`

### tick `113591`, seconds `0.50`, LSTM delta `-0.0289`

Top all feature movements:
- `lag_01__CT_place_ALLEY`: contribution `-0.002232`
- `lag_01__T_place_TSPAWN`: contribution `-0.001050`
- `lag_00__CT4__duck_amount`: contribution `-0.000777`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.000682`
- `lag_01__T_closest_enemy_dist`: contribution `-0.000679`

Top utility-only movements:
- `lag_01__smoke_inv_diff`: contribution `-0.000557`
- `lag_01__utility_inv_diff`: contribution `-0.000451`
- `lag_01__molly_inv_diff`: contribution `-0.000438`
- `lag_01__T_molly_inv`: contribution `-0.000287`
- `lag_01__T_smoke_inv`: contribution `-0.000279`

### tick `115479`, seconds `30.00`, LSTM delta `+0.0144`

Top all feature movements:
- `lag_00__CT_place_TSIDELOWER`: contribution `+0.006854`
- `lag_01__CT_flashed_players`: contribution `+0.001392`
- `lag_01__CT_place_RAMP`: contribution `+0.000601`
- `lag_01__T_flashed_players`: contribution `+0.000471`
- `lag_00__CT_place_RAMP`: contribution `-0.000409`

Top utility-only movements:
- `lag_01__CT3__flash_duration`: contribution `+0.000221`
- `lag_01__CT_flash_duration_sum`: contribution `+0.000165`
- `lag_04__CT5__flash_duration`: contribution `-0.000142`

### tick `115447`, seconds `29.50`, LSTM delta `+0.0127`

Top all feature movements:
- `lag_00__CT_flashed_players`: contribution `+0.003362`
- `lag_00__T_flashed_players`: contribution `+0.000931`
- `lag_00__CT3__flash_duration`: contribution `+0.000494`
- `lag_13__T_place_RAMP`: contribution `+0.000438`
- `lag_13__T_place_TSIDELOWER`: contribution `+0.000418`

Top utility-only movements:
- `lag_00__CT3__flash_duration`: contribution `+0.000494`
- `lag_00__CT_flash_duration_sum`: contribution `+0.000338`
- `lag_13__CT5__flash_duration`: contribution `+0.000314`
- `lag_00__T4__flash_duration`: contribution `+0.000232`
- `lag_03__CT5__flash_duration`: contribution `+0.000158`

### tick `115543`, seconds `31.00`, LSTM delta `-0.0105`

Top all feature movements:
- `lag_01__CT_place_TSIDELOWER`: contribution `+0.003552`
- `lag_01__CT_flashed_players`: contribution `-0.001392`
- `lag_02__CT_place_TSIDELOWER`: contribution `-0.001240`
- `lag_03__CT_flashed_players`: contribution `-0.000915`
- `lag_01__T_shots_fired_sum`: contribution `-0.000784`

Top utility-only movements:
- `lag_06__CT5__flash_duration`: contribution `-0.000613`
- `lag_03__T4__flash_duration`: contribution `-0.000494`
- `lag_02__T_B_site_active_smokes`: contribution `-0.000260`
