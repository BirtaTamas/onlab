# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m1-inferno.csv`
- round_num: `3`

## Largest probability jumps

- tick `21060`, seconds `30.50`, LSTM `0.0234`, delta `-0.0259`
- tick `19140`, seconds `0.50`, LSTM `0.0203`, delta `-0.0226`
- tick `21188`, seconds `32.50`, LSTM `0.0268`, delta `-0.0107`
- tick `20484`, seconds `21.50`, LSTM `0.0404`, delta `-0.0106`
- tick `21220`, seconds `33.00`, LSTM `0.0168`, delta `-0.0099`
- tick `21124`, seconds `31.50`, LSTM `0.0332`, delta `+0.0099`
- tick `20644`, seconds `24.00`, LSTM `0.0456`, delta `+0.0093`
- tick `20132`, seconds `16.00`, LSTM `0.0372`, delta `+0.0072`
- tick `21508`, seconds `37.50`, LSTM `0.0031`, delta `-0.0069`
- tick `19492`, seconds `6.00`, LSTM `0.0218`, delta `-0.0055`

## Top 15 local ridge features

- `lag_00__CT_smokes_last_5s`: coefficient `-0.000370`, |coef| `0.000370`
- `lag_04__T_flashed_players`: coefficient `-0.000211`, |coef| `0.000211`
- `lag_11__CT_smokes_last_5s`: coefficient `-0.000208`, |coef| `0.000208`
- `lag_04__T1__flash_duration`: coefficient `-0.000207`, |coef| `0.000207`
- `lag_04__T_flash_duration_sum`: coefficient `-0.000187`, |coef| `0.000187`
- `lag_04__T5__flash_duration`: coefficient `-0.000162`, |coef| `0.000162`
- `lag_00__T_velocity_mean`: coefficient `-0.000148`, |coef| `0.000148`
- `lag_13__CT2__is_walking`: coefficient `0.000145`, |coef| `0.000145`
- `lag_00__T_kills_last_3s`: coefficient `-0.000142`, |coef| `0.000142`
- `lag_05__CT2__is_walking`: coefficient `0.000140`, |coef| `0.000140`
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000138`, |coef| `0.000138`
- `lag_14__T_place_UNDERPASS`: coefficient `-0.000138`, |coef| `0.000138`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000138`, |coef| `0.000138`
- `lag_01__CT_smokes_last_5s`: coefficient `-0.000138`, |coef| `0.000138`
- `lag_11__T4__is_walking`: coefficient `0.000136`, |coef| `0.000136`

## Top 10 utility ridge features

- `lag_00__CT_smokes_last_5s`: coefficient `-0.000370` (lowers CT win probability)
- `lag_11__CT_smokes_last_5s`: coefficient `-0.000208` (lowers CT win probability)
- `lag_04__T1__flash_duration`: coefficient `-0.000207` (lowers CT win probability)
- `lag_04__T_flash_duration_sum`: coefficient `-0.000187` (lowers CT win probability)
- `lag_04__T5__flash_duration`: coefficient `-0.000162` (lowers CT win probability)
- `lag_01__CT_smokes_last_5s`: coefficient `-0.000138` (lowers CT win probability)
- `lag_01__T2__flash_duration`: coefficient `-0.000135` (lowers CT win probability)
- `lag_07__T1__smoke`: coefficient `0.000092` (raises CT win probability)
- `lag_10__CT1__smoke`: coefficient `0.000085` (raises CT win probability)
- `lag_05__CT1__flash`: coefficient `0.000084` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_04__T_flashed_players`: coefficient `-0.000211` (lowers CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000148` (lowers CT win probability)
- `lag_13__CT2__is_walking`: coefficient `0.000145` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.000142` (lowers CT win probability)
- `lag_05__CT2__is_walking`: coefficient `0.000140` (raises CT win probability)
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000138` (lowers CT win probability)
- `lag_14__T_place_UNDERPASS`: coefficient `-0.000138` (lowers CT win probability)
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000138` (lowers CT win probability)
- `lag_11__T4__is_walking`: coefficient `0.000136` (raises CT win probability)
- `lag_13__T4__is_walking`: coefficient `-0.000134` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `21060`, seconds `30.50`, LSTM delta `-0.0259`

Top all feature movements:
- `lag_04__T_flashed_players`: contribution `-0.001630`
- `lag_04__T1__flash_duration`: contribution `-0.001400`
- `lag_04__T_flash_duration_sum`: contribution `-0.001253`
- `lag_04__T5__flash_duration`: contribution `-0.000875`
- `lag_01__T2__flash_duration`: contribution `-0.000680`

Top utility-only movements:
- `lag_04__T1__flash_duration`: contribution `-0.001400`
- `lag_04__T_flash_duration_sum`: contribution `-0.001253`
- `lag_04__T5__flash_duration`: contribution `-0.000875`
- `lag_01__T2__flash_duration`: contribution `-0.000680`

### tick `19140`, seconds `0.50`, LSTM delta `-0.0226`

Top all feature movements:
- `lag_00__CT_smokes_last_5s`: contribution `-0.006391`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.000568`
- `lag_01__T_closest_enemy_dist`: contribution `-0.000554`
- `lag_01__CT_place_CTSPAWN`: contribution `-0.000543`
- `lag_01__T_place_TSPAWN`: contribution `-0.000538`

Top utility-only movements:
- `lag_00__CT_smokes_last_5s`: contribution `-0.006391`
- `lag_00__CT5__smoke`: contribution `-0.000175`
- `lag_01__CT1__flash`: contribution `-0.000174`
- `lag_00__CT4__smoke`: contribution `-0.000173`
- `lag_01__T_smoke_inv`: contribution `-0.000157`

### tick `21188`, seconds `32.50`, LSTM delta `-0.0107`

Top all feature movements:
- `lag_00__T5__shots_fired`: contribution `-0.001455`
- `lag_04__T_shots_fired_sum`: contribution `-0.000456`
- `lag_01__T5__shots_fired`: contribution `+0.000456`
- `lag_08__T1__flash_duration`: contribution `-0.000404`
- `lag_08__T_flash_duration_sum`: contribution `-0.000390`

Top utility-only movements:
- `lag_08__T1__flash_duration`: contribution `-0.000404`
- `lag_08__T_flash_duration_sum`: contribution `-0.000390`
- `lag_05__T2__flash_duration`: contribution `-0.000355`
- `lag_08__T5__flash_duration`: contribution `-0.000275`

### tick `20484`, seconds `21.50`, LSTM delta `-0.0106`

Top all feature movements:
- `lag_04__CT_place_LIBRARY`: contribution `-0.000817`
- `lag_01__bomb_events_last_5s`: contribution `-0.000478`
- `lag_09__T_A_site_active_infernos`: contribution `-0.000456`
- `lag_08__CT_place_LIBRARY`: contribution `-0.000389`
- `lag_11__T4__is_walking`: contribution `-0.000313`

Top utility-only movements:
- `lag_09__T_A_site_active_infernos`: contribution `-0.000456`
- `lag_09__T_active_infernos`: contribution `-0.000263`

### tick `21220`, seconds `33.00`, LSTM delta `-0.0099`

Top all feature movements:
- `lag_01__T5__shots_fired`: contribution `-0.001746`
- `lag_01__T_shots_fired_sum`: contribution `-0.000978`
- `lag_05__T_shots_fired_sum`: contribution `-0.000685`
- `lag_00__T_kills_last_3s`: contribution `-0.000449`
- `lag_09__T1__flash_duration`: contribution `-0.000318`

Top utility-only movements:
- `lag_09__T1__flash_duration`: contribution `-0.000318`
- `lag_09__T_flash_duration_sum`: contribution `-0.000313`
- `lag_09__T5__flash_duration`: contribution `-0.000215`
