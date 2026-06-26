# Local Round Explainability

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-ninjas-in-pyjamas-vs-natus-vincere-bo3-NyGFAYn592Hu0YfljTVG0N/ninjas-in-pyjamas-vs-natus-vincere-m2-inferno.csv`
- round_num: `6`

## Largest probability jumps

- tick `51727`, seconds `28.00`, LSTM `0.0902`, delta `-0.3202`
- tick `51983`, seconds `32.00`, LSTM `0.0352`, delta `-0.1297`
- tick `51951`, seconds `31.50`, LSTM `0.1648`, delta `+0.1028`
- tick `51663`, seconds `27.00`, LSTM `0.4033`, delta `-0.0712`
- tick `51087`, seconds `18.00`, LSTM `0.4323`, delta `-0.0581`
- tick `50575`, seconds `10.00`, LSTM `0.4388`, delta `+0.0437`
- tick `51151`, seconds `19.00`, LSTM `0.4603`, delta `+0.0408`
- tick `50415`, seconds `7.50`, LSTM `0.3953`, delta `-0.0382`
- tick `50735`, seconds `12.50`, LSTM `0.4682`, delta `-0.0333`
- tick `51631`, seconds `26.50`, LSTM `0.4744`, delta `-0.0320`

## Top 15 local ridge features

- `lag_11__T_flash_duration_sum`: coefficient `-0.002401`, |coef| `0.002401`
- `lag_11__T4__flash_duration`: coefficient `-0.002022`, |coef| `0.002022`
- `lag_11__T_flashed_players`: coefficient `-0.002009`, |coef| `0.002009`
- `lag_11__T1__flash_duration`: coefficient `-0.001883`, |coef| `0.001883`
- `lag_11__T5__flash_duration`: coefficient `-0.001843`, |coef| `0.001843`
- `lag_00__CT_place_BALCONY`: coefficient `-0.001733`, |coef| `0.001733`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001474`, |coef| `0.001474`
- `lag_00__T_kills_last_3s`: coefficient `-0.001454`, |coef| `0.001454`
- `lag_09__CT_flashed_players`: coefficient `0.001400`, |coef| `0.001400`
- `lag_00__CT_place_BANANA`: coefficient `0.001303`, |coef| `0.001303`
- `lag_00__kill_diff_last_3s`: coefficient `0.001274`, |coef| `0.001274`
- `lag_00__CT_place_APARTMENTS`: coefficient `0.001260`, |coef| `0.001260`
- `lag_12__CT_A_site_active_infernos`: coefficient `0.001223`, |coef| `0.001223`
- `lag_02__CT3__duck_amount`: coefficient `0.001215`, |coef| `0.001215`
- `lag_00__CT4__alive`: coefficient `0.001208`, |coef| `0.001208`

## Top 10 utility ridge features

- `lag_11__T_flash_duration_sum`: coefficient `-0.002401` (lowers CT win probability)
- `lag_11__T4__flash_duration`: coefficient `-0.002022` (lowers CT win probability)
- `lag_11__T1__flash_duration`: coefficient `-0.001883` (lowers CT win probability)
- `lag_11__T5__flash_duration`: coefficient `-0.001843` (lowers CT win probability)
- `lag_12__CT_A_site_active_infernos`: coefficient `0.001223` (raises CT win probability)
- `lag_07__T1__flash_duration`: coefficient `0.001147` (raises CT win probability)
- `lag_09__CT1__flash_duration`: coefficient `0.001052` (raises CT win probability)
- `lag_03__T5__flash_duration`: coefficient `0.001023` (raises CT win probability)
- `lag_09__CT2__flash_duration`: coefficient `0.001002` (raises CT win probability)
- `lag_09__CT_flash_duration_sum`: coefficient `0.000938` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_11__T_flashed_players`: coefficient `-0.002009` (lowers CT win probability)
- `lag_00__CT_place_BALCONY`: coefficient `-0.001733` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001474` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001454` (lowers CT win probability)
- `lag_09__CT_flashed_players`: coefficient `0.001400` (raises CT win probability)
- `lag_00__CT_place_BANANA`: coefficient `0.001303` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001274` (raises CT win probability)
- `lag_00__CT_place_APARTMENTS`: coefficient `0.001260` (raises CT win probability)
- `lag_02__CT3__duck_amount`: coefficient `0.001215` (raises CT win probability)
- `lag_00__CT4__alive`: coefficient `0.001208` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `51727`, seconds `28.00`, LSTM delta `-0.3202`

Top all feature movements:
- `lag_11__T_flash_duration_sum`: contribution `-0.016791`
- `lag_11__T4__flash_duration`: contribution `-0.011697`
- `lag_11__T_flashed_players`: contribution `-0.011632`
- `lag_11__T5__flash_duration`: contribution `-0.011388`
- `lag_00__CT_place_BALCONY`: contribution `-0.011123`

Top utility-only movements:
- `lag_11__T_flash_duration_sum`: contribution `-0.016791`
- `lag_11__T4__flash_duration`: contribution `-0.011697`
- `lag_11__T5__flash_duration`: contribution `-0.011388`
- `lag_11__T1__flash_duration`: contribution `-0.009642`
- `lag_12__CT_A_site_active_infernos`: contribution `-0.004316`

### tick `51983`, seconds `32.00`, LSTM delta `-0.1297`

Top all feature movements:
- `lag_00__CT_place_BALCONY`: contribution `+0.011123`
- `lag_01__T_place_BALCONY`: contribution `-0.010193`
- `lag_11__T5__flash_duration`: contribution `+0.006062`
- `lag_09__T_shots_fired_sum`: contribution `-0.005326`
- `lag_02__CT3__is_scoped`: contribution `+0.004790`

Top utility-only movements:
- `lag_11__T5__flash_duration`: contribution `+0.006062`
- `lag_11__T_flash_duration_sum`: contribution `+0.003208`
- `lag_05__T4__flash_duration`: contribution `-0.002749`

### tick `51951`, seconds `31.50`, LSTM delta `+0.1028`

Top all feature movements:
- `lag_00__T_place_BALCONY`: contribution `-0.005467`
- `lag_08__T_shots_fired_sum`: contribution `+0.005073`
- `lag_04__T3__shots_fired`: contribution `+0.004840`
- `lag_00__CT_place_BANANA`: contribution `-0.003856`
- `lag_01__CT3__is_scoped`: contribution `+0.003575`

Top utility-only movements:
- `lag_04__T4__flash_duration`: contribution `+0.002333`
- `lag_10__T5__flash_duration`: contribution `+0.002321`

### tick `51663`, seconds `27.00`, LSTM delta `-0.0712`

Top all feature movements:
- `lag_15__T2__flash_duration`: contribution `-0.006111`
- `lag_00__T_shots_fired_sum`: contribution `-0.005527`
- `lag_09__T_flash_duration_sum`: contribution `-0.004696`
- `lag_09__T5__flash_duration`: contribution `-0.004376`
- `lag_09__T_flashed_players`: contribution `-0.004237`

Top utility-only movements:
- `lag_15__T2__flash_duration`: contribution `-0.006111`
- `lag_09__T_flash_duration_sum`: contribution `-0.004696`
- `lag_09__T5__flash_duration`: contribution `-0.004376`
- `lag_09__T4__flash_duration`: contribution `-0.003819`
- `lag_09__T1__flash_duration`: contribution `-0.002743`

### tick `51087`, seconds `18.00`, LSTM delta `-0.0581`

Top all feature movements:
- `lag_11__T_flash_duration_sum`: contribution `-0.007323`
- `lag_00__T_place_BALCONY`: contribution `-0.005467`
- `lag_15__CT_place_BALCONY`: contribution `-0.004199`
- `lag_11__T_flashed_players`: contribution `-0.003877`
- `lag_04__T3__duck_amount`: contribution `-0.003375`

Top utility-only movements:
- `lag_11__T_flash_duration_sum`: contribution `-0.007323`
- `lag_00__CT1__molly`: contribution `-0.001071`
- `lag_00__CT2__molly`: contribution `-0.001051`
