# Local Round Explainability

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-ninjas-in-pyjamas-vs-natus-vincere-bo3-NyGFAYn592Hu0YfljTVG0N/ninjas-in-pyjamas-vs-natus-vincere-m3-train.csv`
- round_num: `3`

## Largest probability jumps

- tick `13624`, seconds `12.00`, LSTM `0.9194`, delta `+0.0585`
- tick `16600`, seconds `58.50`, LSTM `0.9637`, delta `+0.0408`
- tick `13432`, seconds `9.00`, LSTM `0.8957`, delta `+0.0245`
- tick `12952`, seconds `1.50`, LSTM `0.8410`, delta `+0.0243`
- tick `14968`, seconds `33.00`, LSTM `0.9602`, delta `+0.0238`
- tick `13400`, seconds `8.50`, LSTM `0.8713`, delta `+0.0231`
- tick `13240`, seconds `6.00`, LSTM `0.8079`, delta `-0.0189`
- tick `15768`, seconds `45.50`, LSTM `0.9206`, delta `-0.0166`
- tick `16536`, seconds `57.50`, LSTM `0.9183`, delta `-0.0166`
- tick `13080`, seconds `3.50`, LSTM `0.8495`, delta `-0.0161`

## Top 15 local ridge features

- `lag_00__CT_place_ENTRANCE`: coefficient `0.000590`, |coef| `0.000590`
- `lag_03__T1__flash_duration`: coefficient `0.000536`, |coef| `0.000536`
- `lag_00__kill_diff_last_3s`: coefficient `0.000535`, |coef| `0.000535`
- `lag_00__CT_kills_last_3s`: coefficient `0.000530`, |coef| `0.000530`
- `lag_00__CT_shots_fired_sum`: coefficient `0.000518`, |coef| `0.000518`
- `lag_00__CT5__duck_amount`: coefficient `0.000488`, |coef| `0.000488`
- `lag_10__T_place_DUMPSTER`: coefficient `-0.000487`, |coef| `0.000487`
- `lag_14__CT_place_ENTRANCE`: coefficient `0.000478`, |coef| `0.000478`
- `lag_12__T_place_DUMPSTER`: coefficient `0.000473`, |coef| `0.000473`
- `lag_08__CT_place_ENTRANCE`: coefficient `-0.000425`, |coef| `0.000425`
- `lag_00__CT5__is_walking`: coefficient `-0.000384`, |coef| `0.000384`
- `lag_13__CT_place_TUNNELS`: coefficient `0.000377`, |coef| `0.000377`
- `lag_08__bomb_events_last_5s`: coefficient `-0.000375`, |coef| `0.000375`
- `lag_07__CT_place_ENTRANCE`: coefficient `-0.000369`, |coef| `0.000369`
- `lag_00__CT_place_ELECTRICALBOX`: coefficient `0.000367`, |coef| `0.000367`

## Top 10 utility ridge features

- `lag_03__T1__flash_duration`: coefficient `0.000536` (raises CT win probability)
- `lag_05__CT_flash_duration_sum`: coefficient `0.000342` (raises CT win probability)
- `lag_05__CT4__flash_duration`: coefficient `0.000284` (raises CT win probability)
- `lag_05__CT3__flash_duration`: coefficient `0.000271` (raises CT win probability)
- `lag_00__T5__flash_duration`: coefficient `-0.000269` (lowers CT win probability)
- `lag_00__T1__flash_duration`: coefficient `0.000232` (raises CT win probability)
- `lag_04__T1__flash_duration`: coefficient `0.000210` (raises CT win probability)
- `lag_03__T_flash_duration_sum`: coefficient `0.000206` (raises CT win probability)
- `lag_05__CT5__flash_duration`: coefficient `0.000193` (raises CT win probability)
- `lag_03__utility_inv_diff`: coefficient `0.000181` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_ENTRANCE`: coefficient `0.000590` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000535` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000530` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.000518` (raises CT win probability)
- `lag_00__CT5__duck_amount`: coefficient `0.000488` (raises CT win probability)
- `lag_10__T_place_DUMPSTER`: coefficient `-0.000487` (lowers CT win probability)
- `lag_14__CT_place_ENTRANCE`: coefficient `0.000478` (raises CT win probability)
- `lag_12__T_place_DUMPSTER`: coefficient `0.000473` (raises CT win probability)
- `lag_08__CT_place_ENTRANCE`: coefficient `-0.000425` (lowers CT win probability)
- `lag_00__CT5__is_walking`: coefficient `-0.000384` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `13624`, seconds `12.00`, LSTM delta `+0.0585`

Top all feature movements:
- `lag_12__T_place_DUMPSTER`: contribution `+0.008596`
- `lag_10__T_place_DUMPSTER`: contribution `+0.004427`
- `lag_00__CT_place_ELECTRICALBOX`: contribution `+0.004264`
- `lag_05__CT_flash_duration_sum`: contribution `+0.002676`
- `lag_05__CT_flashed_players`: contribution `+0.002083`

Top utility-only movements:
- `lag_05__CT_flash_duration_sum`: contribution `+0.002676`
- `lag_05__CT4__flash_duration`: contribution `+0.001880`
- `lag_05__CT3__flash_duration`: contribution `+0.001724`
- `lag_00__T5__flash_duration`: contribution `+0.001349`
- `lag_05__CT5__flash_duration`: contribution `+0.000846`

### tick `16600`, seconds `58.50`, LSTM delta `+0.0408`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.003958`
- `lag_03__T1__flash_duration`: contribution `+0.003367`
- `lag_08__bomb_events_last_5s`: contribution `+0.001566`
- `lag_00__CT_kills_last_3s`: contribution `+0.001530`
- `lag_01__CT_shots_fired_sum`: contribution `+0.001345`

Top utility-only movements:
- `lag_03__T1__flash_duration`: contribution `+0.003367`
- `lag_03__T_flash_duration_sum`: contribution `+0.000666`

### tick `13432`, seconds `9.00`, LSTM delta `+0.0245`

Top all feature movements:
- `lag_14__CT_place_ENTRANCE`: contribution `+0.008488`
- `lag_10__CT_place_ENTRANCE`: contribution `+0.001909`
- `lag_06__T_place_DUMPSTER`: contribution `-0.001682`
- `lag_03__T_place_DUMPSTER`: contribution `+0.001571`
- `lag_11__CT_place_ENTRANCE`: contribution `+0.001562`

Top utility-only movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.000509`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.000342`

### tick `12952`, seconds `1.50`, LSTM delta `+0.0243`

Top all feature movements:
- `lag_00__CT_place_ENTRANCE`: contribution `+0.005232`
- `lag_03__CT_place_CTSPAWN`: contribution `+0.000964`
- `lag_02__T_velocity_mean`: contribution `+0.000761`
- `lag_02__CT_velocity_mean`: contribution `+0.000733`
- `lag_00__T3__has_bomb`: contribution `+0.000697`

Top utility-only movements:
- `lag_03__utility_inv_diff`: contribution `+0.000562`
- `lag_03__smoke_inv_diff`: contribution `+0.000547`
- `lag_03__molly_inv_diff`: contribution `+0.000356`
- `lag_03__CT_utility_inv`: contribution `+0.000330`
- `lag_03__CT_flash_inv`: contribution `+0.000311`

### tick `14968`, seconds `33.00`, LSTM delta `+0.0238`

Top all feature movements:
- `lag_10__T_place_DUMPSTER`: contribution `+0.004427`
- `lag_14__T_place_DUMPSTER`: contribution `+0.002538`
- `lag_00__CT_kills_last_3s`: contribution `+0.001530`
- `lag_00__CT_shots_fired_sum`: contribution `+0.001439`
- `lag_13__CT_place_LONGDOG`: contribution `+0.001328`

Top utility-only movements:
- No utility movement among the top local contributors.
