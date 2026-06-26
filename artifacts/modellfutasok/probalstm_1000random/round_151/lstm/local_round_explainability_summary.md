# Local Round Explainability

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-falcons-vs-nrg-bo3-WMQcRUwgyUmu57EEkX9f3P/falcons-vs-nrg-m1-train.csv`
- round_num: `4`

## Largest probability jumps

- tick `25317`, seconds `78.00`, LSTM `0.2511`, delta `-0.2479`
- tick `22373`, seconds `32.00`, LSTM `0.2813`, delta `-0.2191`
- tick `22693`, seconds `37.00`, LSTM `0.6210`, delta `+0.2068`
- tick `22405`, seconds `32.50`, LSTM `0.4537`, delta `+0.1725`
- tick `22341`, seconds `31.50`, LSTM `0.5004`, delta `-0.1196`
- tick `22789`, seconds `38.50`, LSTM `0.4769`, delta `-0.1118`
- tick `26053`, seconds `89.50`, LSTM `0.1375`, delta `+0.1071`
- tick `22277`, seconds `30.50`, LSTM `0.6018`, delta `+0.0825`
- tick `22885`, seconds `40.00`, LSTM `0.3833`, delta `-0.0606`
- tick `22725`, seconds `37.50`, LSTM `0.5616`, delta `-0.0594`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004374`, |coef| `0.004374`
- `lag_00__T_kills_last_3s`: coefficient `-0.003482`, |coef| `0.003482`
- `lag_13__T_place_CTSPAWN`: coefficient `-0.003420`, |coef| `0.003420`
- `lag_00__damage_diff_last_5s`: coefficient `0.003294`, |coef| `0.003294`
- `lag_00__CT_place_TUNNELS`: coefficient `0.003200`, |coef| `0.003200`
- `lag_00__CT1__alive`: coefficient `0.002901`, |coef| `0.002901`
- `lag_00__CT1__is_walking`: coefficient `0.002763`, |coef| `0.002763`
- `lag_03__CT5__is_scoped`: coefficient `-0.002726`, |coef| `0.002726`
- `lag_00__T_damage_last_5s`: coefficient `-0.002702`, |coef| `0.002702`
- `lag_10__CT5__is_scoped`: coefficient `0.002641`, |coef| `0.002641`
- `lag_00__CT1__armor`: coefficient `0.002627`, |coef| `0.002627`
- `lag_00__CT_velocity_mean`: coefficient `-0.002442`, |coef| `0.002442`
- `lag_00__CT1__hp`: coefficient `0.002405`, |coef| `0.002405`
- `lag_00__CT_kills_last_3s`: coefficient `0.002073`, |coef| `0.002073`
- `lag_02__T5__duck_amount`: coefficient `0.002048`, |coef| `0.002048`

## Top 10 utility ridge features

- `lag_13__CT_A_site_active_infernos`: coefficient `-0.001881` (lowers CT win probability)
- `lag_08__CT_A_site_active_infernos`: coefficient `-0.001840` (lowers CT win probability)
- `lag_00__CT_A_site_active_infernos`: coefficient `-0.001390` (lowers CT win probability)
- `lag_06__CT3__flash_duration`: coefficient `-0.001352` (lowers CT win probability)
- `lag_10__CT_A_site_active_infernos`: coefficient `-0.001316` (lowers CT win probability)
- `lag_13__CT_active_infernos`: coefficient `-0.001214` (lowers CT win probability)
- `lag_08__CT_active_infernos`: coefficient `-0.001193` (lowers CT win probability)
- `lag_03__CT2__flash_duration`: coefficient `0.001167` (raises CT win probability)
- `lag_13__active_infernos_total`: coefficient `-0.001135` (lowers CT win probability)
- `lag_01__CT5__molly`: coefficient `0.001117` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004374` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003482` (lowers CT win probability)
- `lag_13__T_place_CTSPAWN`: coefficient `-0.003420` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003294` (raises CT win probability)
- `lag_00__CT_place_TUNNELS`: coefficient `0.003200` (raises CT win probability)
- `lag_00__CT1__alive`: coefficient `0.002901` (raises CT win probability)
- `lag_00__CT1__is_walking`: coefficient `0.002763` (raises CT win probability)
- `lag_03__CT5__is_scoped`: coefficient `-0.002726` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002702` (lowers CT win probability)
- `lag_10__CT5__is_scoped`: coefficient `0.002641` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `25317`, seconds `78.00`, LSTM delta `-0.2479`

Top all feature movements:
- `lag_13__T_place_ENTRANCE`: contribution `-0.028780`
- `lag_13__T_place_CTSPAWN`: contribution `-0.016313`
- `lag_00__T_kills_last_3s`: contribution `-0.011032`
- `lag_00__kill_diff_last_3s`: contribution `-0.010528`
- `lag_00__CT_place_TUNNELS`: contribution `-0.009792`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `22373`, seconds `32.00`, LSTM delta `-0.2191`

Top all feature movements:
- `lag_03__T_place_LONGDOG`: contribution `-0.011725`
- `lag_00__T_kills_last_3s`: contribution `-0.011032`
- `lag_00__kill_diff_last_3s`: contribution `-0.010528`
- `lag_03__CT5__is_scoped`: contribution `+0.009747`
- `lag_06__CT3__flash_duration`: contribution `-0.008900`

Top utility-only movements:
- `lag_06__CT3__flash_duration`: contribution `-0.008900`
- `lag_06__CT_flash_duration_sum`: contribution `-0.004209`

### tick `22693`, seconds `37.00`, LSTM delta `+0.2068`

Top all feature movements:
- `lag_02__T_place_ELECTRICALBOX`: contribution `-0.030279`
- `lag_04__T_place_ELECTRICALBOX`: contribution `+0.025403`
- `lag_13__T_place_LONGDOG`: contribution `+0.011121`
- `lag_00__kill_diff_last_3s`: contribution `+0.010528`
- `lag_10__CT5__is_scoped`: contribution `+0.009444`

Top utility-only movements:
- `lag_13__CT_A_site_active_infernos`: contribution `+0.006640`
- `lag_00__CT2__flash_duration`: contribution `+0.004574`

### tick `22405`, seconds `32.50`, LSTM delta `+0.1725`

Top all feature movements:
- `lag_04__T_place_LONGDOG`: contribution `+0.012015`
- `lag_00__kill_diff_last_3s`: contribution `+0.010528`
- `lag_04__T_place_DUMPSTER`: contribution `+0.008227`
- `lag_00__T_shots_fired_sum`: contribution `+0.007509`
- `lag_00__CT_kills_last_3s`: contribution `+0.005985`

Top utility-only movements:
- `lag_07__CT3__flash_duration`: contribution `+0.005657`
- `lag_04__CT_A_site_active_infernos`: contribution `+0.003518`

### tick `22341`, seconds `31.50`, LSTM delta `-0.1196`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.011032`
- `lag_00__kill_diff_last_3s`: contribution `-0.010528`
- `lag_03__CT5__is_scoped`: contribution `-0.009747`
- `lag_02__T_place_DUMPSTER`: contribution `-0.008427`
- `lag_00__T_damage_last_5s`: contribution `-0.008164`

Top utility-only movements:
- `lag_05__CT3__flash_duration`: contribution `-0.006651`
- `lag_04__CT_A_site_active_infernos`: contribution `-0.003518`
- `lag_04__CT_B_site_active_infernos`: contribution `-0.002757`
- `lag_00__CT3__flash_duration`: contribution `-0.002623`
