# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-mouz-vs-falcons-bo3-OIe4ELGS25ekkV8Rf6FbR4/mouz-vs-falcons-m3-mirage.csv`
- round_num: `16`

## Largest probability jumps

- tick `127319`, seconds `32.50`, LSTM `0.1252`, delta `-0.2221`
- tick `126839`, seconds `25.00`, LSTM `0.3948`, delta `-0.1832`
- tick `127511`, seconds `35.50`, LSTM `0.2537`, delta `+0.1811`
- tick `127991`, seconds `43.00`, LSTM `0.0246`, delta `-0.0854`
- tick `127895`, seconds `41.50`, LSTM `0.1272`, delta `-0.0670`
- tick `127831`, seconds `40.50`, LSTM `0.1910`, delta `-0.0514`
- tick `126967`, seconds `27.00`, LSTM `0.3010`, delta `-0.0468`
- tick `127767`, seconds `39.50`, LSTM `0.2513`, delta `-0.0453`
- tick `127191`, seconds `30.50`, LSTM `0.3330`, delta `-0.0439`
- tick `127351`, seconds `33.00`, LSTM `0.0820`, delta `-0.0432`

## Top 15 local ridge features

- `lag_03__T_place_SNIPERSNEST`: coefficient `-0.002808`, |coef| `0.002808`
- `lag_15__CT_place_STAIRS`: coefficient `-0.002375`, |coef| `0.002375`
- `lag_02__T_place_SNIPERSNEST`: coefficient `-0.002139`, |coef| `0.002139`
- `lag_00__T_kills_last_3s`: coefficient `-0.002086`, |coef| `0.002086`
- `lag_00__kill_diff_last_3s`: coefficient `0.001951`, |coef| `0.001951`
- `lag_14__T_place_SNIPERSNEST`: coefficient `0.001840`, |coef| `0.001840`
- `lag_01__CT_place_STAIRS`: coefficient `0.001810`, |coef| `0.001810`
- `lag_00__CT_place_JUNGLE`: coefficient `0.001742`, |coef| `0.001742`
- `lag_09__CT_place_TRUCK`: coefficient `0.001644`, |coef| `0.001644`
- `lag_09__T_place_SNIPERSNEST`: coefficient `0.001541`, |coef| `0.001541`
- `lag_00__CT_place_UNDERPASS`: coefficient `0.001496`, |coef| `0.001496`
- `lag_03__CT_place_SCAFFOLDING`: coefficient `-0.001379`, |coef| `0.001379`
- `lag_10__CT3__flash_duration`: coefficient `0.001369`, |coef| `0.001369`
- `lag_04__T_place_SNIPERSNEST`: coefficient `-0.001363`, |coef| `0.001363`
- `lag_02__CT_place_JUNGLE`: coefficient `-0.001317`, |coef| `0.001317`

## Top 10 utility ridge features

- `lag_10__CT3__flash_duration`: coefficient `0.001369` (raises CT win probability)
- `lag_13__CT_B_site_active_infernos`: coefficient `0.000889` (raises CT win probability)
- `lag_00__CT3__flash`: coefficient `0.000761` (raises CT win probability)
- `lag_15__T1__flash_duration`: coefficient `0.000667` (raises CT win probability)
- `lag_10__CT_flash_duration_sum`: coefficient `0.000646` (raises CT win probability)
- `lag_01__T1__smoke`: coefficient `0.000609` (raises CT win probability)
- `lag_13__CT_B_site_active_smokes`: coefficient `-0.000535` (lowers CT win probability)
- `lag_15__CT3__flash`: coefficient `0.000493` (raises CT win probability)
- `lag_07__CT5__smoke`: coefficient `0.000479` (raises CT win probability)
- `lag_13__CT_active_infernos`: coefficient `0.000473` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_03__T_place_SNIPERSNEST`: coefficient `-0.002808` (lowers CT win probability)
- `lag_15__CT_place_STAIRS`: coefficient `-0.002375` (lowers CT win probability)
- `lag_02__T_place_SNIPERSNEST`: coefficient `-0.002139` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002086` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001951` (raises CT win probability)
- `lag_14__T_place_SNIPERSNEST`: coefficient `0.001840` (raises CT win probability)
- `lag_01__CT_place_STAIRS`: coefficient `0.001810` (raises CT win probability)
- `lag_00__CT_place_JUNGLE`: coefficient `0.001742` (raises CT win probability)
- `lag_09__CT_place_TRUCK`: coefficient `0.001644` (raises CT win probability)
- `lag_09__T_place_SNIPERSNEST`: coefficient `0.001541` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `127319`, seconds `32.50`, LSTM delta `-0.2221`

Top all feature movements:
- `lag_03__T_place_SNIPERSNEST`: contribution `-0.049899`
- `lag_00__CT_place_JUNGLE`: contribution `-0.011179`
- `lag_02__CT_place_JUNGLE`: contribution `-0.008452`
- `lag_00__T_kills_last_3s`: contribution `-0.006608`
- `lag_10__CT_place_JUNGLE`: contribution `-0.006062`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `126839`, seconds `25.00`, LSTM delta `-0.1832`

Top all feature movements:
- `lag_15__CT_place_STAIRS`: contribution `-0.018481`
- `lag_01__CT_place_STAIRS`: contribution `-0.014090`
- `lag_09__CT_place_TRUCK`: contribution `-0.010601`
- `lag_00__CT_place_UNDERPASS`: contribution `-0.008678`
- `lag_10__CT3__flash_duration`: contribution `-0.007316`

Top utility-only movements:
- `lag_10__CT3__flash_duration`: contribution `-0.007316`
- `lag_13__CT_B_site_active_infernos`: contribution `-0.003053`

### tick `127511`, seconds `35.50`, LSTM delta `+0.1811`

Top all feature movements:
- `lag_02__T_place_SNIPERSNEST`: contribution `+0.038011`
- `lag_09__T_place_SNIPERSNEST`: contribution `+0.027376`
- `lag_00__kill_diff_last_3s`: contribution `+0.009391`
- `lag_06__CT_place_JUNGLE`: contribution `+0.006878`
- `lag_00__T_kills_last_3s`: contribution `+0.006608`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `127991`, seconds `43.00`, LSTM delta `-0.0854`

Top all feature movements:
- `lag_03__CT_place_SCAFFOLDING`: contribution `-0.028786`
- `lag_00__CT_place_JUNGLE`: contribution `-0.011179`
- `lag_00__CT_place_SCAFFOLDING`: contribution `+0.007391`
- `lag_00__T_kills_last_3s`: contribution `-0.006608`
- `lag_00__kill_diff_last_3s`: contribution `-0.004696`

Top utility-only movements:
- `lag_10__CT_A_site_active_infernos`: contribution `-0.001428`

### tick `127895`, seconds `41.50`, LSTM delta `-0.0670`

Top all feature movements:
- `lag_14__T_place_SNIPERSNEST`: contribution `-0.032704`
- `lag_00__CT_place_SCAFFOLDING`: contribution `-0.007391`
- `lag_15__T5__duck_amount`: contribution `-0.004507`
- `lag_14__T_place_CTSPAWN`: contribution `-0.003605`
- `lag_00__CT_place_PALACEINTERIOR`: contribution `+0.003520`

Top utility-only movements:
- `lag_07__CT_A_site_active_infernos`: contribution `-0.001441`
- `lag_12__T_A_site_active_smokes`: contribution `-0.001110`
