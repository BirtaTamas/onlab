# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-gamerlegion-vs-complexity-bo3-A8nOd44IyEYHGVOxrkExMv/gamerlegion-vs-complexity-m1-inferno.csv`
- round_num: `1`

## Largest probability jumps

- tick `2169`, seconds `14.00`, LSTM `0.8401`, delta `+0.2534`
- tick `5433`, seconds `65.00`, LSTM `0.8578`, delta `+0.2018`
- tick `5881`, seconds `72.00`, LSTM `0.7223`, delta `-0.1662`
- tick `5177`, seconds `61.00`, LSTM `0.7667`, delta `-0.1534`
- tick `2137`, seconds `13.50`, LSTM `0.5867`, delta `+0.1141`
- tick `6521`, seconds `82.00`, LSTM `0.7926`, delta `+0.0435`
- tick `6073`, seconds `75.00`, LSTM `0.7834`, delta `+0.0416`
- tick `2201`, seconds `14.50`, LSTM `0.8809`, delta `+0.0408`
- tick `5241`, seconds `62.00`, LSTM `0.6954`, delta `-0.0395`
- tick `2777`, seconds `23.50`, LSTM `0.9532`, delta `+0.0336`

## Top 15 local ridge features

- `lag_14__CT_place_BACKALLEY`: coefficient `0.002750`, |coef| `0.002750`
- `lag_00__kill_diff_last_3s`: coefficient `0.002609`, |coef| `0.002609`
- `lag_08__CT_place_SECONDMID`: coefficient `0.002394`, |coef| `0.002394`
- `lag_08__CT_place_QUAD`: coefficient `-0.002184`, |coef| `0.002184`
- `lag_00__damage_diff_last_5s`: coefficient `0.002045`, |coef| `0.002045`
- `lag_00__T_kills_last_3s`: coefficient `-0.001953`, |coef| `0.001953`
- `lag_14__CT_place_BALCONY`: coefficient `-0.001876`, |coef| `0.001876`
- `lag_08__CT_place_APARTMENTS`: coefficient `0.001842`, |coef| `0.001842`
- `lag_09__CT_place_APARTMENTS`: coefficient `0.001720`, |coef| `0.001720`
- `lag_11__CT4__is_walking`: coefficient `-0.001698`, |coef| `0.001698`
- `lag_15__CT_place_BACKALLEY`: coefficient `0.001685`, |coef| `0.001685`
- `lag_11__T_A_site_active_infernos`: coefficient `0.001520`, |coef| `0.001520`
- `lag_15__CT_place_TOPOFMID`: coefficient `0.001469`, |coef| `0.001469`
- `lag_07__CT_place_BACKALLEY`: coefficient `-0.001426`, |coef| `0.001426`
- `lag_06__CT_place_BALCONY`: coefficient `0.001420`, |coef| `0.001420`

## Top 10 utility ridge features

- `lag_11__T_A_site_active_infernos`: coefficient `0.001520` (raises CT win probability)
- `lag_01__T5__flash_duration`: coefficient `-0.001091` (lowers CT win probability)
- `lag_11__T_active_infernos`: coefficient `0.001054` (raises CT win probability)
- `lag_06__T5__flash_duration`: coefficient `0.000868` (raises CT win probability)
- `lag_11__active_infernos_total`: coefficient `0.000717` (raises CT win probability)
- `lag_06__T_flash_duration_sum`: coefficient `0.000716` (raises CT win probability)
- `lag_03__T_A_site_active_infernos`: coefficient `-0.000716` (lowers CT win probability)
- `lag_06__T3__molly`: coefficient `0.000707` (raises CT win probability)
- `lag_06__T1__flash_duration`: coefficient `0.000653` (raises CT win probability)
- `lag_00__CT1__smoke`: coefficient `0.000603` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_14__CT_place_BACKALLEY`: coefficient `0.002750` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002609` (raises CT win probability)
- `lag_08__CT_place_SECONDMID`: coefficient `0.002394` (raises CT win probability)
- `lag_08__CT_place_QUAD`: coefficient `-0.002184` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002045` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001953` (lowers CT win probability)
- `lag_14__CT_place_BALCONY`: coefficient `-0.001876` (lowers CT win probability)
- `lag_08__CT_place_APARTMENTS`: coefficient `0.001842` (raises CT win probability)
- `lag_09__CT_place_APARTMENTS`: coefficient `0.001720` (raises CT win probability)
- `lag_11__CT4__is_walking`: coefficient `-0.001698` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `2169`, seconds `14.00`, LSTM delta `+0.2534`

Top all feature movements:
- `lag_08__CT_place_SECONDMID`: contribution `+0.049080`
- `lag_14__T_place_LOWERMID`: contribution `+0.013768`
- `lag_00__T_place_SECONDMID`: contribution `+0.008835`
- `lag_08__CT_place_APARTMENTS`: contribution `+0.007074`
- `lag_01__T5__flash_duration`: contribution `+0.006992`

Top utility-only movements:
- `lag_01__T5__flash_duration`: contribution `+0.006992`
- `lag_06__T5__flash_duration`: contribution `+0.005375`
- `lag_06__T_flash_duration_sum`: contribution `+0.004331`
- `lag_06__T1__flash_duration`: contribution `+0.002818`

### tick `5433`, seconds `65.00`, LSTM delta `+0.2018`

Top all feature movements:
- `lag_15__CT_place_BACKALLEY`: contribution `+0.025263`
- `lag_08__CT_place_QUAD`: contribution `+0.017211`
- `lag_00__CT_place_BACKALLEY`: contribution `+0.015057`
- `lag_14__CT_place_BALCONY`: contribution `+0.012038`
- `lag_09__CT_place_QUAD`: contribution `+0.006841`

Top utility-only movements:
- `lag_11__T_A_site_active_infernos`: contribution `+0.004525`
- `lag_11__T_active_infernos`: contribution `+0.002195`

### tick `5881`, seconds `72.00`, LSTM delta `-0.1662`

Top all feature movements:
- `lag_14__CT_place_BACKALLEY`: contribution `-0.041225`
- `lag_00__kill_diff_last_3s`: contribution `-0.006280`
- `lag_00__T_kills_last_3s`: contribution `-0.006187`
- `lag_11__T_A_site_active_infernos`: contribution `-0.004525`
- `lag_00__CT_place_APARTMENTS`: contribution `-0.004380`

Top utility-only movements:
- `lag_11__T_A_site_active_infernos`: contribution `-0.004525`
- `lag_11__T_active_infernos`: contribution `-0.002195`

### tick `5177`, seconds `61.00`, LSTM delta `-0.1534`

Top all feature movements:
- `lag_07__CT_place_BACKALLEY`: contribution `-0.021382`
- `lag_00__CT_place_QUAD`: contribution `-0.010445`
- `lag_06__CT_place_BALCONY`: contribution `-0.009111`
- `lag_08__CT_place_APARTMENTS`: contribution `-0.007074`
- `lag_00__kill_diff_last_3s`: contribution `-0.006280`

Top utility-only movements:
- `lag_03__T_A_site_active_infernos`: contribution `-0.002130`
- `lag_06__T3__molly`: contribution `-0.001571`

### tick `2137`, seconds `13.50`, LSTM delta `+0.1141`

Top all feature movements:
- `lag_07__CT_place_SECONDMID`: contribution `+0.017830`
- `lag_15__CT_place_TOPOFMID`: contribution `+0.010659`
- `lag_15__CT_place_ARCH`: contribution `+0.009513`
- `lag_14__T_place_LOWERMID`: contribution `+0.009179`
- `lag_08__CT_place_APARTMENTS`: contribution `+0.007074`

Top utility-only movements:
- `lag_00__T5__flash_duration`: contribution `+0.002950`
- `lag_05__T5__flash_duration`: contribution `+0.002783`
- `lag_05__T_flash_duration_sum`: contribution `+0.002021`
