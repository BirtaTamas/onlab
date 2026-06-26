# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-furia-vs-vitality-bo3-ZNzuF_vw0WBzn8QEbGrbgj/furia-vs-vitality-m1-overpass.csv`
- round_num: `9`

## Largest probability jumps

- tick `79402`, seconds `74.50`, LSTM `0.2226`, delta `-0.3662`
- tick `78858`, seconds `66.00`, LSTM `0.5993`, delta `+0.2604`
- tick `79722`, seconds `79.50`, LSTM `0.0300`, delta `-0.1116`
- tick `78282`, seconds `57.00`, LSTM `0.5685`, delta `-0.1082`
- tick `79434`, seconds `75.00`, LSTM `0.1212`, delta `-0.1014`
- tick `79370`, seconds `74.00`, LSTM `0.5888`, delta `-0.0907`
- tick `78538`, seconds `61.00`, LSTM `0.4078`, delta `+0.0815`
- tick `78314`, seconds `57.50`, LSTM `0.5023`, delta `-0.0662`
- tick `78922`, seconds `67.00`, LSTM `0.6803`, delta `+0.0534`
- tick `78346`, seconds `58.00`, LSTM `0.4501`, delta `-0.0522`

## Top 15 local ridge features

- `lag_04__CT_place_STORAGEROOM`: coefficient `0.003149`, |coef| `0.003149`
- `lag_14__T_place_CONSTRUCTION`: coefficient `0.003143`, |coef| `0.003143`
- `lag_15__T_place_CONSTRUCTION`: coefficient `0.002904`, |coef| `0.002904`
- `lag_04__CT_place_SNIPERSNEST`: coefficient `0.002770`, |coef| `0.002770`
- `lag_07__CT_place_STORAGEROOM`: coefficient `-0.002643`, |coef| `0.002643`
- `lag_00__kill_diff_last_3s`: coefficient `0.002487`, |coef| `0.002487`
- `lag_08__CT_place_BACKOFA`: coefficient `-0.002243`, |coef| `0.002243`
- `lag_09__T_place_WATER`: coefficient `-0.001926`, |coef| `0.001926`
- `lag_00__damage_diff_last_5s`: coefficient `0.001921`, |coef| `0.001921`
- `lag_07__CT_place_SNIPERSNEST`: coefficient `0.001902`, |coef| `0.001902`
- `lag_00__T_kills_last_3s`: coefficient `-0.001893`, |coef| `0.001893`
- `lag_04__CT_place_BACKOFA`: coefficient `-0.001876`, |coef| `0.001876`
- `lag_00__T_place_LOWERPARK`: coefficient `-0.001833`, |coef| `0.001833`
- `lag_02__CT_place_WATER`: coefficient `0.001810`, |coef| `0.001810`
- `lag_09__T_place_CONNECTOR`: coefficient `0.001793`, |coef| `0.001793`

## Top 10 utility ridge features

- `lag_00__CT1__molly`: coefficient `0.001456` (raises CT win probability)
- `lag_00__CT1__utility_total`: coefficient `0.001260` (raises CT win probability)
- `lag_10__T4__smoke`: coefficient `0.001068` (raises CT win probability)
- `lag_00__CT1__flash`: coefficient `0.001047` (raises CT win probability)
- `lag_01__T_smokes_last_5s`: coefficient `0.000954` (raises CT win probability)
- `lag_01__CT5__flash`: coefficient `0.000920` (raises CT win probability)
- `lag_15__CT5__smoke`: coefficient `-0.000902` (lowers CT win probability)
- `lag_11__CT_active_smokes`: coefficient `0.000902` (raises CT win probability)
- `lag_00__CT_molly_inv`: coefficient `0.000874` (raises CT win probability)
- `lag_00__CT_utility_inv`: coefficient `0.000870` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_04__CT_place_STORAGEROOM`: coefficient `0.003149` (raises CT win probability)
- `lag_14__T_place_CONSTRUCTION`: coefficient `0.003143` (raises CT win probability)
- `lag_15__T_place_CONSTRUCTION`: coefficient `0.002904` (raises CT win probability)
- `lag_04__CT_place_SNIPERSNEST`: coefficient `0.002770` (raises CT win probability)
- `lag_07__CT_place_STORAGEROOM`: coefficient `-0.002643` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002487` (raises CT win probability)
- `lag_08__CT_place_BACKOFA`: coefficient `-0.002243` (lowers CT win probability)
- `lag_09__T_place_WATER`: coefficient `-0.001926` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001921` (raises CT win probability)
- `lag_07__CT_place_SNIPERSNEST`: coefficient `0.001902` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `79402`, seconds `74.50`, LSTM delta `-0.3662`

Top all feature movements:
- `lag_04__CT_place_STORAGEROOM`: contribution `-0.067367`
- `lag_07__CT_place_STORAGEROOM`: contribution `-0.056534`
- `lag_14__T_place_CONSTRUCTION`: contribution `-0.039062`
- `lag_04__CT_place_BACKOFA`: contribution `-0.018115`
- `lag_02__CT_place_WATER`: contribution `-0.011001`

Top utility-only movements:
- `lag_00__CT1__molly`: contribution `-0.003625`

### tick `78858`, seconds `66.00`, LSTM delta `+0.2604`

Top all feature movements:
- `lag_08__CT_place_BACKOFA`: contribution `+0.021661`
- `lag_04__CT_place_SNIPERSNEST`: contribution `+0.014835`
- `lag_10__CT_place_BACKOFA`: contribution `+0.013789`
- `lag_09__T_place_WATER`: contribution `+0.010994`
- `lag_09__T_place_CONNECTOR`: contribution `+0.008680`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `79722`, seconds `79.50`, LSTM delta `-0.1116`

Top all feature movements:
- `lag_14__CT_place_STORAGEROOM`: contribution `-0.014389`
- `lag_01__CT_place_WATER`: contribution `-0.009662`
- `lag_02__CT_place_STAIRS`: contribution `-0.008908`
- `lag_09__T_place_CONNECTOR`: contribution `-0.008680`
- `lag_14__CT_place_BACKOFA`: contribution `-0.007565`

Top utility-only movements:
- `lag_08__T4__flash_duration`: contribution `-0.002021`

### tick `78282`, seconds `57.00`, LSTM delta `-0.1082`

Top all feature movements:
- `lag_07__T_place_CONSTRUCTION`: contribution `-0.016237`
- `lag_02__T_place_PIPE`: contribution `-0.013096`
- `lag_15__CT_place_WATER`: contribution `-0.009196`
- `lag_00__T_kills_last_3s`: contribution `-0.005997`
- `lag_00__kill_diff_last_3s`: contribution `-0.005987`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `79434`, seconds `75.00`, LSTM delta `-0.1014`

Top all feature movements:
- `lag_15__T_place_CONSTRUCTION`: contribution `-0.036092`
- `lag_08__CT_place_STORAGEROOM`: contribution `-0.014773`
- `lag_05__CT_place_STORAGEROOM`: contribution `-0.011544`
- `lag_15__CT_place_WATER`: contribution `+0.009196`
- `lag_00__T_place_LOWERPARK`: contribution `-0.007390`

Top utility-only movements:
- No utility movement among the top local contributors.
