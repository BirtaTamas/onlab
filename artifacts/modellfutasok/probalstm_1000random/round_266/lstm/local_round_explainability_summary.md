# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-heroic-vs-natus-vincere-bo3-P_vZ7pAIyzYcLTUjDHhSUR/heroic-vs-natus-vincere-m2-ancient.csv`
- round_num: `15`

## Largest probability jumps

- tick `119734`, seconds `14.00`, LSTM `0.0375`, delta `-0.1045`
- tick `118870`, seconds `0.50`, LSTM `0.1496`, delta `-0.0617`
- tick `119702`, seconds `13.50`, LSTM `0.1420`, delta `-0.0485`
- tick `119606`, seconds `12.00`, LSTM `0.1717`, delta `+0.0246`
- tick `119158`, seconds `5.00`, LSTM `0.0791`, delta `-0.0236`
- tick `119542`, seconds `11.00`, LSTM `0.1591`, delta `+0.0223`
- tick `118902`, seconds `1.00`, LSTM `0.1286`, delta `-0.0211`
- tick `119478`, seconds `10.00`, LSTM `0.1255`, delta `+0.0204`
- tick `119446`, seconds `9.50`, LSTM `0.1051`, delta `+0.0186`
- tick `121078`, seconds `35.00`, LSTM `0.0062`, delta `-0.0132`

## Top 15 local ridge features

- `lag_00__CT_place_UNKNOWN`: coefficient `0.001673`, |coef| `0.001673`
- `lag_01__CT_place_UNKNOWN`: coefficient `-0.001036`, |coef| `0.001036`
- `lag_15__CT_flashes_last_5s`: coefficient `0.001017`, |coef| `0.001017`
- `lag_12__T_flashes_last_5s`: coefficient `0.000840`, |coef| `0.000840`
- `lag_05__CT4__flash_duration`: coefficient `-0.000713`, |coef| `0.000713`
- `lag_12__T4__flash_duration`: coefficient `-0.000668`, |coef| `0.000668`
- `lag_00__CT4__flash_duration`: coefficient `0.000592`, |coef| `0.000592`
- `lag_03__T_place_TSIDELOWER`: coefficient `0.000592`, |coef| `0.000592`
- `lag_14__CT_flashes_last_5s`: coefficient `0.000533`, |coef| `0.000533`
- `lag_15__T_place_TUNNEL`: coefficient `0.000527`, |coef| `0.000527`
- `lag_03__T_place_TSIDEUPPER`: coefficient `-0.000497`, |coef| `0.000497`
- `lag_03__CT_place_UNKNOWN`: coefficient `0.000488`, |coef| `0.000488`
- `lag_04__CT_place_UNKNOWN`: coefficient `-0.000484`, |coef| `0.000484`
- `lag_00__T_kills_last_3s`: coefficient `-0.000466`, |coef| `0.000466`
- `lag_11__T_flashes_last_5s`: coefficient `0.000441`, |coef| `0.000441`

## Top 10 utility ridge features

- `lag_15__CT_flashes_last_5s`: coefficient `0.001017` (raises CT win probability)
- `lag_12__T_flashes_last_5s`: coefficient `0.000840` (raises CT win probability)
- `lag_05__CT4__flash_duration`: coefficient `-0.000713` (lowers CT win probability)
- `lag_12__T4__flash_duration`: coefficient `-0.000668` (lowers CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `0.000592` (raises CT win probability)
- `lag_14__CT_flashes_last_5s`: coefficient `0.000533` (raises CT win probability)
- `lag_11__T_flashes_last_5s`: coefficient `0.000441` (raises CT win probability)
- `lag_04__CT4__flash_duration`: coefficient `-0.000382` (lowers CT win probability)
- `lag_11__T4__flash_duration`: coefficient `-0.000366` (lowers CT win probability)
- `lag_00__CT4__smoke`: coefficient `0.000327` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_UNKNOWN`: coefficient `0.001673` (raises CT win probability)
- `lag_01__CT_place_UNKNOWN`: coefficient `-0.001036` (lowers CT win probability)
- `lag_03__T_place_TSIDELOWER`: coefficient `0.000592` (raises CT win probability)
- `lag_15__T_place_TUNNEL`: coefficient `0.000527` (raises CT win probability)
- `lag_03__T_place_TSIDEUPPER`: coefficient `-0.000497` (lowers CT win probability)
- `lag_03__CT_place_UNKNOWN`: coefficient `0.000488` (raises CT win probability)
- `lag_04__CT_place_UNKNOWN`: coefficient `-0.000484` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.000466` (lowers CT win probability)
- `lag_08__T_place_TSIDELOWER`: coefficient `-0.000421` (lowers CT win probability)
- `lag_02__T_place_TSIDELOWER`: coefficient `0.000419` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `119734`, seconds `14.00`, LSTM delta `-0.1045`

Top all feature movements:
- `lag_15__CT_flashes_last_5s`: contribution `-0.011186`
- `lag_12__T_flashes_last_5s`: contribution `-0.007612`
- `lag_05__CT4__flash_duration`: contribution `-0.005268`
- `lag_12__T4__flash_duration`: contribution `-0.004918`
- `lag_03__T_place_TSIDELOWER`: contribution `-0.004436`

Top utility-only movements:
- `lag_15__CT_flashes_last_5s`: contribution `-0.011186`
- `lag_12__T_flashes_last_5s`: contribution `-0.007612`
- `lag_05__CT4__flash_duration`: contribution `-0.005268`
- `lag_12__T4__flash_duration`: contribution `-0.004918`
- `lag_00__CT4__flash_duration`: contribution `-0.004372`

### tick `118870`, seconds `0.50`, LSTM delta `-0.0617`

Top all feature movements:
- `lag_01__CT_place_UNKNOWN`: contribution `-0.036354`
- `lag_01__T_place_TSPAWN`: contribution `-0.000679`
- `lag_01__T_closest_enemy_dist`: contribution `-0.000668`
- `lag_00__CT_velocity_mean`: contribution `-0.000522`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.000476`

Top utility-only movements:
- `lag_01__CT1__flash`: contribution `-0.000439`
- `lag_00__CT2__smoke`: contribution `-0.000439`
- `lag_00__T5__smoke`: contribution `-0.000438`
- `lag_01__T2__molly`: contribution `+0.000276`
- `lag_01__T2__smoke`: contribution `-0.000225`

### tick `119702`, seconds `13.50`, LSTM delta `-0.0485`

Top all feature movements:
- `lag_14__CT_flashes_last_5s`: contribution `-0.005857`
- `lag_11__T_flashes_last_5s`: contribution `-0.003995`
- `lag_02__T_place_TSIDELOWER`: contribution `-0.003137`
- `lag_15__T_place_WATER`: contribution `-0.003072`
- `lag_04__CT4__flash_duration`: contribution `-0.002823`

Top utility-only movements:
- `lag_14__CT_flashes_last_5s`: contribution `-0.005857`
- `lag_11__T_flashes_last_5s`: contribution `-0.003995`
- `lag_04__CT4__flash_duration`: contribution `-0.002823`
- `lag_11__T4__flash_duration`: contribution `-0.002694`
- `lag_04__CT_flash_duration_sum`: contribution `-0.000605`

### tick `119606`, seconds `12.00`, LSTM delta `+0.0246`

Top all feature movements:
- `lag_11__CT_flashes_last_5s`: contribution `+0.002681`
- `lag_12__T_place_WATER`: contribution `-0.002071`
- `lag_01__CT4__flash_duration`: contribution `+0.002006`
- `lag_08__T_flashes_last_5s`: contribution `+0.001806`
- `lag_11__T_place_WATER`: contribution `+0.001648`

Top utility-only movements:
- `lag_11__CT_flashes_last_5s`: contribution `+0.002681`
- `lag_01__CT4__flash_duration`: contribution `+0.002006`
- `lag_08__T_flashes_last_5s`: contribution `+0.001806`
- `lag_08__T4__flash_duration`: contribution `+0.001334`
- `lag_01__CT_flash_duration_sum`: contribution `+0.000631`

### tick `119158`, seconds `5.00`, LSTM delta `-0.0236`

Top all feature movements:
- `lag_10__CT_place_UNKNOWN`: contribution `-0.007599`
- `lag_07__CT_flashes_last_5s`: contribution `-0.003284`
- `lag_04__T_flashes_last_5s`: contribution `-0.002202`
- `lag_05__CT_place_UNKNOWN`: contribution `-0.001584`
- `lag_07__CT_place_UNKNOWN`: contribution `+0.001528`

Top utility-only movements:
- `lag_07__CT_flashes_last_5s`: contribution `-0.003284`
- `lag_04__T_flashes_last_5s`: contribution `-0.002202`
- `lag_10__CT1__flash`: contribution `-0.000338`
