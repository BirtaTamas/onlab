# Local Round Explainability

- csv_path: `processed_full\iem_chengdu\iem-chengdu-2025-heroic-vs-natus-vincere-bo3-P_vZ7pAIyzYcLTUjDHhSUR\heroic-vs-natus-vincere-m2-ancient.csv`
- round_num: `15`

## Largest probability jumps

- tick `119734`, seconds `14.00`, LSTM `0.0375`, delta `-0.1045`
- tick `118870`, seconds `0.50`, LSTM `0.1497`, delta `-0.0617`
- tick `119702`, seconds `13.50`, LSTM `0.1420`, delta `-0.0485`
- tick `119606`, seconds `12.00`, LSTM `0.1718`, delta `+0.0246`
- tick `119158`, seconds `5.00`, LSTM `0.0791`, delta `-0.0236`
- tick `119542`, seconds `11.00`, LSTM `0.1591`, delta `+0.0223`
- tick `118902`, seconds `1.00`, LSTM `0.1286`, delta `-0.0211`
- tick `119478`, seconds `10.00`, LSTM `0.1255`, delta `+0.0204`
- tick `119446`, seconds `9.50`, LSTM `0.1051`, delta `+0.0186`
- tick `121078`, seconds `35.00`, LSTM `0.0062`, delta `-0.0133`

## Top 15 local ridge features

- `lag_00__CT_place_UNKNOWN`: coefficient `0.001673`, |coef| `0.001673`
- `lag_01__CT_place_UNKNOWN`: coefficient `-0.001036`, |coef| `0.001036`
- `lag_15__CT_flashes_last_5s`: coefficient `0.001017`, |coef| `0.001017`
- `lag_12__T_flashes_last_5s`: coefficient `0.000840`, |coef| `0.000840`
- `lag_05__CT4__flash_duration`: coefficient `-0.000713`, |coef| `0.000713`
- `lag_12__T4__flash_duration`: coefficient `-0.000668`, |coef| `0.000668`
- `lag_03__T_place_TSIDELOWER`: coefficient `0.000592`, |coef| `0.000592`
- `lag_00__CT4__flash_duration`: coefficient `0.000592`, |coef| `0.000592`
- `lag_14__CT_flashes_last_5s`: coefficient `0.000533`, |coef| `0.000533`
- `lag_15__T_place_TUNNEL`: coefficient `0.000527`, |coef| `0.000527`
- `lag_03__T_place_TSIDEUPPER`: coefficient `-0.000497`, |coef| `0.000497`
- `lag_03__CT_place_UNKNOWN`: coefficient `0.000488`, |coef| `0.000488`
- `lag_04__CT_place_UNKNOWN`: coefficient `-0.000484`, |coef| `0.000484`
- `lag_00__T_kills_last_3s`: coefficient `-0.000466`, |coef| `0.000466`
- `lag_11__T_flashes_last_5s`: coefficient `0.000441`, |coef| `0.000441`
