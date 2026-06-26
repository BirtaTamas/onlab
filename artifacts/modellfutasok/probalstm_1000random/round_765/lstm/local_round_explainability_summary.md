# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-faze-vs-virtuspro-bo3-YDlVsCnS6YPgcr85bBYoPq/faze-vs-virtus-pro-m2-mirage.csv`
- round_num: `5`

## Largest probability jumps

- tick `34011`, seconds `56.00`, LSTM `0.0729`, delta `+0.0401`
- tick `36987`, seconds `102.50`, LSTM `0.0416`, delta `+0.0321`
- tick `30459`, seconds `0.50`, LSTM `0.0158`, delta `-0.0305`
- tick `37243`, seconds `106.50`, LSTM `0.0208`, delta `-0.0270`
- tick `34331`, seconds `61.00`, LSTM `0.0728`, delta `-0.0198`
- tick `34171`, seconds `58.50`, LSTM `0.1323`, delta `+0.0165`
- tick `34043`, seconds `56.50`, LSTM `0.0892`, delta `+0.0163`
- tick `34235`, seconds `59.50`, LSTM `0.1034`, delta `-0.0157`
- tick `35259`, seconds `75.50`, LSTM `0.0159`, delta `-0.0151`
- tick `35163`, seconds `74.00`, LSTM `0.0360`, delta `-0.0147`

## Top 15 local ridge features

- `lag_03__T_place_LADDER`: coefficient `-0.000485`, |coef| `0.000485`
- `lag_00__T_place_STAIRS`: coefficient `-0.000436`, |coef| `0.000436`
- `lag_00__kill_diff_last_3s`: coefficient `0.000419`, |coef| `0.000419`
- `lag_12__T_place_PALACEALLEY`: coefficient `0.000402`, |coef| `0.000402`
- `lag_13__T1__duck_amount`: coefficient `0.000369`, |coef| `0.000369`
- `lag_00__CT_kills_last_3s`: coefficient `0.000340`, |coef| `0.000340`
- `lag_00__damage_diff_last_5s`: coefficient `0.000327`, |coef| `0.000327`
- `lag_15__T_place_PALACEALLEY`: coefficient `0.000326`, |coef| `0.000326`
- `lag_00__T_place_JUNGLE`: coefficient `-0.000309`, |coef| `0.000309`
- `lag_09__CT_place_TSPAWN`: coefficient `-0.000306`, |coef| `0.000306`
- `lag_08__T_place_STAIRS`: coefficient `0.000301`, |coef| `0.000301`
- `lag_13__T_place_PALACEALLEY`: coefficient `0.000290`, |coef| `0.000290`
- `lag_13__T_place_LADDER`: coefficient `0.000286`, |coef| `0.000286`
- `lag_15__T1__duck_amount`: coefficient `0.000278`, |coef| `0.000278`
- `lag_01__CT_place_TSPAWN`: coefficient `0.000269`, |coef| `0.000269`

## Top 10 utility ridge features

- `lag_01__utility_inv_diff`: coefficient `0.000188` (raises CT win probability)
- `lag_01__T3__utility_total`: coefficient `-0.000175` (lowers CT win probability)
- `lag_01__T3__molly`: coefficient `-0.000169` (lowers CT win probability)
- `lag_01__flash_inv_diff`: coefficient `0.000161` (raises CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000155` (raises CT win probability)
- `lag_01__T3__flash`: coefficient `-0.000152` (lowers CT win probability)
- `lag_09__T_A_site_active_infernos`: coefficient `-0.000143` (lowers CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000137` (raises CT win probability)
- `lag_00__T3__flash`: coefficient `-0.000135` (lowers CT win probability)
- `lag_10__T_A_site_active_infernos`: coefficient `-0.000132` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_03__T_place_LADDER`: coefficient `-0.000485` (lowers CT win probability)
- `lag_00__T_place_STAIRS`: coefficient `-0.000436` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000419` (raises CT win probability)
- `lag_12__T_place_PALACEALLEY`: coefficient `0.000402` (raises CT win probability)
- `lag_13__T1__duck_amount`: coefficient `0.000369` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000340` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000327` (raises CT win probability)
- `lag_15__T_place_PALACEALLEY`: coefficient `0.000326` (raises CT win probability)
- `lag_00__T_place_JUNGLE`: coefficient `-0.000309` (lowers CT win probability)
- `lag_09__CT_place_TSPAWN`: coefficient `-0.000306` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `34011`, seconds `56.00`, LSTM delta `+0.0401`

Top all feature movements:
- `lag_03__T_place_LADDER`: contribution `+0.010970`
- `lag_12__T_place_PALACEALLEY`: contribution `+0.001398`
- `lag_00__CT_place_TRUCK`: contribution `+0.001291`
- `lag_10__CT_place_JUNGLE`: contribution `+0.001212`
- `lag_15__T_place_PALACEALLEY`: contribution `+0.001134`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `36987`, seconds `102.50`, LSTM delta `+0.0321`

Top all feature movements:
- `lag_00__T_place_STAIRS`: contribution `+0.008343`
- `lag_01__CT_place_TSPAWN`: contribution `+0.002016`
- `lag_15__T1__duck_amount`: contribution `+0.001089`
- `lag_00__kill_diff_last_3s`: contribution `+0.001008`
- `lag_00__CT_kills_last_3s`: contribution `+0.000982`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `30459`, seconds `0.50`, LSTM delta `-0.0305`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.001287`
- `lag_01__T_place_TSPAWN`: contribution `-0.001170`
- `lag_00__CT_velocity_mean`: contribution `-0.000760`
- `lag_00__T_velocity_mean`: contribution `-0.000741`
- `lag_01__utility_inv_diff`: contribution `-0.000621`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `-0.000621`
- `lag_01__molly_inv_diff`: contribution `-0.000433`
- `lag_01__flash_inv_diff`: contribution `-0.000430`
- `lag_01__smoke_inv_diff`: contribution `-0.000349`
- `lag_01__T_molly_inv`: contribution `-0.000280`

### tick `37243`, seconds `106.50`, LSTM delta `-0.0270`

Top all feature movements:
- `lag_08__T_place_STAIRS`: contribution `-0.005766`
- `lag_09__CT_place_TSPAWN`: contribution `-0.002289`
- `lag_13__T1__duck_amount`: contribution `-0.001446`
- `lag_15__T1__duck_amount`: contribution `-0.001089`
- `lag_00__kill_diff_last_3s`: contribution `-0.001008`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `34331`, seconds `61.00`, LSTM delta `-0.0198`

Top all feature movements:
- `lag_13__T_place_LADDER`: contribution `-0.006470`
- `lag_14__T3__duck_amount`: contribution `+0.000803`
- `lag_00__damage_diff_last_5s`: contribution `-0.000737`
- `lag_03__CT_place_TRUCK`: contribution `-0.000733`
- `lag_10__T_place_PALACEINTERIOR`: contribution `-0.000683`

Top utility-only movements:
- No utility movement among the top local contributors.
