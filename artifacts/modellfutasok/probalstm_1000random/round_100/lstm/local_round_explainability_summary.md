# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-imperial-vs-metizport-bo3-yMtoBsoZq-jiQ0fSUscH7u/imperial-vs-metizport-m2-dust2.csv`
- round_num: `3`

## Largest probability jumps

- tick `20307`, seconds `0.50`, LSTM `0.0161`, delta `-0.0316`
- tick `21459`, seconds `18.50`, LSTM `0.0102`, delta `-0.0099`
- tick `20339`, seconds `1.00`, LSTM `0.0113`, delta `-0.0049`
- tick `21427`, seconds `18.00`, LSTM `0.0202`, delta `-0.0042`
- tick `20691`, seconds `6.50`, LSTM `0.0206`, delta `+0.0040`
- tick `20947`, seconds `10.50`, LSTM `0.0248`, delta `+0.0037`
- tick `21011`, seconds `11.50`, LSTM `0.0232`, delta `-0.0037`
- tick `20563`, seconds `4.50`, LSTM `0.0175`, delta `+0.0037`
- tick `21075`, seconds `12.50`, LSTM `0.0195`, delta `-0.0029`
- tick `21523`, seconds `19.50`, LSTM `0.0057`, delta `-0.0027`

## Top 15 local ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000312`, |coef| `0.000312`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000284`, |coef| `0.000284`
- `lag_00__T_velocity_mean`: coefficient `-0.000275`, |coef| `0.000275`
- `lag_00__CT_velocity_mean`: coefficient `-0.000220`, |coef| `0.000220`
- `lag_01__armor_diff`: coefficient `0.000204`, |coef| `0.000204`
- `lag_01__smoke_inv_diff`: coefficient `0.000201`, |coef| `0.000201`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000181`, |coef| `0.000181`
- `lag_01__utility_inv_diff`: coefficient `0.000176`, |coef| `0.000176`
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000172`, |coef| `0.000172`
- `lag_01__CT_armor_sum`: coefficient `0.000171`, |coef| `0.000171`
- `lag_01__T1__has_bomb`: coefficient `-0.000161`, |coef| `0.000161`
- `lag_01__centroid_distance_xy`: coefficient `-0.000160`, |coef| `0.000160`
- `lag_01__molly_inv_diff`: coefficient `0.000151`, |coef| `0.000151`
- `lag_01__equip_diff`: coefficient `0.000146`, |coef| `0.000146`
- `lag_01__T_smoke_inv`: coefficient `-0.000138`, |coef| `0.000138`

## Top 10 utility ridge features

- `lag_01__smoke_inv_diff`: coefficient `0.000201` (raises CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000176` (raises CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000151` (raises CT win probability)
- `lag_01__T_smoke_inv`: coefficient `-0.000138` (lowers CT win probability)
- `lag_01__T4__molly`: coefficient `-0.000112` (lowers CT win probability)
- `lag_01__T_molly_inv`: coefficient `-0.000110` (lowers CT win probability)
- `lag_01__T2__molly`: coefficient `-0.000106` (lowers CT win probability)
- `lag_01__T2__utility_total`: coefficient `-0.000105` (lowers CT win probability)
- `lag_01__T1__molly`: coefficient `-0.000100` (lowers CT win probability)
- `lag_01__T3__molly`: coefficient `-0.000100` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000312` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000284` (lowers CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000275` (lowers CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000220` (lowers CT win probability)
- `lag_01__armor_diff`: coefficient `0.000204` (raises CT win probability)
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000181` (lowers CT win probability)
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000172` (lowers CT win probability)
- `lag_01__CT_armor_sum`: coefficient `0.000171` (raises CT win probability)
- `lag_01__T1__has_bomb`: coefficient `-0.000161` (lowers CT win probability)
- `lag_01__centroid_distance_xy`: coefficient `-0.000160` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `20307`, seconds `0.50`, LSTM delta `-0.0316`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.001490`
- `lag_01__T_place_TSPAWN`: contribution `-0.001259`
- `lag_00__T_velocity_mean`: contribution `-0.001019`
- `lag_00__CT_velocity_mean`: contribution `-0.000770`
- `lag_01__smoke_inv_diff`: contribution `-0.000639`

Top utility-only movements:
- `lag_01__smoke_inv_diff`: contribution `-0.000639`
- `lag_01__utility_inv_diff`: contribution `-0.000465`
- `lag_01__molly_inv_diff`: contribution `-0.000330`
- `lag_01__T_smoke_inv`: contribution `-0.000316`

### tick `21459`, seconds `18.50`, LSTM delta `-0.0099`

Top all feature movements:
- `lag_08__CT_place_BDOORS`: contribution `-0.000650`
- `lag_07__CT_place_BDOORS`: contribution `-0.000623`
- `lag_15__T_place_TUNNELSTAIRS`: contribution `-0.000546`
- `lag_06__CT_place_BDOORS`: contribution `-0.000462`
- `lag_14__CT_place_BDOORS`: contribution `-0.000409`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `20339`, seconds `1.00`, LSTM delta `-0.0049`

Top all feature movements:
- `lag_02__CT_place_CTSPAWN`: contribution `-0.000552`
- `lag_02__T_place_TSPAWN`: contribution `-0.000481`
- `lag_02__armor_diff`: contribution `-0.000249`
- `lag_02__smoke_inv_diff`: contribution `-0.000244`
- `lag_02__T_closest_enemy_dist`: contribution `-0.000201`

Top utility-only movements:
- `lag_02__smoke_inv_diff`: contribution `-0.000244`
- `lag_02__utility_inv_diff`: contribution `-0.000186`
- `lag_02__molly_inv_diff`: contribution `-0.000134`
- `lag_02__T_smoke_inv`: contribution `-0.000118`

### tick `21427`, seconds `18.00`, LSTM delta `-0.0042`

Top all feature movements:
- `lag_07__CT_place_BDOORS`: contribution `-0.000623`
- `lag_06__CT_place_BDOORS`: contribution `-0.000462`
- `lag_14__CT_place_BDOORS`: contribution `-0.000409`
- `lag_14__T_place_TUNNELSTAIRS`: contribution `-0.000370`
- `lag_14__T_place_LOWERTUNNEL`: contribution `-0.000227`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `20691`, seconds `6.50`, LSTM delta `+0.0040`

Top all feature movements:
- `lag_05__T_place_OUTSIDETUNNEL`: contribution `+0.000222`
- `lag_00__T_velocity_mean`: contribution `+0.000191`
- `lag_13__CT_place_CTSPAWN`: contribution `+0.000175`
- `lag_06__T_place_OUTSIDETUNNEL`: contribution `+0.000172`
- `lag_07__CT_place_MIDDOORS`: contribution `+0.000147`

Top utility-only movements:
- `lag_13__smoke_inv_diff`: contribution `+0.000073`
