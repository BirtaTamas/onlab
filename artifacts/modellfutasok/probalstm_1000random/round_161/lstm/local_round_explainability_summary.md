# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-faze-vs-virtuspro-bo3-YDlVsCnS6YPgcr85bBYoPq/faze-vs-virtus-pro-m2-mirage.csv`
- round_num: `14`

## Largest probability jumps

- tick `97351`, seconds `0.50`, LSTM `0.0157`, delta `-0.0309`
- tick `99015`, seconds `26.50`, LSTM `0.0580`, delta `+0.0116`
- tick `99175`, seconds `29.00`, LSTM `0.0407`, delta `-0.0116`
- tick `99463`, seconds `33.50`, LSTM `0.0445`, delta `+0.0096`
- tick `97895`, seconds `9.00`, LSTM `0.0345`, delta `+0.0084`
- tick `99271`, seconds `30.50`, LSTM `0.0464`, delta `+0.0071`
- tick `101863`, seconds `71.00`, LSTM `0.0340`, delta `+0.0067`
- tick `99911`, seconds `40.50`, LSTM `0.0333`, delta `-0.0063`
- tick `97991`, seconds `10.50`, LSTM `0.0474`, delta `+0.0062`
- tick `97383`, seconds `1.00`, LSTM `0.0098`, delta `-0.0059`

## Top 15 local ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000338`, |coef| `0.000338`
- `lag_00__T_velocity_mean`: coefficient `-0.000283`, |coef| `0.000283`
- `lag_00__T_walking_count`: coefficient `0.000282`, |coef| `0.000282`
- `lag_00__CT_walking_count`: coefficient `0.000278`, |coef| `0.000278`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000253`, |coef| `0.000253`
- `lag_00__CT_velocity_mean`: coefficient `-0.000251`, |coef| `0.000251`
- `lag_00__CT3__is_walking`: coefficient `0.000237`, |coef| `0.000237`
- `lag_00__T1__is_walking`: coefficient `0.000205`, |coef| `0.000205`
- `lag_01__smoke_inv_diff`: coefficient `0.000195`, |coef| `0.000195`
- `lag_00__T2__is_walking`: coefficient `0.000191`, |coef| `0.000191`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000186`, |coef| `0.000186`
- `lag_01__armor_diff`: coefficient `0.000185`, |coef| `0.000185`
- `lag_01__T3__has_bomb`: coefficient `-0.000172`, |coef| `0.000172`
- `lag_00__T4__is_walking`: coefficient `0.000168`, |coef| `0.000168`
- `lag_01__T_place_TOPOFMID`: coefficient `0.000167`, |coef| `0.000167`

## Top 10 utility ridge features

- `lag_01__smoke_inv_diff`: coefficient `0.000195` (raises CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000143` (raises CT win probability)
- `lag_01__T4__smoke`: coefficient `-0.000137` (lowers CT win probability)
- `lag_01__T_smoke_inv`: coefficient `-0.000136` (lowers CT win probability)
- `lag_00__T_flashes_last_5s`: coefficient `-0.000131` (lowers CT win probability)
- `lag_01__T5__utility_total`: coefficient `-0.000114` (lowers CT win probability)
- `lag_01__T3__flash`: coefficient `-0.000113` (lowers CT win probability)
- `lag_01__T5__flash`: coefficient `-0.000109` (lowers CT win probability)
- `lag_01__flash_inv_diff`: coefficient `0.000105` (raises CT win probability)
- `lag_01__T2__smoke`: coefficient `-0.000101` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000338` (lowers CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000283` (lowers CT win probability)
- `lag_00__T_walking_count`: coefficient `0.000282` (raises CT win probability)
- `lag_00__CT_walking_count`: coefficient `0.000278` (raises CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000253` (lowers CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000251` (lowers CT win probability)
- `lag_00__CT3__is_walking`: coefficient `0.000237` (raises CT win probability)
- `lag_00__T1__is_walking`: coefficient `0.000205` (raises CT win probability)
- `lag_00__T2__is_walking`: coefficient `0.000191` (raises CT win probability)
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000186` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `97351`, seconds `0.50`, LSTM delta `-0.0309`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.001617`
- `lag_01__T_place_TSPAWN`: contribution `-0.001122`
- `lag_00__CT_velocity_mean`: contribution `-0.000871`
- `lag_00__T_velocity_mean`: contribution `-0.000757`
- `lag_01__smoke_inv_diff`: contribution `-0.000621`

Top utility-only movements:
- `lag_01__smoke_inv_diff`: contribution `-0.000621`
- `lag_01__utility_inv_diff`: contribution `-0.000408`
- `lag_01__T_smoke_inv`: contribution `-0.000309`
- `lag_01__T5__utility_total`: contribution `-0.000264`
- `lag_01__T3__flash`: contribution `-0.000254`

### tick `99015`, seconds `26.50`, LSTM delta `+0.0116`

Top all feature movements:
- `lag_00__CT_walking_count`: contribution `+0.000748`
- `lag_04__T_place_HOUSE`: contribution `+0.000581`
- `lag_00__T1__duck_amount`: contribution `+0.000493`
- `lag_01__T1__duck_amount`: contribution `+0.000477`
- `lag_06__CT_place_CATWALK`: contribution `+0.000400`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `99175`, seconds `29.00`, LSTM delta `-0.0116`

Top all feature movements:
- `lag_10__T1__duck_amount`: contribution `-0.000451`
- `lag_00__T2__is_walking`: contribution `-0.000439`
- `lag_00__T_place_PALACEINTERIOR`: contribution `-0.000418`
- `lag_11__CT_place_CATWALK`: contribution `-0.000412`
- `lag_15__T_place_SIDEALLEY`: contribution `-0.000385`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `99463`, seconds `33.50`, LSTM delta `+0.0096`

Top all feature movements:
- `lag_00__T_walking_count`: contribution `+0.000675`
- `lag_00__CT3__is_walking`: contribution `+0.000565`
- `lag_00__CT_walking_count`: contribution `+0.000499`
- `lag_00__T1__is_walking`: contribution `+0.000467`
- `lag_05__CT_place_SNIPERSNEST`: contribution `+0.000408`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `97895`, seconds `9.00`, LSTM delta `+0.0084`

Top all feature movements:
- `lag_14__T_place_SIDEALLEY`: contribution `+0.000769`
- `lag_01__CT_place_SHOP`: contribution `+0.000499`
- `lag_04__CT_place_SHOP`: contribution `+0.000474`
- `lag_00__T2__is_walking`: contribution `+0.000439`
- `lag_06__CT_place_SHOP`: contribution `+0.000325`

Top utility-only movements:
- No utility movement among the top local contributors.
