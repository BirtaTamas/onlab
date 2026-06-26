# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-b8-vs-flyquest-bo3-ROTxQXIIApwC88KHLMMjQT/b8-vs-flyquest-m3-inferno.csv`
- round_num: `15`

## Largest probability jumps

- tick `123570`, seconds `39.50`, LSTM `0.0586`, delta `-0.0436`
- tick `123602`, seconds `40.00`, LSTM `0.0235`, delta `-0.0351`
- tick `121074`, seconds `0.50`, LSTM `0.0188`, delta `-0.0331`
- tick `123538`, seconds `39.00`, LSTM `0.1021`, delta `-0.0252`
- tick `123474`, seconds `38.00`, LSTM `0.1155`, delta `+0.0185`
- tick `123378`, seconds `36.50`, LSTM `0.0726`, delta `+0.0155`
- tick `123282`, seconds `35.00`, LSTM `0.0573`, delta `+0.0152`
- tick `123218`, seconds `34.00`, LSTM `0.0306`, delta `+0.0137`
- tick `123442`, seconds `37.50`, LSTM `0.0969`, delta `+0.0123`
- tick `123410`, seconds `37.00`, LSTM `0.0846`, delta `+0.0120`

## Top 15 local ridge features

- `lag_03__CT_place_BACKALLEY`: coefficient `0.000421`, |coef| `0.000421`
- `lag_03__T_place_TRAMP`: coefficient `0.000330`, |coef| `0.000330`
- `lag_15__T1__duck_amount`: coefficient `0.000326`, |coef| `0.000326`
- `lag_09__T_place_UNDERPASS`: coefficient `0.000313`, |coef| `0.000313`
- `lag_01__damage_diff_last_5s`: coefficient `0.000309`, |coef| `0.000309`
- `lag_02__CT_place_BACKALLEY`: coefficient `0.000307`, |coef| `0.000307`
- `lag_10__T_place_UNDERPASS`: coefficient `0.000305`, |coef| `0.000305`
- `lag_11__CT1__duck_amount`: coefficient `0.000303`, |coef| `0.000303`
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000298`, |coef| `0.000298`
- `lag_02__T_place_TRAMP`: coefficient `0.000295`, |coef| `0.000295`
- `lag_00__damage_diff_last_5s`: coefficient `0.000273`, |coef| `0.000273`
- `lag_12__CT1__duck_amount`: coefficient `0.000270`, |coef| `0.000270`
- `lag_09__CT_place_BACKALLEY`: coefficient `-0.000251`, |coef| `0.000251`
- `lag_01__centroid_distance_xy`: coefficient `-0.000244`, |coef| `0.000244`
- `lag_04__CT_place_ARCH`: coefficient `0.000239`, |coef| `0.000239`

## Top 10 utility ridge features

- `lag_01__CT1__molly`: coefficient `0.000207` (raises CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000203` (raises CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000202` (raises CT win probability)
- `lag_02__CT1__molly`: coefficient `0.000194` (raises CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000184` (raises CT win probability)
- `lag_00__T_B_site_active_infernos`: coefficient `-0.000172` (lowers CT win probability)
- `lag_01__T2__utility_total`: coefficient `-0.000155` (lowers CT win probability)
- `lag_01__T2__molly`: coefficient `-0.000151` (lowers CT win probability)
- `lag_01__T2__smoke`: coefficient `-0.000147` (lowers CT win probability)
- `lag_01__T_smoke_inv`: coefficient `-0.000145` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_03__CT_place_BACKALLEY`: coefficient `0.000421` (raises CT win probability)
- `lag_03__T_place_TRAMP`: coefficient `0.000330` (raises CT win probability)
- `lag_15__T1__duck_amount`: coefficient `0.000326` (raises CT win probability)
- `lag_09__T_place_UNDERPASS`: coefficient `0.000313` (raises CT win probability)
- `lag_01__damage_diff_last_5s`: coefficient `0.000309` (raises CT win probability)
- `lag_02__CT_place_BACKALLEY`: coefficient `0.000307` (raises CT win probability)
- `lag_10__T_place_UNDERPASS`: coefficient `0.000305` (raises CT win probability)
- `lag_11__CT1__duck_amount`: coefficient `0.000303` (raises CT win probability)
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000298` (lowers CT win probability)
- `lag_02__T_place_TRAMP`: coefficient `0.000295` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `123570`, seconds `39.50`, LSTM delta `-0.0436`

Top all feature movements:
- `lag_03__CT_place_BACKALLEY`: contribution `-0.006307`
- `lag_09__CT_place_BACKALLEY`: contribution `-0.003763`
- `lag_06__CT_place_BACKALLEY`: contribution `-0.001863`
- `lag_15__T1__duck_amount`: contribution `-0.001276`
- `lag_09__T_place_UNDERPASS`: contribution `-0.001225`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `123602`, seconds `40.00`, LSTM delta `-0.0351`

Top all feature movements:
- `lag_02__CT_place_BACKALLEY`: contribution `-0.004606`
- `lag_04__CT_place_BACKALLEY`: contribution `-0.002743`
- `lag_10__CT_place_BACKALLEY`: contribution `-0.002355`
- `lag_07__CT_place_BACKALLEY`: contribution `+0.001304`
- `lag_10__T_place_UNDERPASS`: contribution `-0.001196`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `121074`, seconds `0.50`, LSTM delta `-0.0331`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.001424`
- `lag_01__T_place_TSPAWN`: contribution `-0.001037`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.000983`
- `lag_01__centroid_distance_xy`: contribution `-0.000945`
- `lag_01__T_closest_enemy_dist`: contribution `-0.000928`

Top utility-only movements:
- `lag_01__smoke_inv_diff`: contribution `-0.000645`
- `lag_01__utility_inv_diff`: contribution `-0.000626`
- `lag_01__molly_inv_diff`: contribution `-0.000514`
- `lag_01__T_smoke_inv`: contribution `-0.000331`
- `lag_01__T2__utility_total`: contribution `-0.000247`

### tick `123538`, seconds `39.00`, LSTM delta `-0.0252`

Top all feature movements:
- `lag_02__CT_place_BACKALLEY`: contribution `-0.004606`
- `lag_05__CT_place_BACKALLEY`: contribution `-0.002780`
- `lag_08__CT_place_BACKALLEY`: contribution `+0.001775`
- `lag_02__T_place_TRAMP`: contribution `-0.000863`
- `lag_14__T1__duck_amount`: contribution `-0.000858`

Top utility-only movements:
- `lag_01__CT1__molly`: contribution `-0.000516`

### tick `123474`, seconds `38.00`, LSTM delta `+0.0185`

Top all feature movements:
- `lag_03__CT_place_BACKALLEY`: contribution `+0.006307`
- `lag_06__CT_place_BACKALLEY`: contribution `-0.001863`
- `lag_15__T1__duck_amount`: contribution `+0.001276`
- `lag_03__T_place_TRAMP`: contribution `+0.000967`
- `lag_01__CT2__duck_amount`: contribution `+0.000833`

Top utility-only movements:
- No utility movement among the top local contributors.
