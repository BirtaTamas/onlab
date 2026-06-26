# Local Round Explainability

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-tyloo-vs-saw-bo3-Ik7_qO98IbLkRUwtJ3asIP/tyloo-vs-saw-m1-nuke.csv`
- round_num: `27`

## Largest probability jumps

- tick `212081`, seconds `33.50`, LSTM `0.1140`, delta `-0.2498`
- tick `211057`, seconds `17.50`, LSTM `0.7058`, delta `+0.1347`
- tick `211793`, seconds `29.00`, LSTM `0.4151`, delta `-0.1265`
- tick `212465`, seconds `39.50`, LSTM `0.0215`, delta `-0.1212`
- tick `211633`, seconds `26.50`, LSTM `0.5576`, delta `-0.0848`
- tick `211121`, seconds `18.50`, LSTM `0.7409`, delta `+0.0394`
- tick `211281`, seconds `21.00`, LSTM `0.7084`, delta `-0.0360`
- tick `212113`, seconds `34.00`, LSTM `0.0794`, delta `-0.0345`
- tick `212433`, seconds `39.00`, LSTM `0.1427`, delta `+0.0329`
- tick `211377`, seconds `22.50`, LSTM `0.6746`, delta `-0.0247`

## Top 15 local ridge features

- `lag_03__CT_place_CONTROL`: coefficient `-0.002132`, |coef| `0.002132`
- `lag_14__CT_place_SECRET`: coefficient `0.002014`, |coef| `0.002014`
- `lag_00__CT_place_CONTROL`: coefficient `0.001801`, |coef| `0.001801`
- `lag_00__T_kills_last_3s`: coefficient `-0.001573`, |coef| `0.001573`
- `lag_00__kill_diff_last_3s`: coefficient `0.001448`, |coef| `0.001448`
- `lag_05__CT_place_SECRET`: coefficient `0.001403`, |coef| `0.001403`
- `lag_00__T_damage_last_5s`: coefficient `-0.001384`, |coef| `0.001384`
- `lag_00__damage_diff_last_5s`: coefficient `0.001283`, |coef| `0.001283`
- `lag_04__T_place_ROOF`: coefficient `-0.001239`, |coef| `0.001239`
- `lag_11__T3__is_scoped`: coefficient `0.001106`, |coef| `0.001106`
- `lag_01__CT2__duck_amount`: coefficient `-0.001068`, |coef| `0.001068`
- `lag_04__T_place_SILO`: coefficient `0.001041`, |coef| `0.001041`
- `lag_02__T2__duck_amount`: coefficient `-0.001034`, |coef| `0.001034`
- `lag_09__damage_diff_last_5s`: coefficient `0.001020`, |coef| `0.001020`
- `lag_13__T_place_ROOF`: coefficient `-0.000984`, |coef| `0.000984`

## Top 10 utility ridge features

- `lag_15__CT1__flash_duration`: coefficient `0.000940` (raises CT win probability)
- `lag_15__CT_flash_duration_sum`: coefficient `0.000775` (raises CT win probability)
- `lag_15__CT5__flash_duration`: coefficient `0.000743` (raises CT win probability)
- `lag_14__CT5__molly`: coefficient `0.000694` (raises CT win probability)
- `lag_00__CT2__flash`: coefficient `0.000659` (raises CT win probability)
- `lag_15__CT_A_site_active_infernos`: coefficient `0.000647` (raises CT win probability)
- `lag_09__CT1__smoke`: coefficient `0.000638` (raises CT win probability)
- `lag_00__CT1__smoke`: coefficient `0.000636` (raises CT win probability)
- `lag_12__T_active_infernos`: coefficient `-0.000636` (lowers CT win probability)
- `lag_00__CT3__utility_total`: coefficient `0.000616` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_03__CT_place_CONTROL`: coefficient `-0.002132` (lowers CT win probability)
- `lag_14__CT_place_SECRET`: coefficient `0.002014` (raises CT win probability)
- `lag_00__CT_place_CONTROL`: coefficient `0.001801` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001573` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001448` (raises CT win probability)
- `lag_05__CT_place_SECRET`: coefficient `0.001403` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001384` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001283` (raises CT win probability)
- `lag_04__T_place_ROOF`: coefficient `-0.001239` (lowers CT win probability)
- `lag_11__T3__is_scoped`: coefficient `0.001106` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `212081`, seconds `33.50`, LSTM delta `-0.2498`

Top all feature movements:
- `lag_03__CT_place_CONTROL`: contribution `-0.022128`
- `lag_14__CT_place_SECRET`: contribution `-0.020733`
- `lag_00__CT_place_CONTROL`: contribution `-0.018698`
- `lag_04__T_place_SILO`: contribution `-0.007073`
- `lag_04__T_place_ROOF`: contribution `-0.007016`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `211057`, seconds `17.50`, LSTM delta `+0.1347`

Top all feature movements:
- `lag_15__CT1__flash_duration`: contribution `+0.005976`
- `lag_13__T_place_ROOF`: contribution `+0.005570`
- `lag_15__CT5__flash_duration`: contribution `+0.005555`
- `lag_15__CT_flash_duration_sum`: contribution `+0.004883`
- `lag_12__CT_shots_fired_sum`: contribution `+0.004863`

Top utility-only movements:
- `lag_15__CT1__flash_duration`: contribution `+0.005976`
- `lag_15__CT5__flash_duration`: contribution `+0.005555`
- `lag_15__CT_flash_duration_sum`: contribution `+0.004883`
- `lag_03__CT1__flash_duration`: contribution `+0.003819`
- `lag_01__CT5__flash_duration`: contribution `+0.003471`

### tick `211793`, seconds `29.00`, LSTM delta `-0.1265`

Top all feature movements:
- `lag_05__CT_place_SECRET`: contribution `-0.014439`
- `lag_11__T3__is_scoped`: contribution `-0.007096`
- `lag_00__T_kills_last_3s`: contribution `-0.004984`
- `lag_00__kill_diff_last_3s`: contribution `-0.003486`
- `lag_03__T3__duck_amount`: contribution `-0.003482`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `212465`, seconds `39.50`, LSTM delta `-0.1212`

Top all feature movements:
- `lag_07__CT_place_HUT`: contribution `-0.008915`
- `lag_15__CT_place_CONTROL`: contribution `-0.008085`
- `lag_00__T_kills_last_3s`: contribution `-0.004984`
- `lag_00__CT_place_HUT`: contribution `-0.003961`
- `lag_02__T2__duck_amount`: contribution `-0.003954`

Top utility-only movements:
- `lag_00__CT3__utility_total`: contribution `-0.001765`

### tick `211633`, seconds `26.50`, LSTM delta `-0.0848`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.004984`
- `lag_00__CT_place_SECRET`: contribution `-0.003903`
- `lag_00__kill_diff_last_3s`: contribution `-0.003486`
- `lag_00__T_damage_last_5s`: contribution `-0.003319`
- `lag_13__CT_shots_fired_sum`: contribution `-0.003143`

Top utility-only movements:
- `lag_00__CT5__molly`: contribution `-0.001316`
