# Local Round Explainability

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-3dmax-vs-rare-atom-bo3-DWQZo2y3LVjgpuOkyCDf4V/3dmax-vs-rare-atom-m2-ancient.csv`
- round_num: `4`

## Largest probability jumps

- tick `22379`, seconds `70.00`, LSTM `0.9462`, delta `+0.2290`
- tick `21867`, seconds `62.00`, LSTM `0.8417`, delta `+0.1947`
- tick `22283`, seconds `68.50`, LSTM `0.7311`, delta `-0.1868`
- tick `21835`, seconds `61.50`, LSTM `0.6470`, delta `+0.1454`
- tick `18891`, seconds `15.50`, LSTM `0.4752`, delta `-0.0774`
- tick `18955`, seconds `16.50`, LSTM `0.5137`, delta `+0.0733`
- tick `21899`, seconds `62.50`, LSTM `0.8946`, delta `+0.0529`
- tick `21003`, seconds `48.50`, LSTM `0.4740`, delta `-0.0368`
- tick `18923`, seconds `16.00`, LSTM `0.4404`, delta `-0.0348`
- tick `21323`, seconds `53.50`, LSTM `0.4973`, delta `+0.0318`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003498`, |coef| `0.003498`
- `lag_00__CT_kills_last_3s`: coefficient `0.003361`, |coef| `0.003361`
- `lag_00__damage_diff_last_5s`: coefficient `0.002818`, |coef| `0.002818`
- `lag_00__CT_damage_last_5s`: coefficient `0.002173`, |coef| `0.002173`
- `lag_01__damage_diff_last_5s`: coefficient `0.001872`, |coef| `0.001872`
- `lag_01__CT_damage_last_5s`: coefficient `0.001861`, |coef| `0.001861`
- `lag_11__CT_place_TSIDEUPPER`: coefficient `0.001854`, |coef| `0.001854`
- `lag_07__T_place_TSIDELOWER`: coefficient `-0.001803`, |coef| `0.001803`
- `lag_01__CT_kills_last_3s`: coefficient `0.001764`, |coef| `0.001764`
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.001751`, |coef| `0.001751`
- `lag_02__T_place_TSIDELOWER`: coefficient `-0.001729`, |coef| `0.001729`
- `lag_00__CT4__is_scoped`: coefficient `-0.001611`, |coef| `0.001611`
- `lag_01__T_place_RAMP`: coefficient `0.001573`, |coef| `0.001573`
- `lag_00__T2__shots_fired`: coefficient `0.001544`, |coef| `0.001544`
- `lag_01__T_place_TSIDELOWER`: coefficient `-0.001502`, |coef| `0.001502`

## Top 10 utility ridge features

- `lag_02__T_B_site_active_infernos`: coefficient `0.001257` (raises CT win probability)
- `lag_12__T4__smoke`: coefficient `-0.001208` (lowers CT win probability)
- `lag_01__T_B_site_active_infernos`: coefficient `0.001205` (raises CT win probability)
- `lag_04__CT2__smoke`: coefficient `-0.001183` (lowers CT win probability)
- `lag_01__T4__molly`: coefficient `-0.001162` (lowers CT win probability)
- `lag_14__T5__smoke`: coefficient `-0.001148` (lowers CT win probability)
- `lag_10__CT_B_site_active_infernos`: coefficient `-0.001146` (lowers CT win probability)
- `lag_11__CT_B_site_active_infernos`: coefficient `-0.001106` (lowers CT win probability)
- `lag_00__T4__molly`: coefficient `-0.001039` (lowers CT win probability)
- `lag_13__T5__smoke`: coefficient `-0.001035` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003498` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003361` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002818` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002173` (raises CT win probability)
- `lag_01__damage_diff_last_5s`: coefficient `0.001872` (raises CT win probability)
- `lag_01__CT_damage_last_5s`: coefficient `0.001861` (raises CT win probability)
- `lag_11__CT_place_TSIDEUPPER`: coefficient `0.001854` (raises CT win probability)
- `lag_07__T_place_TSIDELOWER`: coefficient `-0.001803` (lowers CT win probability)
- `lag_01__CT_kills_last_3s`: coefficient `0.001764` (raises CT win probability)
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.001751` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `22379`, seconds `70.00`, LSTM delta `+0.2290`

Top all feature movements:
- `lag_11__CT_place_TSIDEUPPER`: contribution `+0.013935`
- `lag_00__CT_kills_last_3s`: contribution `+0.009705`
- `lag_00__kill_diff_last_3s`: contribution `+0.008419`
- `lag_03__CT_place_TSIDEUPPER`: contribution `+0.008139`
- `lag_07__T_place_TSIDELOWER`: contribution `+0.006759`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `21867`, seconds `62.00`, LSTM delta `+0.1947`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.009705`
- `lag_00__kill_diff_last_3s`: contribution `+0.008419`
- `lag_02__T_place_TSIDELOWER`: contribution `+0.006482`
- `lag_01__damage_diff_last_5s`: contribution `+0.006336`
- `lag_01__CT_damage_last_5s`: contribution `+0.006086`

Top utility-only movements:
- `lag_02__T_B_site_active_infernos`: contribution `+0.003554`
- `lag_12__CT_B_site_active_infernos`: contribution `+0.003382`

### tick `22283`, seconds `68.50`, LSTM delta `-0.1868`

Top all feature movements:
- `lag_00__CT_place_TSIDEUPPER`: contribution `-0.013162`
- `lag_00__kill_diff_last_3s`: contribution `-0.008419`
- `lag_08__CT_place_TSIDEUPPER`: contribution `-0.007606`
- `lag_00__damage_diff_last_5s`: contribution `-0.006358`
- `lag_01__T_place_RAMP`: contribution `-0.005565`

Top utility-only movements:
- `lag_01__T_B_site_active_infernos`: contribution `-0.003406`

### tick `21835`, seconds `61.50`, LSTM delta `+0.1454`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.009705`
- `lag_00__damage_diff_last_5s`: contribution `+0.009537`
- `lag_00__kill_diff_last_3s`: contribution `+0.008419`
- `lag_00__CT_damage_last_5s`: contribution `+0.007105`
- `lag_01__T_place_TSIDELOWER`: contribution `+0.005630`

Top utility-only movements:
- `lag_11__CT_B_site_active_infernos`: contribution `+0.003799`
- `lag_01__T_B_site_active_infernos`: contribution `+0.003406`

### tick `18891`, seconds `15.50`, LSTM delta `-0.0774`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.008419`
- `lag_10__CT_B_site_active_infernos`: contribution `-0.007876`
- `lag_00__damage_diff_last_5s`: contribution `-0.004832`
- `lag_00__T3__flash_duration`: contribution `-0.003641`
- `lag_10__CT_active_infernos`: contribution `-0.003588`

Top utility-only movements:
- `lag_10__CT_B_site_active_infernos`: contribution `-0.007876`
- `lag_00__T3__flash_duration`: contribution `-0.003641`
- `lag_10__CT_active_infernos`: contribution `-0.003588`
- `lag_02__T_B_site_active_infernos`: contribution `-0.003554`
- `lag_06__T3__flash_duration`: contribution `-0.002825`
