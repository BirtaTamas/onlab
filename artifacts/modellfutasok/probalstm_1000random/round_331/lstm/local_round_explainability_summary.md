# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-betboom-vs-nemiga-train-khA7BVyAiKBjWcyTrFzube/betboom-vs-nemiga-train.csv`
- round_num: `11`

## Largest probability jumps

- tick `91235`, seconds `44.50`, LSTM `0.2199`, delta `-0.2936`
- tick `91107`, seconds `42.50`, LSTM `0.4510`, delta `+0.1118`
- tick `91267`, seconds `45.00`, LSTM `0.1204`, delta `-0.0995`
- tick `91011`, seconds `41.00`, LSTM `0.3619`, delta `-0.0634`
- tick `89475`, seconds `17.00`, LSTM `0.5463`, delta `+0.0542`
- tick `91043`, seconds `41.50`, LSTM `0.3106`, delta `-0.0514`
- tick `90851`, seconds `38.50`, LSTM `0.5176`, delta `-0.0473`
- tick `91139`, seconds `43.00`, LSTM `0.4928`, delta `+0.0418`
- tick `88899`, seconds `8.00`, LSTM `0.5385`, delta `+0.0394`
- tick `90979`, seconds `40.50`, LSTM `0.4253`, delta `-0.0341`

## Top 15 local ridge features

- `lag_00__CT_place_LONGDOG`: coefficient `0.001821`, |coef| `0.001821`
- `lag_07__CT3__is_scoped`: coefficient `0.001533`, |coef| `0.001533`
- `lag_04__bomb_events_last_5s`: coefficient `-0.001520`, |coef| `0.001520`
- `lag_05__T5__duck_amount`: coefficient `0.001475`, |coef| `0.001475`
- `lag_04__T_place_BACKOFB`: coefficient `0.001459`, |coef| `0.001459`
- `lag_00__bomb_events_last_5s`: coefficient `0.001407`, |coef| `0.001407`
- `lag_01__CT_place_CONNECTOR`: coefficient `-0.001406`, |coef| `0.001406`
- `lag_00__damage_diff_last_5s`: coefficient `0.001391`, |coef| `0.001391`
- `lag_15__CT5__is_walking`: coefficient `0.001355`, |coef| `0.001355`
- `lag_03__CT3__is_walking`: coefficient `0.001319`, |coef| `0.001319`
- `lag_07__CT5__is_walking`: coefficient `0.001222`, |coef| `0.001222`
- `lag_00__kill_diff_last_3s`: coefficient `0.001204`, |coef| `0.001204`
- `lag_00__T_flashed_players`: coefficient `0.001192`, |coef| `0.001192`
- `lag_14__T1__is_walking`: coefficient `0.001182`, |coef| `0.001182`
- `lag_03__T_flashed_players`: coefficient `-0.001174`, |coef| `0.001174`

## Top 10 utility ridge features

- `lag_05__T_B_site_active_infernos`: coefficient `-0.001040` (lowers CT win probability)
- `lag_00__CT4__smoke`: coefficient `0.001020` (raises CT win probability)
- `lag_05__T_A_site_active_infernos`: coefficient `-0.001016` (lowers CT win probability)
- `lag_12__CT2__molly`: coefficient `0.000977` (raises CT win probability)
- `lag_00__CT4__utility_total`: coefficient `0.000947` (raises CT win probability)
- `lag_00__T_B_site_active_smokes`: coefficient `-0.000921` (lowers CT win probability)
- `lag_12__CT2__smoke`: coefficient `0.000879` (raises CT win probability)
- `lag_08__T5__molly`: coefficient `0.000857` (raises CT win probability)
- `lag_00__CT4__flash`: coefficient `0.000810` (raises CT win probability)
- `lag_00__T_A_site_active_smokes`: coefficient `-0.000796` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_LONGDOG`: coefficient `0.001821` (raises CT win probability)
- `lag_07__CT3__is_scoped`: coefficient `0.001533` (raises CT win probability)
- `lag_04__bomb_events_last_5s`: coefficient `-0.001520` (lowers CT win probability)
- `lag_05__T5__duck_amount`: coefficient `0.001475` (raises CT win probability)
- `lag_04__T_place_BACKOFB`: coefficient `0.001459` (raises CT win probability)
- `lag_00__bomb_events_last_5s`: coefficient `0.001407` (raises CT win probability)
- `lag_01__CT_place_CONNECTOR`: coefficient `-0.001406` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001391` (raises CT win probability)
- `lag_15__CT5__is_walking`: coefficient `0.001355` (raises CT win probability)
- `lag_03__CT3__is_walking`: coefficient `0.001319` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `91235`, seconds `44.50`, LSTM delta `-0.2936`

Top all feature movements:
- `lag_00__CT_place_LONGDOG`: contribution `-0.011879`
- `lag_04__T_place_BACKOFB`: contribution `-0.007835`
- `lag_07__CT3__is_scoped`: contribution `-0.006974`
- `lag_05__T5__duck_amount`: contribution `-0.005601`
- `lag_01__CT_place_CONNECTOR`: contribution `-0.005029`

Top utility-only movements:
- `lag_05__T_A_site_active_infernos`: contribution `-0.003024`
- `lag_05__T_B_site_active_infernos`: contribution `-0.002939`

### tick `91107`, seconds `42.50`, LSTM delta `+0.1118`

Top all feature movements:
- `lag_01__T5__duck_amount`: contribution `+0.004216`
- `lag_03__CT3__is_scoped`: contribution `+0.003957`
- `lag_00__T_place_BACKOFB`: contribution `+0.003784`
- `lag_03__CT3__is_walking`: contribution `+0.003150`
- `lag_02__bomb_events_last_5s`: contribution `+0.003015`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `91267`, seconds `45.00`, LSTM delta `-0.0995`

Top all feature movements:
- `lag_01__CT_place_LONGDOG`: contribution `-0.005492`
- `lag_01__T5__is_scoped`: contribution `+0.004725`
- `lag_05__T_place_BACKOFB`: contribution `-0.004664`
- `lag_02__CT3__is_scoped`: contribution `-0.003534`
- `lag_08__CT3__is_scoped`: contribution `-0.003197`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `91011`, seconds `41.00`, LSTM delta `-0.0634`

Top all feature movements:
- `lag_00__T5__duck_amount`: contribution `-0.003423`
- `lag_15__CT5__is_walking`: contribution `-0.003247`
- `lag_00__CT3__is_scoped`: contribution `-0.002907`
- `lag_10__bomb_events_last_5s`: contribution `-0.001979`
- `lag_09__bomb_events_last_5s`: contribution `-0.001933`

Top utility-only movements:
- `lag_05__CT2__molly`: contribution `-0.001334`

### tick `89475`, seconds `17.00`, LSTM delta `+0.0542`

Top all feature movements:
- `lag_05__T_place_BACKOFB`: contribution `+0.004664`
- `lag_15__CT_place_ELECTRICALBOX`: contribution `+0.004504`
- `lag_07__T_place_TSTAIRS`: contribution `+0.003939`
- `lag_05__T_place_TSTAIRS`: contribution `+0.003620`
- `lag_13__CT_place_BACKOFB`: contribution `+0.002645`

Top utility-only movements:
- `lag_00__T_utility_damage_last_5s`: contribution `+0.001679`
- `lag_10__T2__flash_duration`: contribution `+0.001662`
- `lag_14__T_active_infernos`: contribution `+0.001580`
- `lag_10__T_utility_damage_last_5s`: contribution `+0.001308`
