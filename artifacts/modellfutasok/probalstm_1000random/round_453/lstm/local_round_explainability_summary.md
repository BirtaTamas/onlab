# Local Round Explainability

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-the-mongolz-vs-natus-vincere-bo3-C0GZxMhpGHBr28LeyjgICZ/the-mongolz-vs-natus-vincere-m1-mirage.csv`
- round_num: `15`

## Largest probability jumps

- tick `121447`, seconds `76.00`, LSTM `0.8618`, delta `+0.2049`
- tick `119239`, seconds `41.50`, LSTM `0.6426`, delta `+0.1807`
- tick `117671`, seconds `17.00`, LSTM `0.3858`, delta `-0.0579`
- tick `121479`, seconds `76.50`, LSTM `0.9140`, delta `+0.0522`
- tick `121607`, seconds `78.50`, LSTM `0.9629`, delta `+0.0398`
- tick `122055`, seconds `85.50`, LSTM `0.9730`, delta `+0.0338`
- tick `119559`, seconds `46.50`, LSTM `0.6512`, delta `-0.0337`
- tick `117831`, seconds `19.50`, LSTM `0.3598`, delta `+0.0329`
- tick `119335`, seconds `43.00`, LSTM `0.6901`, delta `+0.0327`
- tick `118087`, seconds `23.50`, LSTM `0.4074`, delta `+0.0327`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.002204`, |coef| `0.002204`
- `lag_04__CT3__flash_duration`: coefficient `0.001894`, |coef| `0.001894`
- `lag_00__kill_diff_last_3s`: coefficient `0.001838`, |coef| `0.001838`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001670`, |coef| `0.001670`
- `lag_00__CT_damage_last_5s`: coefficient `0.001667`, |coef| `0.001667`
- `lag_12__CT4__is_walking`: coefficient `-0.001633`, |coef| `0.001633`
- `lag_04__T3__flash_duration`: coefficient `0.001633`, |coef| `0.001633`
- `lag_00__CT_place_SCAFFOLDING`: coefficient `-0.001608`, |coef| `0.001608`
- `lag_02__T_A_site_active_infernos`: coefficient `0.001491`, |coef| `0.001491`
- `lag_11__CT_place_UNDERPASS`: coefficient `0.001473`, |coef| `0.001473`
- `lag_03__T_A_site_active_infernos`: coefficient `0.001458`, |coef| `0.001458`
- `lag_02__T_place_CONNECTOR`: coefficient `-0.001411`, |coef| `0.001411`
- `lag_00__damage_diff_last_5s`: coefficient `0.001404`, |coef| `0.001404`
- `lag_00__T3__is_walking`: coefficient `-0.001370`, |coef| `0.001370`
- `lag_14__CT_place_JUNGLE`: coefficient `-0.001366`, |coef| `0.001366`

## Top 10 utility ridge features

- `lag_04__CT3__flash_duration`: coefficient `0.001894` (raises CT win probability)
- `lag_04__T3__flash_duration`: coefficient `0.001633` (raises CT win probability)
- `lag_02__T_A_site_active_infernos`: coefficient `0.001491` (raises CT win probability)
- `lag_03__T_A_site_active_infernos`: coefficient `0.001458` (raises CT win probability)
- `lag_00__T2__molly`: coefficient `-0.001054` (lowers CT win probability)
- `lag_02__T_active_infernos`: coefficient `0.001031` (raises CT win probability)
- `lag_03__T_active_infernos`: coefficient `0.001006` (raises CT win probability)
- `lag_15__T3__smoke`: coefficient `-0.000940` (lowers CT win probability)
- `lag_07__T5__smoke`: coefficient `-0.000931` (lowers CT win probability)
- `lag_04__T5__molly`: coefficient `-0.000908` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.002204` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001838` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001670` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001667` (raises CT win probability)
- `lag_12__CT4__is_walking`: coefficient `-0.001633` (lowers CT win probability)
- `lag_00__CT_place_SCAFFOLDING`: coefficient `-0.001608` (lowers CT win probability)
- `lag_11__CT_place_UNDERPASS`: coefficient `0.001473` (raises CT win probability)
- `lag_02__T_place_CONNECTOR`: coefficient `-0.001411` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001404` (raises CT win probability)
- `lag_00__T3__is_walking`: coefficient `-0.001370` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `121447`, seconds `76.00`, LSTM delta `+0.2049`

Top all feature movements:
- `lag_04__CT3__flash_duration`: contribution `+0.013146`
- `lag_04__T3__flash_duration`: contribution `+0.011008`
- `lag_14__CT_place_JUNGLE`: contribution `+0.008765`
- `lag_11__CT_place_UNDERPASS`: contribution `+0.008543`
- `lag_02__T_place_CONNECTOR`: contribution `+0.006834`

Top utility-only movements:
- `lag_04__CT3__flash_duration`: contribution `+0.013146`
- `lag_04__T3__flash_duration`: contribution `+0.011008`
- `lag_03__T_A_site_active_infernos`: contribution `+0.004340`
- `lag_01__CT_A_site_active_infernos`: contribution `+0.002897`

### tick `119239`, seconds `41.50`, LSTM delta `+0.1807`

Top all feature movements:
- `lag_07__CT_place_JUNGLE`: contribution `+0.008475`
- `lag_00__CT_kills_last_3s`: contribution `+0.006364`
- `lag_00__CT_shots_fired_sum`: contribution `+0.005799`
- `lag_05__CT_place_CATWALK`: contribution `+0.004585`
- `lag_13__T_place_PALACEALLEY`: contribution `+0.004550`

Top utility-only movements:
- `lag_02__T_A_site_active_infernos`: contribution `+0.004438`
- `lag_00__T2__molly`: contribution `+0.002349`

### tick `117671`, seconds `17.00`, LSTM delta `-0.0579`

Top all feature movements:
- `lag_00__CT_place_SCAFFOLDING`: contribution `-0.033560`
- `lag_07__CT_place_JUNGLE`: contribution `+0.016949`
- `lag_11__CT_place_CATWALK`: contribution `-0.003430`
- `lag_00__T1__duck_amount`: contribution `-0.002307`
- `lag_15__CT5__shots_fired`: contribution `-0.002179`

Top utility-only movements:
- `lag_10__CT_A_site_active_infernos`: contribution `-0.001590`
- `lag_10__T3__flash_duration`: contribution `-0.001173`

### tick `121479`, seconds `76.50`, LSTM delta `+0.0522`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.005799`
- `lag_12__CT_place_UNDERPASS`: contribution `+0.004195`
- `lag_00__T_place_PALACEINTERIOR`: contribution `-0.003823`
- `lag_01__CT_shots_fired_sum`: contribution `+0.003780`
- `lag_05__CT3__flash_duration`: contribution `+0.003749`

Top utility-only movements:
- `lag_05__CT3__flash_duration`: contribution `+0.003749`
- `lag_05__T3__flash_duration`: contribution `+0.003272`

### tick `121607`, seconds `78.50`, LSTM delta `+0.0398`

Top all feature movements:
- `lag_02__CT_shots_fired_sum`: contribution `-0.010903`
- `lag_00__CT_kills_last_3s`: contribution `+0.006364`
- `lag_06__CT2__duck_amount`: contribution `-0.004649`
- `lag_00__kill_diff_last_3s`: contribution `+0.004423`
- `lag_00__CT_shots_fired_sum`: contribution `+0.003480`

Top utility-only movements:
- `lag_09__CT3__flash_duration`: contribution `+0.002133`
