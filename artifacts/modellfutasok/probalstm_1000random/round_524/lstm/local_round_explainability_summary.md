# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-saw-bo3-tIR5RlOpBrnlpEe6MBVyNd/heroic-vs-saw-m2-train.csv`
- round_num: `6`

## Largest probability jumps

- tick `40082`, seconds `74.50`, LSTM `0.8288`, delta `+0.2648`
- tick `36562`, seconds `19.50`, LSTM `0.5632`, delta `-0.1454`
- tick `36530`, seconds `19.00`, LSTM `0.7087`, delta `+0.0933`
- tick `40242`, seconds `77.00`, LSTM `0.9697`, delta `+0.0773`
- tick `37010`, seconds `26.50`, LSTM `0.6597`, delta `+0.0772`
- tick `36178`, seconds `13.50`, LSTM `0.5985`, delta `+0.0401`
- tick `40114`, seconds `75.00`, LSTM `0.8672`, delta `+0.0383`
- tick `38226`, seconds `45.50`, LSTM `0.7054`, delta `+0.0362`
- tick `37874`, seconds `40.00`, LSTM `0.6683`, delta `+0.0318`
- tick `39666`, seconds `68.00`, LSTM `0.6653`, delta `-0.0284`

## Top 15 local ridge features

- `lag_04__CT_place_BACKOFB`: coefficient `0.004567`, |coef| `0.004567`
- `lag_04__CT_place_LONGDOG`: coefficient `-0.004293`, |coef| `0.004293`
- `lag_00__CT_kills_last_3s`: coefficient `0.002677`, |coef| `0.002677`
- `lag_01__T3__duck_amount`: coefficient `-0.002515`, |coef| `0.002515`
- `lag_00__kill_diff_last_3s`: coefficient `0.002445`, |coef| `0.002445`
- `lag_06__CT_place_CONNECTOR`: coefficient `-0.002408`, |coef| `0.002408`
- `lag_01__T1__duck_amount`: coefficient `0.002374`, |coef| `0.002374`
- `lag_00__CT_damage_last_5s`: coefficient `0.002033`, |coef| `0.002033`
- `lag_01__T5__duck_amount`: coefficient `-0.002009`, |coef| `0.002009`
- `lag_15__T_place_BACKOFB`: coefficient `-0.002003`, |coef| `0.002003`
- `lag_00__damage_diff_last_5s`: coefficient `0.001986`, |coef| `0.001986`
- `lag_04__T5__duck_amount`: coefficient `0.001962`, |coef| `0.001962`
- `lag_00__T3__alive`: coefficient `-0.001941`, |coef| `0.001941`
- `lag_02__T3__duck_amount`: coefficient `0.001938`, |coef| `0.001938`
- `lag_05__T_B_site_active_infernos`: coefficient `-0.001932`, |coef| `0.001932`

## Top 10 utility ridge features

- `lag_05__T_B_site_active_infernos`: coefficient `-0.001932` (lowers CT win probability)
- `lag_02__T5__molly`: coefficient `-0.001702` (lowers CT win probability)
- `lag_05__T_active_infernos`: coefficient `-0.001430` (lowers CT win probability)
- `lag_05__active_infernos_total`: coefficient `-0.000975` (lowers CT win probability)
- `lag_00__T_B_site_active_infernos`: coefficient `0.000830` (raises CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `0.000754` (raises CT win probability)
- `lag_02__CT_utility_damage_last_5s`: coefficient `0.000740` (raises CT win probability)
- `lag_02__T5__utility_total`: coefficient `-0.000657` (lowers CT win probability)
- `lag_02__utility_damage_diff_last_5s`: coefficient `0.000612` (raises CT win probability)
- `lag_00__T_active_infernos`: coefficient `0.000611` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_04__CT_place_BACKOFB`: coefficient `0.004567` (raises CT win probability)
- `lag_04__CT_place_LONGDOG`: coefficient `-0.004293` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002677` (raises CT win probability)
- `lag_01__T3__duck_amount`: coefficient `-0.002515` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002445` (raises CT win probability)
- `lag_06__CT_place_CONNECTOR`: coefficient `-0.002408` (lowers CT win probability)
- `lag_01__T1__duck_amount`: coefficient `0.002374` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002033` (raises CT win probability)
- `lag_01__T5__duck_amount`: coefficient `-0.002009` (lowers CT win probability)
- `lag_15__T_place_BACKOFB`: coefficient `-0.002003` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `40082`, seconds `74.50`, LSTM delta `+0.2648`

Top all feature movements:
- `lag_04__CT_place_LONGDOG`: contribution `+0.028002`
- `lag_04__CT_place_BACKOFB`: contribution `+0.026075`
- `lag_01__T3__duck_amount`: contribution `+0.009481`
- `lag_01__T1__duck_amount`: contribution `+0.009295`
- `lag_06__CT_place_CONNECTOR`: contribution `+0.008611`

Top utility-only movements:
- `lag_05__T_B_site_active_infernos`: contribution `+0.005463`

### tick `36562`, seconds `19.50`, LSTM delta `-0.1454`

Top all feature movements:
- `lag_00__CT_place_TMAIN`: contribution `-0.011286`
- `lag_10__CT_place_TMAIN`: contribution `-0.008078`
- `lag_07__T2__is_scoped`: contribution `-0.007122`
- `lag_07__CT_place_ELECTRICALBOX`: contribution `-0.006815`
- `lag_00__kill_diff_last_3s`: contribution `-0.005885`

Top utility-only movements:
- `lag_02__CT_utility_damage_last_5s`: contribution `-0.004727`
- `lag_00__CT2__flash_duration`: contribution `-0.003844`
- `lag_09__T4__flash_duration`: contribution `-0.003593`
- `lag_02__utility_damage_diff_last_5s`: contribution `-0.003204`
- `lag_12__CT_utility_damage_last_5s`: contribution `-0.003108`

### tick `36530`, seconds `19.00`, LSTM delta `+0.0933`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.007729`
- `lag_15__CT_place_ELECTRICALBOX`: contribution `+0.007403`
- `lag_00__kill_diff_last_3s`: contribution `+0.005885`
- `lag_06__CT_place_ELECTRICALBOX`: contribution `+0.005276`
- `lag_00__CT_shots_fired_sum`: contribution `+0.004050`

Top utility-only movements:
- `lag_08__T4__flash_duration`: contribution `+0.003339`

### tick `40242`, seconds `77.00`, LSTM delta `+0.0773`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.007729`
- `lag_01__T5__duck_amount`: contribution `-0.007630`
- `lag_00__kill_diff_last_3s`: contribution `+0.005885`
- `lag_09__CT_place_BACKOFB`: contribution `+0.005480`
- `lag_00__CT_shots_fired_sum`: contribution `+0.004050`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `37010`, seconds `26.50`, LSTM delta `+0.0772`

Top all feature movements:
- `lag_14__CT_place_TMAIN`: contribution `+0.009098`
- `lag_00__CT_kills_last_3s`: contribution `+0.007729`
- `lag_00__kill_diff_last_3s`: contribution `+0.005885`
- `lag_10__T_place_DUMPSTER`: contribution `+0.005385`
- `lag_00__CT_shots_fired_sum`: contribution `-0.004050`

Top utility-only movements:
- `lag_13__CT1__flash_duration`: contribution `+0.002025`
- `lag_14__CT2__flash_duration`: contribution `+0.001196`
- `lag_00__T4__flash`: contribution `+0.000974`
