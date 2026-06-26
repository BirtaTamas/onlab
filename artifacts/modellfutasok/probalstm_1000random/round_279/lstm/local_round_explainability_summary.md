# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-3dmax-vs-m80-bo3-DeIrLPYSKhgd10M8zQmUUV/3dmax-vs-m80-m2-train.csv`
- round_num: `16`

## Largest probability jumps

- tick `122274`, seconds `39.50`, LSTM `0.8449`, delta `+0.2727`
- tick `122466`, seconds `42.50`, LSTM `0.9506`, delta `+0.1038`
- tick `121986`, seconds `35.00`, LSTM `0.4449`, delta `-0.0780`
- tick `122242`, seconds `39.00`, LSTM `0.5722`, delta `+0.0607`
- tick `121858`, seconds `33.00`, LSTM `0.5093`, delta `-0.0551`
- tick `122018`, seconds `35.50`, LSTM `0.4873`, delta `+0.0424`
- tick `121474`, seconds `27.00`, LSTM `0.5617`, delta `-0.0423`
- tick `121186`, seconds `22.50`, LSTM `0.6495`, delta `+0.0398`
- tick `120002`, seconds `4.00`, LSTM `0.6437`, delta `-0.0310`
- tick `120450`, seconds `11.00`, LSTM `0.6450`, delta `-0.0282`

## Top 15 local ridge features

- `lag_06__CT_place_TMAIN`: coefficient `-0.002440`, |coef| `0.002440`
- `lag_00__CT_place_ELECTRICALBOX`: coefficient `0.002150`, |coef| `0.002150`
- `lag_01__CT_place_LONGDOG`: coefficient `-0.001325`, |coef| `0.001325`
- `lag_02__T_place_DUMPSTER`: coefficient `-0.001323`, |coef| `0.001323`
- `lag_07__CT_place_TMAIN`: coefficient `-0.001311`, |coef| `0.001311`
- `lag_02__T_flashes_last_5s`: coefficient `-0.001232`, |coef| `0.001232`
- `lag_09__T_shots_fired_sum`: coefficient `0.001205`, |coef| `0.001205`
- `lag_01__T_place_BACKOFB`: coefficient `-0.001144`, |coef| `0.001144`
- `lag_00__damage_diff_last_5s`: coefficient `0.001136`, |coef| `0.001136`
- `lag_06__T2__duck_amount`: coefficient `-0.001127`, |coef| `0.001127`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001118`, |coef| `0.001118`
- `lag_08__T_shots_fired_sum`: coefficient `-0.001080`, |coef| `0.001080`
- `lag_00__T_place_BACKOFB`: coefficient `-0.001079`, |coef| `0.001079`
- `lag_00__kill_diff_last_3s`: coefficient `0.001053`, |coef| `0.001053`
- `lag_12__CT_place_TMAIN`: coefficient `-0.001027`, |coef| `0.001027`

## Top 10 utility ridge features

- `lag_02__T_flashes_last_5s`: coefficient `-0.001232` (lowers CT win probability)
- `lag_01__T_flashes_last_5s`: coefficient `-0.000781` (lowers CT win probability)
- `lag_08__T_flashes_last_5s`: coefficient `0.000669` (raises CT win probability)
- `lag_04__T1__smoke`: coefficient `-0.000630` (lowers CT win probability)
- `lag_07__T_flashes_last_5s`: coefficient `-0.000590` (lowers CT win probability)
- `lag_13__CT5__smoke`: coefficient `-0.000580` (lowers CT win probability)
- `lag_05__T_flashes_last_5s`: coefficient `-0.000563` (lowers CT win probability)
- `lag_14__T_flashes_last_5s`: coefficient `-0.000553` (lowers CT win probability)
- `lag_13__CT5__utility_total`: coefficient `-0.000524` (lowers CT win probability)
- `lag_01__T4__smoke`: coefficient `-0.000499` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_06__CT_place_TMAIN`: coefficient `-0.002440` (lowers CT win probability)
- `lag_00__CT_place_ELECTRICALBOX`: coefficient `0.002150` (raises CT win probability)
- `lag_01__CT_place_LONGDOG`: coefficient `-0.001325` (lowers CT win probability)
- `lag_02__T_place_DUMPSTER`: coefficient `-0.001323` (lowers CT win probability)
- `lag_07__CT_place_TMAIN`: coefficient `-0.001311` (lowers CT win probability)
- `lag_09__T_shots_fired_sum`: coefficient `0.001205` (raises CT win probability)
- `lag_01__T_place_BACKOFB`: coefficient `-0.001144` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001136` (raises CT win probability)
- `lag_06__T2__duck_amount`: coefficient `-0.001127` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001118` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `122274`, seconds `39.50`, LSTM delta `+0.2727`

Top all feature movements:
- `lag_06__CT_place_TMAIN`: contribution `+0.027041`
- `lag_00__CT_place_ELECTRICALBOX`: contribution `+0.024993`
- `lag_02__T_place_DUMPSTER`: contribution `+0.012035`
- `lag_01__CT_place_LONGDOG`: contribution `+0.008641`
- `lag_09__T_shots_fired_sum`: contribution `+0.008132`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `122466`, seconds `42.50`, LSTM delta `+0.1038`

Top all feature movements:
- `lag_12__CT_place_TMAIN`: contribution `+0.011378`
- `lag_05__CT_place_ELECTRICALBOX`: contribution `+0.010878`
- `lag_08__T_place_DUMPSTER`: contribution `+0.009181`
- `lag_06__CT_place_ELECTRICALBOX`: contribution `+0.008785`
- `lag_07__CT_place_LONGDOG`: contribution `+0.004664`

Top utility-only movements:
- `lag_01__CT_B_site_active_infernos`: contribution `+0.001696`

### tick `121986`, seconds `35.00`, LSTM delta `-0.0780`

Top all feature movements:
- `lag_07__CT_place_TMAIN`: contribution `-0.014524`
- `lag_08__T_place_DUMPSTER`: contribution `-0.009181`
- `lag_08__T_flashes_last_5s`: contribution `-0.006058`
- `lag_00__T_shots_fired_sum`: contribution `-0.005370`
- `lag_06__T2__duck_amount`: contribution `-0.004308`

Top utility-only movements:
- `lag_08__T_flashes_last_5s`: contribution `-0.006058`

### tick `122242`, seconds `39.00`, LSTM delta `+0.0607`

Top all feature movements:
- `lag_15__CT_place_TMAIN`: contribution `+0.009708`
- `lag_05__CT_place_TMAIN`: contribution `+0.009282`
- `lag_08__T_shots_fired_sum`: contribution `-0.007290`
- `lag_01__T_place_DUMPSTER`: contribution `+0.004280`
- `lag_00__CT_shots_fired_sum`: contribution `+0.003107`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `121858`, seconds `33.00`, LSTM delta `-0.0551`

Top all feature movements:
- `lag_03__CT_place_TMAIN`: contribution `-0.007406`
- `lag_04__T_place_DUMPSTER`: contribution `-0.006292`
- `lag_14__T_flashes_last_5s`: contribution `-0.005013`
- `lag_12__CT_place_BACKOFB`: contribution `-0.002628`
- `lag_00__damage_diff_last_5s`: contribution `-0.002562`

Top utility-only movements:
- `lag_14__T_flashes_last_5s`: contribution `-0.005013`
- `lag_04__T_flashes_last_5s`: contribution `-0.001303`
