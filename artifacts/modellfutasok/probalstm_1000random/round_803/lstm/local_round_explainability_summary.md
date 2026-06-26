# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-eternal-fire-vs-natus-vincere-bo3-TFptrqwLQ_nOvi5zixIc9R/eternal-fire-vs-natus-vincere-m2-dust2.csv`
- round_num: `18`

## Largest probability jumps

- tick `145846`, seconds `30.50`, LSTM `0.1148`, delta `-0.1487`
- tick `145814`, seconds `30.00`, LSTM `0.2635`, delta `-0.0750`
- tick `145910`, seconds `31.50`, LSTM `0.0378`, delta `-0.0725`
- tick `144694`, seconds `12.50`, LSTM `0.2765`, delta `-0.0376`
- tick `143926`, seconds `0.50`, LSTM `0.3057`, delta `-0.0327`
- tick `144662`, seconds `12.00`, LSTM `0.3142`, delta `+0.0313`
- tick `144566`, seconds `10.50`, LSTM `0.2789`, delta `+0.0281`
- tick `144246`, seconds `5.50`, LSTM `0.2755`, delta `-0.0245`
- tick `143990`, seconds `1.50`, LSTM `0.3003`, delta `-0.0235`
- tick `145366`, seconds `23.00`, LSTM `0.3089`, delta `+0.0211`

## Top 15 local ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.001437`, |coef| `0.001437`
- `lag_00__T_place_SHORTSTAIRS`: coefficient `-0.001291`, |coef| `0.001291`
- `lag_00__CT_place_SHORTSTAIRS`: coefficient `0.001273`, |coef| `0.001273`
- `lag_10__CT_place_SHORTSTAIRS`: coefficient `-0.001135`, |coef| `0.001135`
- `lag_05__CT_place_EXTENDEDA`: coefficient `-0.000991`, |coef| `0.000991`
- `lag_05__CT_place_UNDERA`: coefficient `0.000970`, |coef| `0.000970`
- `lag_00__T_kills_last_3s`: coefficient `-0.000947`, |coef| `0.000947`
- `lag_09__CT_place_SHORTSTAIRS`: coefficient `-0.000899`, |coef| `0.000899`
- `lag_02__T_shots_fired_sum`: coefficient `-0.000808`, |coef| `0.000808`
- `lag_00__T_damage_last_5s`: coefficient `-0.000769`, |coef| `0.000769`
- `lag_10__CT_place_EXTENDEDA`: coefficient `0.000741`, |coef| `0.000741`
- `lag_00__CT5__is_walking`: coefficient `0.000735`, |coef| `0.000735`
- `lag_14__T_place_CATWALK`: coefficient `-0.000722`, |coef| `0.000722`
- `lag_00__kill_diff_last_3s`: coefficient `0.000720`, |coef| `0.000720`
- `lag_04__CT_place_EXTENDEDA`: coefficient `-0.000708`, |coef| `0.000708`

## Top 10 utility ridge features

- `lag_09__CT_smokes_last_5s`: coefficient `-0.000674` (lowers CT win probability)
- `lag_06__CT1__smoke`: coefficient `0.000522` (raises CT win probability)
- `lag_08__CT4__smoke`: coefficient `-0.000467` (lowers CT win probability)
- `lag_00__CT5__flash`: coefficient `0.000447` (raises CT win probability)
- `lag_13__CT_smokes_last_5s`: coefficient `0.000430` (raises CT win probability)
- `lag_07__CT4__smoke`: coefficient `-0.000424` (lowers CT win probability)
- `lag_07__CT_smokes_last_5s`: coefficient `-0.000405` (lowers CT win probability)
- `lag_04__CT5__smoke`: coefficient `-0.000397` (lowers CT win probability)
- `lag_05__CT1__smoke`: coefficient `0.000395` (raises CT win probability)
- `lag_05__CT5__smoke`: coefficient `-0.000389` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.001437` (lowers CT win probability)
- `lag_00__T_place_SHORTSTAIRS`: coefficient `-0.001291` (lowers CT win probability)
- `lag_00__CT_place_SHORTSTAIRS`: coefficient `0.001273` (raises CT win probability)
- `lag_10__CT_place_SHORTSTAIRS`: coefficient `-0.001135` (lowers CT win probability)
- `lag_05__CT_place_EXTENDEDA`: coefficient `-0.000991` (lowers CT win probability)
- `lag_05__CT_place_UNDERA`: coefficient `0.000970` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.000947` (lowers CT win probability)
- `lag_09__CT_place_SHORTSTAIRS`: coefficient `-0.000899` (lowers CT win probability)
- `lag_02__T_shots_fired_sum`: coefficient `-0.000808` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.000769` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `145846`, seconds `30.50`, LSTM delta `-0.1487`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.009695`
- `lag_00__CT_place_SHORTSTAIRS`: contribution `-0.007094`
- `lag_10__CT_place_SHORTSTAIRS`: contribution `-0.006325`
- `lag_05__CT_place_EXTENDEDA`: contribution `-0.005561`
- `lag_00__T_place_SHORTSTAIRS`: contribution `-0.005426`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `145814`, seconds `30.00`, LSTM delta `-0.0750`

Top all feature movements:
- `lag_09__CT_place_SHORTSTAIRS`: contribution `-0.005014`
- `lag_00__T_shots_fired_sum`: contribution `-0.004309`
- `lag_04__CT_place_EXTENDEDA`: contribution `-0.003977`
- `lag_11__CT_place_EXTENDEDA`: contribution `-0.003264`
- `lag_09__CT_place_EXTENDEDA`: contribution `-0.003121`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `145910`, seconds `31.50`, LSTM delta `-0.0725`

Top all feature movements:
- `lag_02__T_shots_fired_sum`: contribution `-0.005450`
- `lag_02__CT_place_SHORTSTAIRS`: contribution `-0.003438`
- `lag_01__T_shots_fired_sum`: contribution `+0.003429`
- `lag_14__CT_place_EXTENDEDA`: contribution `-0.003410`
- `lag_12__CT_place_EXTENDEDA`: contribution `+0.003249`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `144694`, seconds `12.50`, LSTM delta `-0.0376`

Top all feature movements:
- `lag_13__CT_smokes_last_5s`: contribution `-0.007430`
- `lag_05__CT_place_BDOORS`: contribution `+0.003101`
- `lag_13__T2__duck_amount`: contribution `-0.002403`
- `lag_00__CT5__is_walking`: contribution `-0.001763`
- `lag_11__CT2__duck_amount`: contribution `-0.001581`

Top utility-only movements:
- `lag_13__CT_smokes_last_5s`: contribution `-0.007430`
- `lag_02__T4__flash_duration`: contribution `-0.001104`
- `lag_11__CT3__flash_duration`: contribution `-0.000959`
- `lag_05__CT5__smoke`: contribution `+0.000852`
- `lag_11__T4__flash_duration`: contribution `-0.000815`

### tick `143926`, seconds `0.50`, LSTM delta `-0.0327`

Top all feature movements:
- `lag_00__T_velocity_mean`: contribution `-0.001208`
- `lag_01__CT_place_CTSPAWN`: contribution `-0.001176`
- `lag_00__CT_velocity_mean`: contribution `-0.000855`
- `lag_01__T_money_sum`: contribution `-0.000693`
- `lag_01__T_start_balance_sum`: contribution `-0.000675`

Top utility-only movements:
- `lag_01__molly_inv_diff`: contribution `-0.000329`
- `lag_01__T_molly_inv`: contribution `-0.000317`
- `lag_01__T_smoke_inv`: contribution `-0.000304`
