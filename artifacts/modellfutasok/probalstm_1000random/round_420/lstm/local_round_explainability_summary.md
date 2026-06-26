# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-imperial-vs-liquid-bo3-eiIGPV5tjvJFQ73hC8D8JI/imperial-vs-liquid-m3-anubis.csv`
- round_num: `8`

## Largest probability jumps

- tick `62798`, seconds `65.50`, LSTM `0.4428`, delta `-0.2120`
- tick `63054`, seconds `69.50`, LSTM `0.0343`, delta `-0.1771`
- tick `62830`, seconds `66.00`, LSTM `0.3115`, delta `-0.1314`
- tick `62158`, seconds `55.50`, LSTM `0.5783`, delta `-0.1060`
- tick `62126`, seconds `55.00`, LSTM `0.6842`, delta `-0.0905`
- tick `62958`, seconds `68.00`, LSTM `0.3400`, delta `+0.0894`
- tick `61678`, seconds `48.00`, LSTM `0.7673`, delta `+0.0859`
- tick `62990`, seconds `68.50`, LSTM `0.2542`, delta `-0.0858`
- tick `61486`, seconds `45.00`, LSTM `0.6122`, delta `+0.0842`
- tick `62862`, seconds `66.50`, LSTM `0.2619`, delta `-0.0495`

## Top 15 local ridge features

- `lag_00__CT_place_FOUNTAIN`: coefficient `0.003613`, |coef| `0.003613`
- `lag_00__CT_place_BRICKS`: coefficient `0.002779`, |coef| `0.002779`
- `lag_01__CT_place_FOUNTAIN`: coefficient `0.002710`, |coef| `0.002710`
- `lag_00__CT5__is_scoped`: coefficient `-0.002311`, |coef| `0.002311`
- `lag_03__T_place_FOUNTAIN`: coefficient `-0.002311`, |coef| `0.002311`
- `lag_03__T_place_MAIN`: coefficient `0.002092`, |coef| `0.002092`
- `lag_03__CT_place_BRICKS`: coefficient `-0.002049`, |coef| `0.002049`
- `lag_04__T_place_FOUNTAIN`: coefficient `-0.001872`, |coef| `0.001872`
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `0.001785`, |coef| `0.001785`
- `lag_00__damage_diff_last_5s`: coefficient `0.001702`, |coef| `0.001702`
- `lag_04__T_place_MAIN`: coefficient `0.001650`, |coef| `0.001650`
- `lag_01__CT5__is_scoped`: coefficient `-0.001524`, |coef| `0.001524`
- `lag_10__CT_place_BACKOFB`: coefficient `0.001510`, |coef| `0.001510`
- `lag_08__CT_place_FOUNTAIN`: coefficient `0.001450`, |coef| `0.001450`
- `lag_06__T3__is_walking`: coefficient `-0.001338`, |coef| `0.001338`

## Top 10 utility ridge features

- `lag_00__CT5__molly`: coefficient `0.001219` (raises CT win probability)
- `lag_13__T_B_site_active_infernos`: coefficient `0.001190` (raises CT win probability)
- `lag_00__CT5__utility_total`: coefficient `0.000999` (raises CT win probability)
- `lag_14__T_B_site_active_infernos`: coefficient `0.000926` (raises CT win probability)
- `lag_00__CT5__flash`: coefficient `0.000872` (raises CT win probability)
- `lag_01__CT5__molly`: coefficient `0.000763` (raises CT win probability)
- `lag_14__active_infernos_total`: coefficient `0.000668` (raises CT win probability)
- `lag_14__T_active_infernos`: coefficient `0.000646` (raises CT win probability)
- `lag_15__CT_B_site_active_infernos`: coefficient `0.000615` (raises CT win probability)
- `lag_10__CT_active_infernos`: coefficient `-0.000590` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_FOUNTAIN`: coefficient `0.003613` (raises CT win probability)
- `lag_00__CT_place_BRICKS`: coefficient `0.002779` (raises CT win probability)
- `lag_01__CT_place_FOUNTAIN`: coefficient `0.002710` (raises CT win probability)
- `lag_00__CT5__is_scoped`: coefficient `-0.002311` (lowers CT win probability)
- `lag_03__T_place_FOUNTAIN`: coefficient `-0.002311` (lowers CT win probability)
- `lag_03__T_place_MAIN`: coefficient `0.002092` (raises CT win probability)
- `lag_03__CT_place_BRICKS`: coefficient `-0.002049` (lowers CT win probability)
- `lag_04__T_place_FOUNTAIN`: coefficient `-0.001872` (lowers CT win probability)
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `0.001785` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001702` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `62798`, seconds `65.50`, LSTM delta `-0.2120`

Top all feature movements:
- `lag_00__CT_place_FOUNTAIN`: contribution `-0.038002`
- `lag_03__T_place_MAIN`: contribution `-0.013525`
- `lag_03__T_place_FOUNTAIN`: contribution `-0.010925`
- `lag_10__CT_place_BACKOFB`: contribution `-0.008620`
- `lag_00__CT5__is_scoped`: contribution `-0.008266`

Top utility-only movements:
- `lag_13__T_B_site_active_infernos`: contribution `-0.003365`
- `lag_00__CT5__molly`: contribution `-0.003023`

### tick `63054`, seconds `69.50`, LSTM delta `-0.1771`

Top all feature movements:
- `lag_03__CT_place_BRICKS`: contribution `-0.039345`
- `lag_02__CT_place_BRICKS`: contribution `-0.016421`
- `lag_08__CT_place_FOUNTAIN`: contribution `-0.015253`
- `lag_11__T_place_FOUNTAIN`: contribution `-0.006251`
- `lag_11__T_place_MAIN`: contribution `-0.006022`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `62830`, seconds `66.00`, LSTM delta `-0.1314`

Top all feature movements:
- `lag_01__CT_place_FOUNTAIN`: contribution `-0.028509`
- `lag_04__T_place_MAIN`: contribution `-0.010666`
- `lag_04__T_place_FOUNTAIN`: contribution `-0.008850`
- `lag_11__CT_place_BACKOFB`: contribution `-0.006112`
- `lag_07__T_place_STREET`: contribution `-0.005620`

Top utility-only movements:
- `lag_14__T_B_site_active_infernos`: contribution `-0.002619`

### tick `62158`, seconds `55.50`, LSTM delta `-0.1060`

Top all feature movements:
- `lag_01__CT_place_BACKOFB`: contribution `-0.004262`
- `lag_00__T_kills_last_3s`: contribution `-0.004068`
- `lag_02__CT5__is_scoped`: contribution `-0.003755`
- `lag_00__kill_diff_last_3s`: contribution `-0.003192`
- `lag_04__T5__duck_amount`: contribution `-0.003110`

Top utility-only movements:
- `lag_00__T_utility_damage_last_5s`: contribution `-0.002185`

### tick `62126`, seconds `55.00`, LSTM delta `-0.0905`

Top all feature movements:
- `lag_14__CT_place_OUTSIDELONG`: contribution `-0.005478`
- `lag_01__CT5__is_scoped`: contribution `-0.005450`
- `lag_00__CT_shots_fired_sum`: contribution `-0.004301`
- `lag_08__T_place_MAIN`: contribution `-0.004292`
- `lag_00__damage_diff_last_5s`: contribution `-0.003725`

Top utility-only movements:
- No utility movement among the top local contributors.
