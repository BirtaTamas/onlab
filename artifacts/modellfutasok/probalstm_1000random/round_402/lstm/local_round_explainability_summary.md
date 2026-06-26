# Local Round Explainability

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-vitality-vs-mouz-bo3-kZzxcq2ibUgPOmQh0hZOgn/vitality-vs-mouz-m2-train.csv`
- round_num: `5`

## Largest probability jumps

- tick `31608`, seconds `64.00`, LSTM `0.9285`, delta `+0.0344`
- tick `27544`, seconds `0.50`, LSTM `0.9290`, delta `+0.0309`
- tick `31640`, seconds `64.50`, LSTM `0.9570`, delta `+0.0285`
- tick `27704`, seconds `3.00`, LSTM `0.9290`, delta `-0.0158`
- tick `31000`, seconds `54.50`, LSTM `0.8779`, delta `+0.0152`
- tick `29592`, seconds `32.50`, LSTM `0.9173`, delta `+0.0152`
- tick `31384`, seconds `60.50`, LSTM `0.8837`, delta `-0.0152`
- tick `31448`, seconds `61.50`, LSTM `0.8895`, delta `+0.0139`
- tick `30008`, seconds `39.00`, LSTM `0.9044`, delta `-0.0133`
- tick `29816`, seconds `36.00`, LSTM `0.9246`, delta `+0.0129`

## Top 15 local ridge features

- `lag_00__CT4__is_scoped`: coefficient `-0.000556`, |coef| `0.000556`
- `lag_00__CT_shots_fired_sum`: coefficient `0.000516`, |coef| `0.000516`
- `lag_00__CT3__is_walking`: coefficient `-0.000473`, |coef| `0.000473`
- `lag_00__CT_walking_count`: coefficient `-0.000449`, |coef| `0.000449`
- `lag_00__CT5__is_walking`: coefficient `-0.000401`, |coef| `0.000401`
- `lag_00__CT2__duck_amount`: coefficient `0.000400`, |coef| `0.000400`
- `lag_00__T_place_BACKOFB`: coefficient `-0.000392`, |coef| `0.000392`
- `lag_00__T3__is_walking`: coefficient `-0.000388`, |coef| `0.000388`
- `lag_15__T_place_LONGDOG`: coefficient `-0.000388`, |coef| `0.000388`
- `lag_00__T_walking_count`: coefficient `-0.000372`, |coef| `0.000372`
- `lag_03__T_place_DUMPSTER`: coefficient `-0.000362`, |coef| `0.000362`
- `lag_07__CT4__is_scoped`: coefficient `0.000359`, |coef| `0.000359`
- `lag_00__CT2__shots_fired`: coefficient `0.000355`, |coef| `0.000355`
- `lag_00__CT_scoped_count`: coefficient `-0.000349`, |coef| `0.000349`
- `lag_00__CT_place_ENTRANCE`: coefficient `0.000349`, |coef| `0.000349`

## Top 10 utility ridge features

- `lag_08__CT1__molly`: coefficient `0.000345` (raises CT win probability)
- `lag_09__CT4__molly`: coefficient `-0.000333` (lowers CT win probability)
- `lag_09__CT1__molly`: coefficient `0.000278` (raises CT win probability)
- `lag_10__CT4__molly`: coefficient `-0.000272` (lowers CT win probability)
- `lag_13__CT1__smoke`: coefficient `0.000230` (raises CT win probability)
- `lag_07__CT1__molly`: coefficient `0.000215` (raises CT win probability)
- `lag_08__CT4__molly`: coefficient `-0.000199` (lowers CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000178` (raises CT win probability)
- `lag_11__CT1__molly`: coefficient `0.000174` (raises CT win probability)
- `lag_10__CT1__molly`: coefficient `0.000161` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT4__is_scoped`: coefficient `-0.000556` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.000516` (raises CT win probability)
- `lag_00__CT3__is_walking`: coefficient `-0.000473` (lowers CT win probability)
- `lag_00__CT_walking_count`: coefficient `-0.000449` (lowers CT win probability)
- `lag_00__CT5__is_walking`: coefficient `-0.000401` (lowers CT win probability)
- `lag_00__CT2__duck_amount`: coefficient `0.000400` (raises CT win probability)
- `lag_00__T_place_BACKOFB`: coefficient `-0.000392` (lowers CT win probability)
- `lag_00__T3__is_walking`: coefficient `-0.000388` (lowers CT win probability)
- `lag_15__T_place_LONGDOG`: coefficient `-0.000388` (lowers CT win probability)
- `lag_00__T_walking_count`: coefficient `-0.000372` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `31608`, seconds `64.00`, LSTM delta `+0.0344`

Top all feature movements:
- `lag_03__T_place_DUMPSTER`: contribution `+0.003295`
- `lag_00__CT_shots_fired_sum`: contribution `+0.001791`
- `lag_00__CT2__duck_amount`: contribution `+0.001492`
- `lag_12__CT5__duck_amount`: contribution `+0.001247`
- `lag_07__CT4__is_scoped`: contribution `+0.001222`

Top utility-only movements:
- `lag_08__CT1__molly`: contribution `+0.000858`
- `lag_09__CT4__molly`: contribution `+0.000821`

### tick `27544`, seconds `0.50`, LSTM delta `+0.0309`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `+0.001171`
- `lag_00__CT_velocity_mean`: contribution `+0.000970`
- `lag_01__CT_closest_enemy_dist`: contribution `+0.000809`
- `lag_01__T_closest_enemy_dist`: contribution `+0.000763`
- `lag_01__T_place_TSPAWN`: contribution `+0.000759`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `+0.000630`
- `lag_01__smoke_inv_diff`: contribution `+0.000504`
- `lag_01__molly_inv_diff`: contribution `+0.000491`
- `lag_01__CT_molly_inv`: contribution `+0.000368`
- `lag_01__CT_utility_inv`: contribution `+0.000316`

### tick `31640`, seconds `64.50`, LSTM delta `+0.0285`

Top all feature movements:
- `lag_00__CT4__is_scoped`: contribution `+0.001894`
- `lag_15__T_place_LONGDOG`: contribution `+0.001804`
- `lag_00__CT_shots_fired_sum`: contribution `+0.001791`
- `lag_04__T_place_DUMPSTER`: contribution `+0.001541`
- `lag_00__CT_walking_count`: contribution `+0.001209`

Top utility-only movements:
- `lag_09__CT1__molly`: contribution `+0.000693`
- `lag_10__CT4__molly`: contribution `+0.000670`

### tick `27704`, seconds `3.00`, LSTM delta `-0.0158`

Top all feature movements:
- `lag_00__CT_place_ENTRANCE`: contribution `-0.006185`
- `lag_02__CT_place_ENTRANCE`: contribution `-0.002466`
- `lag_03__CT_place_ENTRANCE`: contribution `-0.000929`
- `lag_01__CT_place_ENTRANCE`: contribution `+0.000354`
- `lag_01__CT_place_CTSPAWN`: contribution `-0.000255`

Top utility-only movements:
- `lag_06__smoke_inv_diff`: contribution `-0.000245`
- `lag_06__CT4__molly`: contribution `-0.000227`
- `lag_06__CT1__molly`: contribution `+0.000223`
- `lag_06__CT_smoke_inv`: contribution `-0.000177`
- `lag_06__CT4__utility_total`: contribution `-0.000168`

### tick `31000`, seconds `54.50`, LSTM delta `+0.0152`

Top all feature movements:
- `lag_05__CT_place_ELECTRICALBOX`: contribution `+0.002274`
- `lag_00__CT4__is_scoped`: contribution `+0.001894`
- `lag_00__CT2__duck_amount`: contribution `+0.001499`
- `lag_01__T_place_LONGDOG`: contribution `+0.001423`
- `lag_10__CT_place_ELECTRICALBOX`: contribution `-0.001254`

Top utility-only movements:
- No utility movement among the top local contributors.
