# Local Round Explainability

- csv_path: `processed_full\esports_world_cup\esports-world-cup-2025-aurora-vs-falcons-bo3-5oHSxtVT-5F3Op7ZcgBMjW\aurora-vs-falcons-m2-train.csv`
- round_num: `9`

## Largest probability jumps

- tick `63200`, seconds `53.50`, LSTM `0.7978`, delta `+0.1243`
- tick `62656`, seconds `45.00`, LSTM `0.6473`, delta `+0.1152`
- tick `63232`, seconds `54.00`, LSTM `0.8961`, delta `+0.0983`
- tick `61792`, seconds `31.50`, LSTM `0.5639`, delta `+0.0862`
- tick `62688`, seconds `45.50`, LSTM `0.7110`, delta `+0.0638`
- tick `59840`, seconds `1.00`, LSTM `0.1994`, delta `-0.0508`
- tick `62848`, seconds `48.00`, LSTM `0.6112`, delta `-0.0458`
- tick `60544`, seconds `12.00`, LSTM `0.3712`, delta `+0.0458`
- tick `60416`, seconds `10.00`, LSTM `0.2427`, delta `-0.0396`
- tick `63136`, seconds `52.50`, LSTM `0.6738`, delta `+0.0395`

## Top 15 local ridge features

- `lag_11__CT_place_IVY`: coefficient `-0.002522`, |coef| `0.002522`
- `lag_12__CT_place_IVY`: coefficient `-0.001767`, |coef| `0.001767`
- `lag_07__T_place_BACKOFB`: coefficient `0.001707`, |coef| `0.001707`
- `lag_00__CT_kills_last_3s`: coefficient `0.001654`, |coef| `0.001654`
- `lag_08__T_place_DUMPSTER`: coefficient `-0.001613`, |coef| `0.001613`
- `lag_05__CT_place_CONNECTOR`: coefficient `-0.001495`, |coef| `0.001495`
- `lag_14__T_place_DUMPSTER`: coefficient `0.001420`, |coef| `0.001420`
- `lag_08__T_place_BACKOFB`: coefficient `0.001400`, |coef| `0.001400`
- `lag_13__T1__is_walking`: coefficient `0.001312`, |coef| `0.001312`
- `lag_00__CT_damage_last_5s`: coefficient `0.001268`, |coef| `0.001268`
- `lag_01__CT_shots_fired_sum`: coefficient `0.001262`, |coef| `0.001262`
- `lag_01__CT4__shots_fired`: coefficient `0.001212`, |coef| `0.001212`
- `lag_13__CT_place_CONNECTOR`: coefficient `0.001175`, |coef| `0.001175`
- `lag_14__CT_place_CONNECTOR`: coefficient `0.001168`, |coef| `0.001168`
- `lag_00__T_flashes_last_5s`: coefficient `-0.001165`, |coef| `0.001165`

## Top 10 utility ridge features

- `lag_00__T_flashes_last_5s`: coefficient `-0.001165` (lowers CT win probability)
- `lag_14__T_A_site_active_infernos`: coefficient `0.000833` (raises CT win probability)
- `lag_14__T_B_site_active_infernos`: coefficient `0.000784` (raises CT win probability)
- `lag_00__T2__flash`: coefficient `-0.000778` (lowers CT win probability)
- `lag_00__T4__utility_total`: coefficient `-0.000774` (lowers CT win probability)
- `lag_00__T2__utility_total`: coefficient `-0.000761` (lowers CT win probability)
- `lag_00__T_molly_inv`: coefficient `-0.000728` (lowers CT win probability)
- `lag_00__T_utility_inv`: coefficient `-0.000727` (lowers CT win probability)
- `lag_00__T2__molly`: coefficient `-0.000713` (lowers CT win probability)
- `lag_00__T4__molly`: coefficient `-0.000699` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_11__CT_place_IVY`: coefficient `-0.002522` (lowers CT win probability)
- `lag_12__CT_place_IVY`: coefficient `-0.001767` (lowers CT win probability)
- `lag_07__T_place_BACKOFB`: coefficient `0.001707` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001654` (raises CT win probability)
- `lag_08__T_place_DUMPSTER`: coefficient `-0.001613` (lowers CT win probability)
- `lag_05__CT_place_CONNECTOR`: coefficient `-0.001495` (lowers CT win probability)
- `lag_14__T_place_DUMPSTER`: coefficient `0.001420` (raises CT win probability)
- `lag_08__T_place_BACKOFB`: coefficient `0.001400` (raises CT win probability)
- `lag_13__T1__is_walking`: coefficient `0.001312` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001268` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `63200`, seconds `53.50`, LSTM delta `+0.1243`

Top all feature movements:
- `lag_11__CT_place_IVY`: contribution `+0.028779`
- `lag_11__T_place_TSTAIRS`: contribution `+0.006023`
- `lag_05__CT_place_CONNECTOR`: contribution `+0.005347`
- `lag_07__T_place_BACKOFB`: contribution `+0.004583`
- `lag_13__CT_place_CONNECTOR`: contribution `+0.004200`

Top utility-only movements:
- `lag_14__T_A_site_active_infernos`: contribution `+0.002480`
- `lag_14__T_B_site_active_infernos`: contribution `+0.002217`
- `lag_07__T_A_site_active_infernos`: contribution `+0.001398`

### tick `62656`, seconds `45.00`, LSTM delta `+0.1152`

Top all feature movements:
- `lag_08__T_place_DUMPSTER`: contribution `+0.014664`
- `lag_14__T_place_DUMPSTER`: contribution `+0.012914`
- `lag_00__CT_kills_last_3s`: contribution `+0.004775`
- `lag_01__CT_shots_fired_sum`: contribution `+0.004385`
- `lag_01__CT4__shots_fired`: contribution `+0.003266`

Top utility-only movements:
- `lag_00__T4__utility_total`: contribution `+0.001806`
- `lag_00__T4__molly`: contribution `+0.001524`

### tick `63232`, seconds `54.00`, LSTM delta `+0.0983`

Top all feature movements:
- `lag_12__CT_place_IVY`: contribution `+0.020168`
- `lag_05__CT_place_CONNECTOR`: contribution `+0.005347`
- `lag_00__CT_kills_last_3s`: contribution `+0.004775`
- `lag_01__CT_shots_fired_sum`: contribution `+0.004385`
- `lag_14__CT_place_CONNECTOR`: contribution `+0.004178`

Top utility-only movements:
- `lag_15__T_A_site_active_infernos`: contribution `+0.001505`

### tick `61792`, seconds `31.50`, LSTM delta `+0.0862`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.004775`
- `lag_02__T_place_IVY`: contribution `+0.003038`
- `lag_14__T2__duck_amount`: contribution `+0.002705`
- `lag_14__CT2__duck_amount`: contribution `+0.002692`
- `lag_01__CT_shots_fired_sum`: contribution `+0.002631`

Top utility-only movements:
- `lag_00__T2__molly`: contribution `+0.001588`

### tick `62688`, seconds `45.50`, LSTM delta `+0.0638`

Top all feature movements:
- `lag_00__CT_place_IVY`: contribution `+0.013057`
- `lag_07__T_place_BACKOFB`: contribution `+0.004583`
- `lag_02__CT4__shots_fired`: contribution `+0.002632`
- `lag_01__CT_shots_fired_sum`: contribution `-0.002631`
- `lag_02__CT_shots_fired_sum`: contribution `+0.002429`

Top utility-only movements:
- `lag_00__T3__molly`: contribution `+0.001198`
