# Local Round Explainability

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-spirit-vs-furia-bo3-_zQK5XUu10iN1JLmPA8zQ4/spirit-vs-furia-m2-nuke.csv`
- round_num: `12`

## Largest probability jumps

- tick `98315`, seconds `39.00`, LSTM `0.3176`, delta `+0.1624`
- tick `96811`, seconds `15.50`, LSTM `0.2079`, delta `-0.1241`
- tick `100107`, seconds `67.00`, LSTM `0.3504`, delta `-0.1011`
- tick `97227`, seconds `22.00`, LSTM `0.1787`, delta `-0.0833`
- tick `100331`, seconds `70.50`, LSTM `0.1994`, delta `-0.0671`
- tick `98731`, seconds `45.50`, LSTM `0.4632`, delta `+0.0596`
- tick `97291`, seconds `23.00`, LSTM `0.1096`, delta `-0.0564`
- tick `96683`, seconds `13.50`, LSTM `0.3327`, delta `+0.0480`
- tick `100907`, seconds `79.50`, LSTM `0.0151`, delta `-0.0479`
- tick `98347`, seconds `39.50`, LSTM `0.3595`, delta `+0.0419`

## Top 15 local ridge features

- `lag_00__T_place_HEAVEN`: coefficient `-0.002319`, |coef| `0.002319`
- `lag_00__CT_place_ROOF`: coefficient `-0.002304`, |coef| `0.002304`
- `lag_00__CT_place_VENTS`: coefficient `-0.001104`, |coef| `0.001104`
- `lag_14__CT_place_ADMIN`: coefficient `0.001064`, |coef| `0.001064`
- `lag_01__T_place_HEAVEN`: coefficient `-0.001007`, |coef| `0.001007`
- `lag_09__CT_place_CONTROL`: coefficient `-0.001006`, |coef| `0.001006`
- `lag_02__CT_place_ROOF`: coefficient `-0.000979`, |coef| `0.000979`
- `lag_12__CT_place_VENDING`: coefficient `0.000929`, |coef| `0.000929`
- `lag_02__T_place_RAFTERS`: coefficient `-0.000880`, |coef| `0.000880`
- `lag_12__CT_place_DECON`: coefficient `-0.000853`, |coef| `0.000853`
- `lag_09__T_place_RAFTERS`: coefficient `-0.000845`, |coef| `0.000845`
- `lag_10__CT_place_DECON`: coefficient `0.000830`, |coef| `0.000830`
- `lag_02__CT_place_VENTS`: coefficient `-0.000826`, |coef| `0.000826`
- `lag_00__T_place_RAFTERS`: coefficient `-0.000822`, |coef| `0.000822`
- `lag_01__CT_place_VENTS`: coefficient `-0.000804`, |coef| `0.000804`

## Top 10 utility ridge features

- `lag_00__T_utility_damage_last_5s`: coefficient `-0.000659` (lowers CT win probability)
- `lag_04__T_utility_damage_last_5s`: coefficient `-0.000420` (lowers CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000417` (raises CT win probability)
- `lag_01__T_utility_damage_last_5s`: coefficient `-0.000391` (lowers CT win probability)
- `lag_08__T_utility_damage_last_5s`: coefficient `0.000375` (raises CT win probability)
- `lag_05__CT_B_site_active_smokes`: coefficient `0.000348` (raises CT win probability)
- `lag_04__CT_B_site_active_smokes`: coefficient `0.000348` (raises CT win probability)
- `lag_04__CT_A_site_active_smokes`: coefficient `0.000310` (raises CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000295` (raises CT win probability)
- `lag_00__CT5__flash`: coefficient `0.000292` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_HEAVEN`: coefficient `-0.002319` (lowers CT win probability)
- `lag_00__CT_place_ROOF`: coefficient `-0.002304` (lowers CT win probability)
- `lag_00__CT_place_VENTS`: coefficient `-0.001104` (lowers CT win probability)
- `lag_14__CT_place_ADMIN`: coefficient `0.001064` (raises CT win probability)
- `lag_01__T_place_HEAVEN`: coefficient `-0.001007` (lowers CT win probability)
- `lag_09__CT_place_CONTROL`: coefficient `-0.001006` (lowers CT win probability)
- `lag_02__CT_place_ROOF`: coefficient `-0.000979` (lowers CT win probability)
- `lag_12__CT_place_VENDING`: coefficient `0.000929` (raises CT win probability)
- `lag_02__T_place_RAFTERS`: coefficient `-0.000880` (lowers CT win probability)
- `lag_12__CT_place_DECON`: coefficient `-0.000853` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `98315`, seconds `39.00`, LSTM delta `+0.1624`

Top all feature movements:
- `lag_00__CT_place_ROOF`: contribution `+0.067965`
- `lag_12__CT_place_VENDING`: contribution `+0.015915`
- `lag_15__CT_place_TROPHY`: contribution `+0.011333`
- `lag_07__CT_place_VENDING`: contribution `+0.010165`
- `lag_08__CT_place_SQUEAKY`: contribution `+0.007577`

Top utility-only movements:
- `lag_06__T_A_site_active_infernos`: contribution `+0.000739`

### tick `96811`, seconds `15.50`, LSTM delta `-0.1241`

Top all feature movements:
- `lag_14__CT_place_ADMIN`: contribution `-0.014785`
- `lag_09__CT_place_CONTROL`: contribution `-0.010441`
- `lag_04__CT_place_TROPHY`: contribution `-0.010303`
- `lag_06__CT_place_CONTROL`: contribution `-0.007860`
- `lag_01__CT_place_OBSERVATION`: contribution `-0.005974`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `100107`, seconds `67.00`, LSTM delta `-0.1011`

Top all feature movements:
- `lag_00__T_place_HEAVEN`: contribution `-0.028455`
- `lag_12__CT_place_DECON`: contribution `-0.013556`
- `lag_10__CT_place_DECON`: contribution `-0.013198`
- `lag_00__CT_place_VENTS`: contribution `-0.009262`
- `lag_00__CT_place_TUNNELS`: contribution `-0.002457`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `97227`, seconds `22.00`, LSTM delta `-0.0833`

Top all feature movements:
- `lag_00__CT_place_ROOF`: contribution `-0.067965`
- `lag_14__CT_place_OBSERVATION`: contribution `-0.006068`
- `lag_00__CT_place_LOBBY`: contribution `-0.005787`
- `lag_00__T2__duck_amount`: contribution `+0.002081`
- `lag_13__CT_place_TROPHY`: contribution `-0.002069`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `100331`, seconds `70.50`, LSTM delta `-0.0671`

Top all feature movements:
- `lag_00__T_place_HEAVEN`: contribution `-0.028455`
- `lag_07__T_place_HEAVEN`: contribution `-0.009283`
- `lag_07__CT_place_VENTS`: contribution `-0.005277`
- `lag_07__T1__duck_amount`: contribution `-0.002064`
- `lag_09__T_place_CATWALK`: contribution `-0.002015`

Top utility-only movements:
- No utility movement among the top local contributors.
