# Local Round Explainability

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-tyloo-vs-saw-bo3-Ik7_qO98IbLkRUwtJ3asIP/tyloo-vs-saw-m1-nuke.csv`
- round_num: `29`

## Largest probability jumps

- tick `229420`, seconds `49.50`, LSTM `0.7269`, delta `+0.1518`
- tick `229836`, seconds `56.00`, LSTM `0.8432`, delta `+0.1466`
- tick `227148`, seconds `14.00`, LSTM `0.2465`, delta `-0.1097`
- tick `227340`, seconds `17.00`, LSTM `0.4539`, delta `+0.0720`
- tick `227212`, seconds `15.00`, LSTM `0.2894`, delta `+0.0676`
- tick `227052`, seconds `12.50`, LSTM `0.4542`, delta `-0.0649`
- tick `228332`, seconds `32.50`, LSTM `0.5591`, delta `+0.0628`
- tick `227020`, seconds `12.00`, LSTM `0.5191`, delta `-0.0614`
- tick `227116`, seconds `13.50`, LSTM `0.3561`, delta `-0.0576`
- tick `229900`, seconds `57.00`, LSTM `0.9338`, delta `+0.0526`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001731`, |coef| `0.001731`
- `lag_03__CT_place_HUTROOF`: coefficient `0.001725`, |coef| `0.001725`
- `lag_11__CT_place_RAFTERS`: coefficient `-0.001677`, |coef| `0.001677`
- `lag_00__kill_diff_last_3s`: coefficient `0.001665`, |coef| `0.001665`
- `lag_13__CT_place_HEAVEN`: coefficient `-0.001533`, |coef| `0.001533`
- `lag_12__T_place_VENDING`: coefficient `0.001425`, |coef| `0.001425`
- `lag_05__T_place_HUT`: coefficient `0.001375`, |coef| `0.001375`
- `lag_00__CT_place_SECRET`: coefficient `-0.001353`, |coef| `0.001353`
- `lag_11__CT_place_VENTS`: coefficient `0.001338`, |coef| `0.001338`
- `lag_03__CT_place_VENTS`: coefficient `-0.001221`, |coef| `0.001221`
- `lag_02__T_place_HUT`: coefficient `0.001215`, |coef| `0.001215`
- `lag_00__T_place_LOBBY`: coefficient `-0.001170`, |coef| `0.001170`
- `lag_00__damage_diff_last_5s`: coefficient `0.001160`, |coef| `0.001160`
- `lag_09__CT_place_ADMIN`: coefficient `0.001111`, |coef| `0.001111`
- `lag_10__CT_A_site_active_infernos`: coefficient `-0.001034`, |coef| `0.001034`

## Top 10 utility ridge features

- `lag_10__CT_A_site_active_infernos`: coefficient `-0.001034` (lowers CT win probability)
- `lag_15__T_A_site_active_infernos`: coefficient `0.000693` (raises CT win probability)
- `lag_10__CT_active_infernos`: coefficient `-0.000678` (lowers CT win probability)
- `lag_10__CT_B_site_active_infernos`: coefficient `-0.000605` (lowers CT win probability)
- `lag_07__CT_B_site_active_infernos`: coefficient `-0.000562` (lowers CT win probability)
- `lag_04__T1__molly`: coefficient `-0.000543` (lowers CT win probability)
- `lag_00__T1__flash`: coefficient `-0.000532` (lowers CT win probability)
- `lag_00__T_utility_damage_last_5s`: coefficient `0.000528` (raises CT win probability)
- `lag_00__CT_A_site_active_infernos`: coefficient `0.000514` (raises CT win probability)
- `lag_03__CT_B_site_active_infernos`: coefficient `-0.000503` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001731` (raises CT win probability)
- `lag_03__CT_place_HUTROOF`: coefficient `0.001725` (raises CT win probability)
- `lag_11__CT_place_RAFTERS`: coefficient `-0.001677` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001665` (raises CT win probability)
- `lag_13__CT_place_HEAVEN`: coefficient `-0.001533` (lowers CT win probability)
- `lag_12__T_place_VENDING`: coefficient `0.001425` (raises CT win probability)
- `lag_05__T_place_HUT`: coefficient `0.001375` (raises CT win probability)
- `lag_00__CT_place_SECRET`: coefficient `-0.001353` (lowers CT win probability)
- `lag_11__CT_place_VENTS`: coefficient `0.001338` (raises CT win probability)
- `lag_03__CT_place_VENTS`: coefficient `-0.001221` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `229420`, seconds `49.50`, LSTM delta `+0.1518`

Top all feature movements:
- `lag_03__CT_place_HUTROOF`: contribution `+0.012068`
- `lag_11__CT_place_VENTS`: contribution `+0.011225`
- `lag_03__CT_place_VENTS`: contribution `+0.010243`
- `lag_12__T_place_VENDING`: contribution `+0.007225`
- `lag_00__CT_kills_last_3s`: contribution `+0.004997`

Top utility-only movements:
- `lag_10__CT_A_site_active_infernos`: contribution `+0.003649`
- `lag_10__CT_B_site_active_infernos`: contribution `+0.002077`

### tick `229836`, seconds `56.00`, LSTM delta `+0.1466`

Top all feature movements:
- `lag_00__CT_place_SECRET`: contribution `+0.013925`
- `lag_05__T_place_HUT`: contribution `+0.012815`
- `lag_02__T_place_HUT`: contribution `+0.011323`
- `lag_06__CT_place_SECRET`: contribution `+0.006984`
- `lag_11__CT_place_GARAGE`: contribution `+0.006649`

Top utility-only movements:
- `lag_15__T_A_site_active_infernos`: contribution `+0.002061`

### tick `227148`, seconds `14.00`, LSTM delta `-0.1097`

Top all feature movements:
- `lag_11__CT_place_RAFTERS`: contribution `-0.017922`
- `lag_13__CT_place_HEAVEN`: contribution `-0.008277`
- `lag_12__CT_place_ADMIN`: contribution `-0.006841`
- `lag_15__CT_place_ADMIN`: contribution `-0.006273`
- `lag_11__CT_place_HEAVEN`: contribution `-0.004595`

Top utility-only movements:
- `lag_07__CT_B_site_active_infernos`: contribution `-0.001930`
- `lag_00__CT_A_site_active_infernos`: contribution `-0.001815`

### tick `227340`, seconds `17.00`, LSTM delta `+0.0720`

Top all feature movements:
- `lag_05__CT_place_OBSERVATION`: contribution `+0.013993`
- `lag_01__CT_place_OBSERVATION`: contribution `+0.006468`
- `lag_01__T5__is_scoped`: contribution `+0.004410`
- `lag_12__CT_place_RAFTERS`: contribution `+0.003114`
- `lag_04__CT_place_ADMIN`: contribution `+0.002803`

Top utility-only movements:
- `lag_06__CT_B_site_active_infernos`: contribution `+0.001656`
- `lag_06__CT_A_site_active_infernos`: contribution `+0.001247`

### tick `227212`, seconds `15.00`, LSTM delta `+0.0676`

Top all feature movements:
- `lag_13__CT_place_HEAVEN`: contribution `+0.016555`
- `lag_11__CT_place_RAFTERS`: contribution `+0.008961`
- `lag_01__CT_place_OBSERVATION`: contribution `-0.006468`
- `lag_00__kill_diff_last_3s`: contribution `+0.004007`
- `lag_14__CT_place_ADMIN`: contribution `+0.003706`

Top utility-only movements:
- `lag_10__CT_A_site_active_infernos`: contribution `+0.003649`
- `lag_10__CT_active_infernos`: contribution `+0.001562`
- `lag_09__CT_A_site_active_infernos`: contribution `-0.001503`
