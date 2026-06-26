# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-falcons-vs-natus-vincere-bo3-z3OpWwYDPa33wwfDY8_B1Q/falcons-vs-natus-vincere-m1-nuke.csv`
- round_num: `12`

## Largest probability jumps

- tick `88416`, seconds `52.00`, LSTM `0.1434`, delta `-0.2465`
- tick `88640`, seconds `55.50`, LSTM `0.2182`, delta `+0.1243`
- tick `88384`, seconds `51.50`, LSTM `0.3899`, delta `-0.0832`
- tick `89184`, seconds `64.00`, LSTM `0.0275`, delta `-0.0649`
- tick `88128`, seconds `47.50`, LSTM `0.5392`, delta `-0.0561`
- tick `88768`, seconds `57.50`, LSTM `0.2204`, delta `-0.0535`
- tick `88800`, seconds `58.00`, LSTM `0.1703`, delta `-0.0500`
- tick `88320`, seconds `50.50`, LSTM `0.5017`, delta `-0.0336`
- tick `88352`, seconds `51.00`, LSTM `0.4731`, delta `-0.0285`
- tick `88672`, seconds `56.00`, LSTM `0.2443`, delta `+0.0261`

## Top 15 local ridge features

- `lag_03__CT_place_CRANE`: coefficient `-0.002588`, |coef| `0.002588`
- `lag_01__CT_place_SECRET`: coefficient `0.002185`, |coef| `0.002185`
- `lag_00__CT_place_CRANE`: coefficient `-0.001908`, |coef| `0.001908`
- `lag_15__CT_place_MINI`: coefficient `0.001468`, |coef| `0.001468`
- `lag_08__T_place_ROOF`: coefficient `-0.001457`, |coef| `0.001457`
- `lag_10__CT_place_CONTROL`: coefficient `-0.001308`, |coef| `0.001308`
- `lag_00__T_kills_last_3s`: coefficient `-0.001198`, |coef| `0.001198`
- `lag_02__CT_place_MINI`: coefficient `0.001175`, |coef| `0.001175`
- `lag_04__T_place_SQUEAKY`: coefficient `0.001171`, |coef| `0.001171`
- `lag_08__CT_place_SECRET`: coefficient `-0.001157`, |coef| `0.001157`
- `lag_08__CT_place_MINI`: coefficient `-0.001131`, |coef| `0.001131`
- `lag_01__CT_place_CRANE`: coefficient `-0.001089`, |coef| `0.001089`
- `lag_11__CT_place_CONTROL`: coefficient `-0.001061`, |coef| `0.001061`
- `lag_00__CT_place_SECRET`: coefficient `0.001053`, |coef| `0.001053`
- `lag_01__CT_place_CONTROL`: coefficient `-0.000925`, |coef| `0.000925`

## Top 10 utility ridge features

- `lag_00__CT3__molly`: coefficient `0.000866` (raises CT win probability)
- `lag_00__CT3__utility_total`: coefficient `0.000742` (raises CT win probability)
- `lag_00__CT3__flash`: coefficient `0.000647` (raises CT win probability)
- `lag_13__CT1__molly`: coefficient `0.000630` (raises CT win probability)
- `lag_08__T2__molly`: coefficient `0.000584` (raises CT win probability)
- `lag_00__CT_active_infernos`: coefficient `0.000563` (raises CT win probability)
- `lag_00__CT_molly_inv`: coefficient `0.000543` (raises CT win probability)
- `lag_04__T_active_infernos`: coefficient `-0.000522` (lowers CT win probability)
- `lag_00__CT_utility_inv`: coefficient `0.000470` (raises CT win probability)
- `lag_02__T_utility_damage_last_5s`: coefficient `-0.000415` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_03__CT_place_CRANE`: coefficient `-0.002588` (lowers CT win probability)
- `lag_01__CT_place_SECRET`: coefficient `0.002185` (raises CT win probability)
- `lag_00__CT_place_CRANE`: coefficient `-0.001908` (lowers CT win probability)
- `lag_15__CT_place_MINI`: coefficient `0.001468` (raises CT win probability)
- `lag_08__T_place_ROOF`: coefficient `-0.001457` (lowers CT win probability)
- `lag_10__CT_place_CONTROL`: coefficient `-0.001308` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001198` (lowers CT win probability)
- `lag_02__CT_place_MINI`: coefficient `0.001175` (raises CT win probability)
- `lag_04__T_place_SQUEAKY`: coefficient `0.001171` (raises CT win probability)
- `lag_08__CT_place_SECRET`: coefficient `-0.001157` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `88416`, seconds `52.00`, LSTM delta `-0.2465`

Top all feature movements:
- `lag_03__CT_place_CRANE`: contribution `-0.042459`
- `lag_01__CT_place_SECRET`: contribution `-0.022496`
- `lag_08__T_place_ROOF`: contribution `-0.008254`
- `lag_04__T_place_SQUEAKY`: contribution `-0.007292`
- `lag_02__CT_place_MINI`: contribution `-0.007201`

Top utility-only movements:
- `lag_00__CT3__molly`: contribution `-0.002137`

### tick `88640`, seconds `55.50`, LSTM delta `+0.1243`

Top all feature movements:
- `lag_00__CT_place_CRANE`: contribution `+0.031297`
- `lag_08__CT_place_SECRET`: contribution `+0.011914`
- `lag_15__CT_place_MINI`: contribution `+0.009002`
- `lag_10__CT_place_CRANE`: contribution `+0.005360`
- `lag_09__CT_place_MINI`: contribution `+0.004965`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `88384`, seconds `51.50`, LSTM delta `-0.0832`

Top all feature movements:
- `lag_02__CT_place_CRANE`: contribution `-0.014822`
- `lag_00__CT_place_SECRET`: contribution `-0.010834`
- `lag_15__CT_place_MINI`: contribution `-0.009002`
- `lag_03__T_place_SQUEAKY`: contribution `-0.005060`
- `lag_07__T_place_ROOF`: contribution `-0.004359`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `89184`, seconds `64.00`, LSTM delta `-0.0649`

Top all feature movements:
- `lag_01__CT_place_SQUEAKY`: contribution `-0.010525`
- `lag_13__CT_place_CONTROL`: contribution `-0.008764`
- `lag_00__CT_place_SQUEAKY`: contribution `-0.005469`
- `lag_00__T_kills_last_3s`: contribution `-0.003795`
- `lag_00__T_shots_fired_sum`: contribution `-0.002593`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `88128`, seconds `47.50`, LSTM delta `-0.0561`

Top all feature movements:
- `lag_01__T_place_SQUEAKY`: contribution `-0.004486`
- `lag_00__T_kills_last_3s`: contribution `-0.003795`
- `lag_09__CT_shots_fired_sum`: contribution `+0.002528`
- `lag_15__CT4__duck_amount`: contribution `-0.002405`
- `lag_00__T_damage_last_5s`: contribution `-0.002062`

Top utility-only movements:
- No utility movement among the top local contributors.
