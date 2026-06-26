# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-the-mongolz-vs-natus-vincere-bo3-PG4ywdeF4kSxWHc10zCBZ3/the-mongolz-vs-natus-vincere-m1-nuke.csv`
- round_num: `5`

## Largest probability jumps

- tick `31792`, seconds `43.50`, LSTM `0.5421`, delta `+0.1795`
- tick `31824`, seconds `44.00`, LSTM `0.7049`, delta `+0.1627`
- tick `31920`, seconds `45.50`, LSTM `0.8971`, delta `+0.1416`
- tick `32880`, seconds `60.50`, LSTM `0.9643`, delta `+0.0652`
- tick `31760`, seconds `43.00`, LSTM `0.3627`, delta `+0.0630`
- tick `31568`, seconds `40.00`, LSTM `0.3075`, delta `-0.0563`
- tick `31664`, seconds `41.50`, LSTM `0.2609`, delta `-0.0522`
- tick `31536`, seconds `39.50`, LSTM `0.3638`, delta `-0.0512`
- tick `31056`, seconds `32.00`, LSTM `0.4318`, delta `-0.0426`
- tick `31504`, seconds `39.00`, LSTM `0.4149`, delta `-0.0412`

## Top 15 local ridge features

- `lag_07__T_place_DECON`: coefficient `-0.001855`, |coef| `0.001855`
- `lag_00__T_place_DECON`: coefficient `-0.001612`, |coef| `0.001612`
- `lag_03__T_place_DECON`: coefficient `-0.001461`, |coef| `0.001461`
- `lag_04__T_place_DECON`: coefficient `-0.001094`, |coef| `0.001094`
- `lag_00__T2__shots_fired`: coefficient `0.001066`, |coef| `0.001066`
- `lag_00__T_place_OBSERVATION`: coefficient `-0.001060`, |coef| `0.001060`
- `lag_09__T_place_DECON`: coefficient `0.000996`, |coef| `0.000996`
- `lag_06__T_place_DECON`: coefficient `-0.000963`, |coef| `0.000963`
- `lag_02__T_place_DECON`: coefficient `-0.000923`, |coef| `0.000923`
- `lag_13__T_place_DECON`: coefficient `0.000920`, |coef| `0.000920`
- `lag_07__CT_place_SQUEAKY`: coefficient `0.000894`, |coef| `0.000894`
- `lag_12__T_place_OBSERVATION`: coefficient `0.000877`, |coef| `0.000877`
- `lag_01__T2__shots_fired`: coefficient `0.000874`, |coef| `0.000874`
- `lag_08__T_place_OBSERVATION`: coefficient `0.000869`, |coef| `0.000869`
- `lag_04__T2__shots_fired`: coefficient `0.000825`, |coef| `0.000825`

## Top 10 utility ridge features

- `lag_00__T3__utility_total`: coefficient `-0.000492` (lowers CT win probability)
- `lag_13__T_A_site_active_infernos`: coefficient `0.000470` (raises CT win probability)
- `lag_00__T1__molly`: coefficient `-0.000464` (lowers CT win probability)
- `lag_00__T3__flash`: coefficient `-0.000453` (lowers CT win probability)
- `lag_13__T_B_site_active_infernos`: coefficient `0.000443` (raises CT win probability)
- `lag_00__T_utility_inv`: coefficient `-0.000418` (lowers CT win probability)
- `lag_00__T_molly_inv`: coefficient `-0.000408` (lowers CT win probability)
- `lag_14__T_A_site_active_infernos`: coefficient `0.000398` (raises CT win probability)
- `lag_00__T_smoke_inv`: coefficient `-0.000391` (lowers CT win probability)
- `lag_09__T_A_site_active_infernos`: coefficient `0.000375` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_07__T_place_DECON`: coefficient `-0.001855` (lowers CT win probability)
- `lag_00__T_place_DECON`: coefficient `-0.001612` (lowers CT win probability)
- `lag_03__T_place_DECON`: coefficient `-0.001461` (lowers CT win probability)
- `lag_04__T_place_DECON`: coefficient `-0.001094` (lowers CT win probability)
- `lag_00__T2__shots_fired`: coefficient `0.001066` (raises CT win probability)
- `lag_00__T_place_OBSERVATION`: coefficient `-0.001060` (lowers CT win probability)
- `lag_09__T_place_DECON`: coefficient `0.000996` (raises CT win probability)
- `lag_06__T_place_DECON`: coefficient `-0.000963` (lowers CT win probability)
- `lag_02__T_place_DECON`: coefficient `-0.000923` (lowers CT win probability)
- `lag_13__T_place_DECON`: coefficient `0.000920` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `31792`, seconds `43.50`, LSTM delta `+0.1795`

Top all feature movements:
- `lag_03__T_place_DECON`: contribution `+0.023467`
- `lag_09__T_place_DECON`: contribution `+0.016002`
- `lag_06__T_place_DECON`: contribution `+0.015466`
- `lag_08__T_place_OBSERVATION`: contribution `+0.014720`
- `lag_01__CT_place_VENDING`: contribution `+0.011927`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `31824`, seconds `44.00`, LSTM delta `+0.1627`

Top all feature movements:
- `lag_07__T_place_DECON`: contribution `+0.029796`
- `lag_04__T_place_DECON`: contribution `+0.017573`
- `lag_09__T_place_OBSERVATION`: contribution `+0.011934`
- `lag_12__T_place_DECON`: contribution `+0.011564`
- `lag_10__T_place_DECON`: contribution `+0.009642`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `31920`, seconds `45.50`, LSTM delta `+0.1416`

Top all feature movements:
- `lag_07__T_place_DECON`: contribution `+0.029796`
- `lag_00__T_place_OBSERVATION`: contribution `+0.017941`
- `lag_12__T_place_OBSERVATION`: contribution `+0.014850`
- `lag_13__T_place_DECON`: contribution `+0.014785`
- `lag_08__CT_place_VENDING`: contribution `+0.011935`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `32880`, seconds `60.50`, LSTM delta `+0.0652`

Top all feature movements:
- `lag_03__CT_place_OBSERVATION`: contribution `+0.011707`
- `lag_11__T_place_VENTS`: contribution `+0.008660`
- `lag_10__T_place_MINI`: contribution `+0.004446`
- `lag_00__CT_kills_last_3s`: contribution `+0.001869`
- `lag_04__CT1__duck_amount`: contribution `+0.001685`

Top utility-only movements:
- `lag_00__T3__utility_total`: contribution `+0.000802`

### tick `31760`, seconds `43.00`, LSTM delta `+0.0630`

Top all feature movements:
- `lag_02__T_place_DECON`: contribution `+0.014830`
- `lag_10__T_place_DECON`: contribution `+0.009642`
- `lag_07__T_place_OBSERVATION`: contribution `+0.009125`
- `lag_05__T_place_DECON`: contribution `+0.007675`
- `lag_00__CT_place_VENDING`: contribution `+0.006242`

Top utility-only movements:
- `lag_08__T_A_site_active_infernos`: contribution `+0.001009`
- `lag_08__T_B_site_active_infernos`: contribution `+0.000898`
