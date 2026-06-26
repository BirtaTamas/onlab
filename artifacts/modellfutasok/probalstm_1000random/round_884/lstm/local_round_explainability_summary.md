# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-liquid-bo3-9v1WXdmzbeO2q7iD5Nu_mP/mouz-vs-liquid-m2-nuke.csv`
- round_num: `10`

## Largest probability jumps

- tick `70399`, seconds `36.00`, LSTM `0.4787`, delta `+0.3297`
- tick `70303`, seconds `34.50`, LSTM `0.1922`, delta `-0.2882`
- tick `71551`, seconds `54.00`, LSTM `0.3203`, delta `-0.1919`
- tick `71039`, seconds `46.00`, LSTM `0.4819`, delta `-0.0863`
- tick `71327`, seconds `50.50`, LSTM `0.4705`, delta `-0.0727`
- tick `72351`, seconds `66.50`, LSTM `0.1337`, delta `-0.0702`
- tick `72095`, seconds `62.50`, LSTM `0.1047`, delta `+0.0696`
- tick `71583`, seconds `54.50`, LSTM `0.2509`, delta `-0.0694`
- tick `73119`, seconds `78.50`, LSTM `0.0182`, delta `-0.0664`
- tick `72543`, seconds `69.50`, LSTM `0.0768`, delta `-0.0642`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002044`, |coef| `0.002044`
- `lag_08__CT_place_ADMIN`: coefficient `-0.001983`, |coef| `0.001983`
- `lag_00__damage_diff_last_5s`: coefficient `0.001968`, |coef| `0.001968`
- `lag_08__T_place_HUT`: coefficient `0.001937`, |coef| `0.001937`
- `lag_00__CT_place_TROPHY`: coefficient `0.001821`, |coef| `0.001821`
- `lag_09__CT_place_RAFTERS`: coefficient `0.001789`, |coef| `0.001789`
- `lag_03__CT_place_MINI`: coefficient `-0.001779`, |coef| `0.001779`
- `lag_09__CT_place_HEAVEN`: coefficient `-0.001676`, |coef| `0.001676`
- `lag_14__T2__duck_amount`: coefficient `-0.001641`, |coef| `0.001641`
- `lag_12__CT_place_HELL`: coefficient `-0.001607`, |coef| `0.001607`
- `lag_00__T_place_HEAVEN`: coefficient `-0.001588`, |coef| `0.001588`
- `lag_14__CT_place_RAFTERS`: coefficient `0.001566`, |coef| `0.001566`
- `lag_13__CT_place_RAFTERS`: coefficient `0.001519`, |coef| `0.001519`
- `lag_09__CT_place_ADMIN`: coefficient `-0.001502`, |coef| `0.001502`
- `lag_05__T_place_MINI`: coefficient `0.001475`, |coef| `0.001475`

## Top 10 utility ridge features

- `lag_06__T_B_site_active_smokes`: coefficient `-0.000829` (lowers CT win probability)
- `lag_00__CT3__utility_total`: coefficient `0.000824` (raises CT win probability)
- `lag_09__CT2__molly`: coefficient `-0.000798` (lowers CT win probability)
- `lag_00__CT3__molly`: coefficient `0.000711` (raises CT win probability)
- `lag_06__CT_A_site_active_infernos`: coefficient `0.000697` (raises CT win probability)
- `lag_08__CT_flashes_last_5s`: coefficient `-0.000676` (lowers CT win probability)
- `lag_15__CT1__smoke`: coefficient `-0.000667` (lowers CT win probability)
- `lag_00__CT3__smoke`: coefficient `0.000637` (raises CT win probability)
- `lag_03__CT_B_site_active_infernos`: coefficient `0.000623` (raises CT win probability)
- `lag_06__CT2__molly`: coefficient `0.000618` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002044` (raises CT win probability)
- `lag_08__CT_place_ADMIN`: coefficient `-0.001983` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001968` (raises CT win probability)
- `lag_08__T_place_HUT`: coefficient `0.001937` (raises CT win probability)
- `lag_00__CT_place_TROPHY`: coefficient `0.001821` (raises CT win probability)
- `lag_09__CT_place_RAFTERS`: coefficient `0.001789` (raises CT win probability)
- `lag_03__CT_place_MINI`: coefficient `-0.001779` (lowers CT win probability)
- `lag_09__CT_place_HEAVEN`: coefficient `-0.001676` (lowers CT win probability)
- `lag_14__T2__duck_amount`: coefficient `-0.001641` (lowers CT win probability)
- `lag_12__CT_place_HELL`: coefficient `-0.001607` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `70399`, seconds `36.00`, LSTM delta `+0.3297`

Top all feature movements:
- `lag_08__T_place_HUT`: contribution `+0.018059`
- `lag_08__CT_place_ADMIN`: contribution `+0.013778`
- `lag_03__CT_place_MINI`: contribution `+0.010906`
- `lag_09__CT_place_RAFTERS`: contribution `+0.009558`
- `lag_09__CT_place_HEAVEN`: contribution `+0.009051`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `70303`, seconds `34.50`, LSTM delta `-0.2882`

Top all feature movements:
- `lag_09__CT_place_ADMIN`: contribution `-0.010437`
- `lag_09__CT_place_RAFTERS`: contribution `-0.009558`
- `lag_09__CT_place_HEAVEN`: contribution `-0.009051`
- `lag_12__CT_place_HELL`: contribution `-0.008714`
- `lag_14__T_place_SQUEAKY`: contribution `-0.008591`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `71551`, seconds `54.00`, LSTM delta `-0.1919`

Top all feature movements:
- `lag_05__T_place_MINI`: contribution `-0.020517`
- `lag_07__CT_place_VENDING`: contribution `-0.019084`
- `lag_15__CT_place_TROPHY`: contribution `-0.016297`
- `lag_08__CT_place_VENDING`: contribution `-0.014948`
- `lag_07__CT_place_TROPHY`: contribution `-0.011970`

Top utility-only movements:
- `lag_00__CT3__utility_total`: contribution `-0.002359`
- `lag_00__CT3__molly`: contribution `-0.001754`

### tick `71039`, seconds `46.00`, LSTM delta `-0.0863`

Top all feature movements:
- `lag_00__CT_place_TROPHY`: contribution `-0.026896`
- `lag_11__CT_place_SQUEAKY`: contribution `-0.011323`
- `lag_03__T_place_MINI`: contribution `-0.010594`
- `lag_05__CT_place_TROPHY`: contribution `-0.010143`
- `lag_05__CT_place_CONTROL`: contribution `-0.009514`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `71327`, seconds `50.50`, LSTM delta `-0.0727`

Top all feature movements:
- `lag_00__CT_place_TROPHY`: contribution `-0.026896`
- `lag_09__CT_place_TROPHY`: contribution `-0.016379`
- `lag_08__CT_place_TROPHY`: contribution `+0.014721`
- `lag_09__CT_place_VENDING`: contribution `+0.011820`
- `lag_14__CT_place_TROPHY`: contribution `-0.011059`

Top utility-only movements:
- No utility movement among the top local contributors.
