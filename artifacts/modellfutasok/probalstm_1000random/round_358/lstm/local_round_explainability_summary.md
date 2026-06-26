# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-liquid-bo3-9v1WXdmzbeO2q7iD5Nu_mP/mouz-vs-liquid-m2-nuke.csv`
- round_num: `9`

## Largest probability jumps

- tick `62383`, seconds `52.50`, LSTM `0.0461`, delta `-0.0983`
- tick `59055`, seconds `0.50`, LSTM `0.1196`, delta `-0.0739`
- tick `61903`, seconds `45.00`, LSTM `0.1263`, delta `-0.0578`
- tick `61135`, seconds `33.00`, LSTM `0.2292`, delta `-0.0492`
- tick `60015`, seconds `15.50`, LSTM `0.1531`, delta `+0.0450`
- tick `60143`, seconds `17.50`, LSTM `0.2101`, delta `+0.0386`
- tick `61551`, seconds `39.50`, LSTM `0.1814`, delta `-0.0382`
- tick `61775`, seconds `43.00`, LSTM `0.1810`, delta `+0.0310`
- tick `60303`, seconds `20.00`, LSTM `0.2692`, delta `+0.0308`
- tick `59599`, seconds `9.00`, LSTM `0.1323`, delta `-0.0302`

## Top 15 local ridge features

- `lag_15__CT_place_SQUEAKY`: coefficient `-0.001128`, |coef| `0.001128`
- `lag_08__CT_place_SQUEAKY`: coefficient `0.000987`, |coef| `0.000987`
- `lag_04__T_place_SILO`: coefficient `0.000846`, |coef| `0.000846`
- `lag_13__CT_place_SECRET`: coefficient `-0.000836`, |coef| `0.000836`
- `lag_10__T_place_SILO`: coefficient `0.000791`, |coef| `0.000791`
- `lag_08__T_utility_damage_last_5s`: coefficient `-0.000780`, |coef| `0.000780`
- `lag_14__CT_place_ADMIN`: coefficient `-0.000708`, |coef| `0.000708`
- `lag_00__CT_place_SQUEAKY`: coefficient `-0.000672`, |coef| `0.000672`
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000671`, |coef| `0.000671`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000659`, |coef| `0.000659`
- `lag_09__T_place_SILO`: coefficient `0.000656`, |coef| `0.000656`
- `lag_08__CT_place_RAFTERS`: coefficient `-0.000590`, |coef| `0.000590`
- `lag_01__CT_place_TUNNELS`: coefficient `0.000587`, |coef| `0.000587`
- `lag_09__CT4__duck_amount`: coefficient `-0.000584`, |coef| `0.000584`
- `lag_00__CT_place_TUNNELS`: coefficient `0.000581`, |coef| `0.000581`

## Top 10 utility ridge features

- `lag_08__T_utility_damage_last_5s`: coefficient `-0.000780` (lowers CT win probability)
- `lag_08__utility_damage_diff_last_5s`: coefficient `0.000521` (raises CT win probability)
- `lag_00__CT_B_site_active_smokes`: coefficient `0.000478` (raises CT win probability)
- `lag_00__CT_A_site_active_smokes`: coefficient `0.000464` (raises CT win probability)
- `lag_01__flash_inv_diff`: coefficient `0.000408` (raises CT win probability)
- `lag_12__CT_B_site_active_smokes`: coefficient `0.000404` (raises CT win probability)
- `lag_12__CT_A_site_active_smokes`: coefficient `0.000391` (raises CT win probability)
- `lag_13__CT_B_site_active_smokes`: coefficient `0.000386` (raises CT win probability)
- `lag_11__CT_B_site_active_smokes`: coefficient `0.000385` (raises CT win probability)
- `lag_00__T_B_site_active_infernos`: coefficient `-0.000382` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_15__CT_place_SQUEAKY`: coefficient `-0.001128` (lowers CT win probability)
- `lag_08__CT_place_SQUEAKY`: coefficient `0.000987` (raises CT win probability)
- `lag_04__T_place_SILO`: coefficient `0.000846` (raises CT win probability)
- `lag_13__CT_place_SECRET`: coefficient `-0.000836` (lowers CT win probability)
- `lag_10__T_place_SILO`: coefficient `0.000791` (raises CT win probability)
- `lag_14__CT_place_ADMIN`: coefficient `-0.000708` (lowers CT win probability)
- `lag_00__CT_place_SQUEAKY`: coefficient `-0.000672` (lowers CT win probability)
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000671` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000659` (lowers CT win probability)
- `lag_09__T_place_SILO`: coefficient `0.000656` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `62383`, seconds `52.50`, LSTM delta `-0.0983`

Top all feature movements:
- `lag_15__CT_place_SQUEAKY`: contribution `-0.014999`
- `lag_08__CT_place_SQUEAKY`: contribution `-0.013128`
- `lag_08__T_utility_damage_last_5s`: contribution `-0.005788`
- `lag_08__utility_damage_diff_last_5s`: contribution `-0.002447`
- `lag_05__CT_place_RAFTERS`: contribution `-0.002369`

Top utility-only movements:
- `lag_08__T_utility_damage_last_5s`: contribution `-0.005788`
- `lag_08__utility_damage_diff_last_5s`: contribution `-0.002447`

### tick `59055`, seconds `0.50`, LSTM delta `-0.0739`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.003208`
- `lag_01__T_place_TSPAWN`: contribution `-0.002917`
- `lag_01__T_closest_enemy_dist`: contribution `-0.002466`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.002283`
- `lag_01__centroid_distance_xy`: contribution `-0.001985`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `-0.000969`
- `lag_01__flash_inv_diff`: contribution `-0.000920`
- `lag_01__molly_inv_diff`: contribution `-0.000891`
- `lag_01__T1__flash`: contribution `-0.000707`
- `lag_01__T_utility_inv`: contribution `-0.000692`

### tick `61903`, seconds `45.00`, LSTM delta `-0.0578`

Top all feature movements:
- `lag_00__CT_place_SQUEAKY`: contribution `-0.008932`
- `lag_13__CT_place_SECRET`: contribution `-0.008601`
- `lag_08__CT_place_SECRET`: contribution `-0.003776`
- `lag_00__T5__duck_amount`: contribution `-0.002063`
- `lag_04__CT_place_CATWALK`: contribution `-0.001849`

Top utility-only movements:
- `lag_02__T_A_site_active_infernos`: contribution `-0.000783`
- `lag_11__T_A_site_active_infernos`: contribution `-0.000774`
- `lag_11__T_active_infernos`: contribution `-0.000703`
- `lag_11__CT_B_site_active_smokes`: contribution `-0.000640`

### tick `61135`, seconds `33.00`, LSTM delta `-0.0492`

Top all feature movements:
- `lag_10__T_place_SILO`: contribution `-0.005373`
- `lag_08__CT_place_RAFTERS`: contribution `-0.003152`
- `lag_05__CT_place_MINI`: contribution `-0.002969`
- `lag_10__T_place_ROOF`: contribution `-0.002895`
- `lag_14__CT_place_HEAVEN`: contribution `-0.002357`

Top utility-only movements:
- `lag_00__CT_B_site_active_smokes`: contribution `-0.000795`
- `lag_00__CT_A_site_active_smokes`: contribution `-0.000746`

### tick `60015`, seconds `15.50`, LSTM delta `+0.0450`

Top all feature movements:
- `lag_14__CT_place_ADMIN`: contribution `+0.009836`
- `lag_00__T_place_SILO`: contribution `+0.002415`
- `lag_00__T_place_ROOF`: contribution `+0.002374`
- `lag_00__CT_place_OBSERVATION`: contribution `+0.002290`
- `lag_12__CT_place_MINI`: contribution `+0.002261`

Top utility-only movements:
- `lag_10__T_A_site_active_infernos`: contribution `+0.000657`
