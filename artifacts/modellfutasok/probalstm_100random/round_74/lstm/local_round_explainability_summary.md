# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-fluxo-bo3-sWQe-jgKNP3vaioXQrjxgB/astralis-vs-fluxo-m3-nuke.csv`
- round_num: `5`

## Largest probability jumps

- tick `35978`, seconds `73.00`, LSTM `0.1020`, delta `-0.1606`
- tick `32042`, seconds `11.50`, LSTM `0.2468`, delta `+0.0765`
- tick `31338`, seconds `0.50`, LSTM `0.1536`, delta `-0.0696`
- tick `32714`, seconds `22.00`, LSTM `0.2982`, delta `+0.0583`
- tick `36394`, seconds `79.50`, LSTM `0.0517`, delta `-0.0536`
- tick `33930`, seconds `41.00`, LSTM `0.4011`, delta `+0.0521`
- tick `36330`, seconds `78.50`, LSTM `0.1168`, delta `-0.0518`
- tick `34314`, seconds `47.00`, LSTM `0.2660`, delta `+0.0508`
- tick `33706`, seconds `37.50`, LSTM `0.3783`, delta `-0.0499`
- tick `33066`, seconds `27.50`, LSTM `0.3282`, delta `+0.0499`

## Top 15 local ridge features

- `lag_00__T_place_SECRET`: coefficient `0.002580`, |coef| `0.002580`
- `lag_02__T_place_SECRET`: coefficient `0.001526`, |coef| `0.001526`
- `lag_09__T_place_ROOF`: coefficient `0.001206`, |coef| `0.001206`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001184`, |coef| `0.001184`
- `lag_03__T_place_SECRET`: coefficient `0.001180`, |coef| `0.001180`
- `lag_10__T_place_SECRET`: coefficient `-0.001141`, |coef| `0.001141`
- `lag_00__T_place_TUNNELS`: coefficient `-0.001126`, |coef| `0.001126`
- `lag_01__T_place_SECRET`: coefficient `0.001124`, |coef| `0.001124`
- `lag_02__T_place_TUNNELS`: coefficient `-0.001115`, |coef| `0.001115`
- `lag_10__T_place_ROOF`: coefficient `0.001025`, |coef| `0.001025`
- `lag_00__CT_place_TUNNELS`: coefficient `0.001021`, |coef| `0.001021`
- `lag_06__T4__duck_amount`: coefficient `-0.000909`, |coef| `0.000909`
- `lag_00__T_kills_last_3s`: coefficient `-0.000876`, |coef| `0.000876`
- `lag_00__CT4__is_walking`: coefficient `-0.000870`, |coef| `0.000870`
- `lag_03__CT_place_VENTS`: coefficient `0.000857`, |coef| `0.000857`

## Top 10 utility ridge features

- `lag_03__CT3__smoke`: coefficient `0.000669` (raises CT win probability)
- `lag_12__T_B_site_active_smokes`: coefficient `0.000455` (raises CT win probability)
- `lag_12__T_active_smokes`: coefficient `0.000454` (raises CT win probability)
- `lag_06__T_A_site_active_infernos`: coefficient `-0.000439` (lowers CT win probability)
- `lag_12__T_A_site_active_smokes`: coefficient `0.000428` (raises CT win probability)
- `lag_06__T_B_site_active_infernos`: coefficient `-0.000416` (lowers CT win probability)
- `lag_07__T_A_site_active_infernos`: coefficient `-0.000384` (lowers CT win probability)
- `lag_09__T_A_site_active_infernos`: coefficient `-0.000376` (lowers CT win probability)
- `lag_02__CT_B_site_active_smokes`: coefficient `-0.000373` (lowers CT win probability)
- `lag_02__CT_A_site_active_smokes`: coefficient `-0.000368` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_SECRET`: coefficient `0.002580` (raises CT win probability)
- `lag_02__T_place_SECRET`: coefficient `0.001526` (raises CT win probability)
- `lag_09__T_place_ROOF`: coefficient `0.001206` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001184` (lowers CT win probability)
- `lag_03__T_place_SECRET`: coefficient `0.001180` (raises CT win probability)
- `lag_10__T_place_SECRET`: coefficient `-0.001141` (lowers CT win probability)
- `lag_00__T_place_TUNNELS`: coefficient `-0.001126` (lowers CT win probability)
- `lag_01__T_place_SECRET`: coefficient `0.001124` (raises CT win probability)
- `lag_02__T_place_TUNNELS`: coefficient `-0.001115` (lowers CT win probability)
- `lag_10__T_place_ROOF`: coefficient `0.001025` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `35978`, seconds `73.00`, LSTM delta `-0.1606`

Top all feature movements:
- `lag_00__T_place_SECRET`: contribution `-0.013575`
- `lag_02__T_place_SECRET`: contribution `-0.008029`
- `lag_03__T_place_SECRET`: contribution `-0.006207`
- `lag_10__T_place_SECRET`: contribution `-0.006004`
- `lag_00__T_shots_fired_sum`: contribution `-0.004440`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `32042`, seconds `11.50`, LSTM delta `+0.0765`

Top all feature movements:
- `lag_09__T_place_ROOF`: contribution `+0.006830`
- `lag_01__CT_place_HUTROOF`: contribution `+0.005387`
- `lag_12__CT_place_HELL`: contribution `+0.005230`
- `lag_13__CT_place_HELL`: contribution `+0.003762`
- `lag_00__T_place_ROOF`: contribution `+0.003573`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `31338`, seconds `0.50`, LSTM delta `-0.0696`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.003846`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.002724`
- `lag_01__T_closest_enemy_dist`: contribution `-0.002711`
- `lag_01__T_place_TSPAWN`: contribution `-0.002486`
- `lag_01__centroid_distance_xy`: contribution `-0.002261`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `-0.000847`
- `lag_01__flash_inv_diff`: contribution `-0.000681`
- `lag_01__molly_inv_diff`: contribution `-0.000662`

### tick `32714`, seconds `22.00`, LSTM delta `+0.0583`

Top all feature movements:
- `lag_03__CT_place_VENTS`: contribution `+0.007189`
- `lag_10__T_place_ROOF`: contribution `+0.005806`
- `lag_13__CT_place_MINI`: contribution `+0.003354`
- `lag_01__T_place_SILO`: contribution `+0.002842`
- `lag_08__CT_place_MINI`: contribution `+0.002786`

Top utility-only movements:
- `lag_06__T_A_site_active_infernos`: contribution `+0.001306`
- `lag_06__T_B_site_active_infernos`: contribution `+0.001177`

### tick `36394`, seconds `79.50`, LSTM delta `-0.0536`

Top all feature movements:
- `lag_00__CT_place_VENDING`: contribution `-0.012404`
- `lag_09__CT_place_LOCKERROOM`: contribution `-0.007479`
- `lag_00__CT_place_SQUEAKY`: contribution `-0.007097`
- `lag_10__T_place_ROOF`: contribution `-0.005806`
- `lag_00__CT_place_LOBBY`: contribution `-0.004794`

Top utility-only movements:
- No utility movement among the top local contributors.
