# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-mouz-vs-pain-bo3-Ao8EIC0rxvFkpkJ5bGImFu/mouz-vs-pain-m3-dust2.csv`
- round_num: `6`

## Largest probability jumps

- tick `37864`, seconds `40.00`, LSTM `0.0475`, delta `-0.1510`
- tick `37608`, seconds `36.00`, LSTM `0.2183`, delta `+0.0772`
- tick `35688`, seconds `6.00`, LSTM `0.1333`, delta `-0.0627`
- tick `37832`, seconds `39.50`, LSTM `0.1985`, delta `+0.0627`
- tick `35336`, seconds `0.50`, LSTM `0.0852`, delta `-0.0590`
- tick `37224`, seconds `30.00`, LSTM `0.0918`, delta `-0.0473`
- tick `37000`, seconds `26.50`, LSTM `0.1175`, delta `-0.0400`
- tick `37640`, seconds `36.50`, LSTM `0.1834`, delta `-0.0349`
- tick `35816`, seconds `8.00`, LSTM `0.1356`, delta `+0.0339`
- tick `39080`, seconds `59.00`, LSTM `0.0119`, delta `-0.0307`

## Top 15 local ridge features

- `lag_08__T_place_SIDE`: coefficient `0.001872`, |coef| `0.001872`
- `lag_00__T_place_SIDE`: coefficient `-0.001019`, |coef| `0.001019`
- `lag_00__CT_smokes_last_5s`: coefficient `0.000816`, |coef| `0.000816`
- `lag_09__CT_place_TUNNELSTAIRS`: coefficient `0.000756`, |coef| `0.000756`
- `lag_02__CT_place_TUNNELSTAIRS`: coefficient `0.000599`, |coef| `0.000599`
- `lag_00__T_place_PIT`: coefficient `-0.000589`, |coef| `0.000589`
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000579`, |coef| `0.000579`
- `lag_12__CT_place_UPPERTUNNEL`: coefficient `0.000578`, |coef| `0.000578`
- `lag_00__T_velocity_mean`: coefficient `-0.000551`, |coef| `0.000551`
- `lag_07__CT_place_ARAMP`: coefficient `-0.000545`, |coef| `0.000545`
- `lag_00__CT_velocity_mean`: coefficient `-0.000544`, |coef| `0.000544`
- `lag_05__T_place_PIT`: coefficient `-0.000532`, |coef| `0.000532`
- `lag_08__CT_place_TUNNELSTAIRS`: coefficient `-0.000525`, |coef| `0.000525`
- `lag_14__CT_smokes_last_5s`: coefficient `0.000523`, |coef| `0.000523`
- `lag_00__CT_flashes_last_5s`: coefficient `0.000519`, |coef| `0.000519`

## Top 10 utility ridge features

- `lag_00__CT_smokes_last_5s`: coefficient `0.000816` (raises CT win probability)
- `lag_14__CT_smokes_last_5s`: coefficient `0.000523` (raises CT win probability)
- `lag_00__CT_flashes_last_5s`: coefficient `0.000519` (raises CT win probability)
- `lag_01__CT_smokes_last_5s`: coefficient `0.000419` (raises CT win probability)
- `lag_10__CT_smokes_last_5s`: coefficient `-0.000376` (lowers CT win probability)
- `lag_14__CT_flashes_last_5s`: coefficient `0.000333` (raises CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000329` (raises CT win probability)
- `lag_04__CT3__flash_duration`: coefficient `0.000317` (raises CT win probability)
- `lag_03__CT3__flash_duration`: coefficient `0.000316` (raises CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000307` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_08__T_place_SIDE`: coefficient `0.001872` (raises CT win probability)
- `lag_00__T_place_SIDE`: coefficient `-0.001019` (lowers CT win probability)
- `lag_09__CT_place_TUNNELSTAIRS`: coefficient `0.000756` (raises CT win probability)
- `lag_02__CT_place_TUNNELSTAIRS`: coefficient `0.000599` (raises CT win probability)
- `lag_00__T_place_PIT`: coefficient `-0.000589` (lowers CT win probability)
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000579` (lowers CT win probability)
- `lag_12__CT_place_UPPERTUNNEL`: coefficient `0.000578` (raises CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000551` (lowers CT win probability)
- `lag_07__CT_place_ARAMP`: coefficient `-0.000545` (lowers CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000544` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `37864`, seconds `40.00`, LSTM delta `-0.1510`

Top all feature movements:
- `lag_08__T_place_SIDE`: contribution `-0.036221`
- `lag_09__CT_place_TUNNELSTAIRS`: contribution `-0.010647`
- `lag_12__CT_place_UPPERTUNNEL`: contribution `-0.004431`
- `lag_07__CT_place_ARAMP`: contribution `-0.003394`
- `lag_05__T_place_PIT`: contribution `-0.003358`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `37608`, seconds `36.00`, LSTM delta `+0.0772`

Top all feature movements:
- `lag_00__T_place_SIDE`: contribution `+0.019707`
- `lag_12__T_place_SIDE`: contribution `+0.007651`
- `lag_01__CT_place_TUNNELSTAIRS`: contribution `+0.005247`
- `lag_12__CT_place_TUNNELSTAIRS`: contribution `+0.004880`
- `lag_13__T_place_PIT`: contribution `+0.002805`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `35688`, seconds `6.00`, LSTM delta `-0.0627`

Top all feature movements:
- `lag_00__CT_smokes_last_5s`: contribution `-0.014111`
- `lag_10__CT_smokes_last_5s`: contribution `-0.006503`
- `lag_00__CT_flashes_last_5s`: contribution `-0.005710`
- `lag_10__CT_flashes_last_5s`: contribution `-0.002634`
- `lag_07__CT_place_UNDERA`: contribution `-0.002163`

Top utility-only movements:
- `lag_00__CT_smokes_last_5s`: contribution `-0.014111`
- `lag_10__CT_smokes_last_5s`: contribution `-0.006503`
- `lag_00__CT_flashes_last_5s`: contribution `-0.005710`
- `lag_10__CT_flashes_last_5s`: contribution `-0.002634`
- `lag_12__utility_inv_diff`: contribution `-0.000566`

### tick `37832`, seconds `39.50`, LSTM delta `+0.0627`

Top all feature movements:
- `lag_08__CT_place_TUNNELSTAIRS`: contribution `+0.007397`
- `lag_07__T_place_SIDE`: contribution `+0.002727`
- `lag_11__CT_place_UPPERTUNNEL`: contribution `+0.002633`
- `lag_15__T_shots_fired_sum`: contribution `+0.001504`
- `lag_08__CT_place_LOWERTUNNEL`: contribution `+0.001295`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `35336`, seconds `0.50`, LSTM delta `-0.0590`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.002767`
- `lag_00__T_velocity_mean`: contribution `-0.002036`
- `lag_01__T_place_TSPAWN`: contribution `-0.001948`
- `lag_00__CT_velocity_mean`: contribution `-0.001777`
- `lag_01__utility_inv_diff`: contribution `-0.001014`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `-0.001014`
- `lag_01__smoke_inv_diff`: contribution `-0.000781`
- `lag_01__flash_inv_diff`: contribution `-0.000763`
- `lag_01__CT1__molly`: contribution `-0.000595`
- `lag_01__T1__utility_total`: contribution `-0.000579`
