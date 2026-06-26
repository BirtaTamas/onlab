# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-astralis-vs-natus-vincere-bo3-4-6Sb81TUo41h9OxcK0xKz/astralis-vs-natus-vincere-m3-nuke.csv`
- round_num: `3`

## Largest probability jumps

- tick `35021`, seconds `75.00`, LSTM `0.0521`, delta `-0.0931`
- tick `30253`, seconds `0.50`, LSTM `0.1227`, delta `-0.0686`
- tick `31245`, seconds `16.00`, LSTM `0.1763`, delta `+0.0507`
- tick `33997`, seconds `59.00`, LSTM `0.2407`, delta `-0.0460`
- tick `34189`, seconds `62.00`, LSTM `0.2235`, delta `-0.0420`
- tick `34381`, seconds `65.00`, LSTM `0.1900`, delta `-0.0418`
- tick `36141`, seconds `92.50`, LSTM `0.0181`, delta `-0.0414`
- tick `34989`, seconds `74.50`, LSTM `0.1452`, delta `-0.0410`
- tick `32333`, seconds `33.00`, LSTM `0.1754`, delta `-0.0407`
- tick `33837`, seconds `56.50`, LSTM `0.2423`, delta `+0.0387`

## Top 15 local ridge features

- `lag_00__CT_place_MINI`: coefficient `-0.001460`, |coef| `0.001460`
- `lag_14__T_place_GARAGE`: coefficient `-0.001191`, |coef| `0.001191`
- `lag_15__T_place_GARAGE`: coefficient `-0.000988`, |coef| `0.000988`
- `lag_00__CT2__is_walking`: coefficient `-0.000938`, |coef| `0.000938`
- `lag_13__CT_place_SECRET`: coefficient `-0.000864`, |coef| `0.000864`
- `lag_00__T_place_ROOF`: coefficient `-0.000785`, |coef| `0.000785`
- `lag_14__CT_place_MINI`: coefficient `0.000757`, |coef| `0.000757`
- `lag_09__CT_place_OBSERVATION`: coefficient `-0.000753`, |coef| `0.000753`
- `lag_00__CT_place_CRANE`: coefficient `-0.000712`, |coef| `0.000712`
- `lag_08__T3__duck_amount`: coefficient `-0.000701`, |coef| `0.000701`
- `lag_01__CT_place_MINI`: coefficient `-0.000679`, |coef| `0.000679`
- `lag_08__CT_place_HUTROOF`: coefficient `-0.000678`, |coef| `0.000678`
- `lag_15__CT_place_MINI`: coefficient `0.000673`, |coef| `0.000673`
- `lag_00__CT_place_HEAVEN`: coefficient `0.000620`, |coef| `0.000620`
- `lag_14__CT_place_SECRET`: coefficient `-0.000617`, |coef| `0.000617`

## Top 10 utility ridge features

- `lag_11__T_A_site_active_smokes`: coefficient `0.000461` (raises CT win probability)
- `lag_01__T1__smoke`: coefficient `-0.000454` (lowers CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000430` (raises CT win probability)
- `lag_00__CT_B_site_active_smokes`: coefficient `0.000408` (raises CT win probability)
- `lag_02__T_active_infernos`: coefficient `0.000407` (raises CT win probability)
- `lag_04__T_A_site_active_smokes`: coefficient `0.000405` (raises CT win probability)
- `lag_00__CT_A_site_active_smokes`: coefficient `0.000368` (raises CT win probability)
- `lag_10__T_active_infernos`: coefficient `0.000360` (raises CT win probability)
- `lag_01__T1__molly`: coefficient `-0.000356` (lowers CT win probability)
- `lag_11__T_active_smokes`: coefficient `0.000354` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_MINI`: coefficient `-0.001460` (lowers CT win probability)
- `lag_14__T_place_GARAGE`: coefficient `-0.001191` (lowers CT win probability)
- `lag_15__T_place_GARAGE`: coefficient `-0.000988` (lowers CT win probability)
- `lag_00__CT2__is_walking`: coefficient `-0.000938` (lowers CT win probability)
- `lag_13__CT_place_SECRET`: coefficient `-0.000864` (lowers CT win probability)
- `lag_00__T_place_ROOF`: coefficient `-0.000785` (lowers CT win probability)
- `lag_14__CT_place_MINI`: coefficient `0.000757` (raises CT win probability)
- `lag_09__CT_place_OBSERVATION`: coefficient `-0.000753` (lowers CT win probability)
- `lag_00__CT_place_CRANE`: coefficient `-0.000712` (lowers CT win probability)
- `lag_08__T3__duck_amount`: coefficient `-0.000701` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `35021`, seconds `75.00`, LSTM delta `-0.0931`

Top all feature movements:
- `lag_14__T_place_GARAGE`: contribution `-0.014316`
- `lag_15__T_place_GARAGE`: contribution `-0.011881`
- `lag_00__CT_place_CRANE`: contribution `-0.011687`
- `lag_08__CT_place_HUTROOF`: contribution `-0.004744`
- `lag_15__CT_place_MINI`: contribution `-0.004125`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `30253`, seconds `0.50`, LSTM delta `-0.0686`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.002941`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.002509`
- `lag_01__T_place_TSPAWN`: contribution `-0.002459`
- `lag_01__T_closest_enemy_dist`: contribution `-0.002319`
- `lag_01__centroid_distance_xy`: contribution `-0.002127`

Top utility-only movements:
- `lag_01__molly_inv_diff`: contribution `-0.001201`
- `lag_01__T_molly_inv`: contribution `-0.000777`
- `lag_01__T_smoke_inv`: contribution `-0.000757`
- `lag_01__T1__smoke`: contribution `-0.000674`
- `lag_01__T1__molly`: contribution `-0.000563`

### tick `31245`, seconds `16.00`, LSTM delta `+0.0507`

Top all feature movements:
- `lag_03__CT_place_OBSERVATION`: contribution `+0.008603`
- `lag_07__CT_place_HUT`: contribution `+0.005261`
- `lag_09__CT_place_HEAVEN`: contribution `+0.001903`
- `lag_09__T_shots_fired_sum`: contribution `+0.001769`
- `lag_13__T3__duck_amount`: contribution `+0.001561`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `33997`, seconds `59.00`, LSTM delta `-0.0460`

Top all feature movements:
- `lag_05__CT_place_HUT`: contribution `-0.004564`
- `lag_05__CT_place_LOBBY`: contribution `-0.004082`
- `lag_04__CT_place_HUT`: contribution `-0.004018`
- `lag_00__CT_place_HEAVEN`: contribution `-0.003348`
- `lag_03__CT_place_VENTS`: contribution `-0.002813`

Top utility-only movements:
- `lag_04__T_A_site_active_smokes`: contribution `-0.001152`

### tick `34189`, seconds `62.00`, LSTM delta `-0.0420`

Top all feature movements:
- `lag_11__CT_place_HUT`: contribution `-0.005512`
- `lag_05__CT_place_HUT`: contribution `-0.004564`
- `lag_05__CT_place_LOBBY`: contribution `-0.004082`
- `lag_10__CT_place_LOBBY`: contribution `-0.003813`
- `lag_00__CT_place_VENTS`: contribution `-0.002314`

Top utility-only movements:
- No utility movement among the top local contributors.
