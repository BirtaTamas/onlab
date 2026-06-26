# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-furia-vs-3dmax-bo3-Oe166BQltZjvHlE8qlepgF/furia-vs-3dmax-m1-nuke.csv`
- round_num: `6`

## Largest probability jumps

- tick `39838`, seconds `83.00`, LSTM `0.8682`, delta `+0.3123`
- tick `36158`, seconds `25.50`, LSTM `0.6941`, delta `+0.1416`
- tick `39038`, seconds `70.50`, LSTM `0.8589`, delta `+0.1232`
- tick `39102`, seconds `71.50`, LSTM `0.8196`, delta `-0.0654`
- tick `39230`, seconds `73.50`, LSTM `0.7773`, delta `-0.0644`
- tick `39006`, seconds `70.00`, LSTM `0.7356`, delta `+0.0571`
- tick `39518`, seconds `78.00`, LSTM `0.6161`, delta `-0.0553`
- tick `39870`, seconds `83.50`, LSTM `0.9156`, delta `+0.0474`
- tick `36190`, seconds `26.00`, LSTM `0.6588`, delta `-0.0352`
- tick `38494`, seconds `62.00`, LSTM `0.6654`, delta `+0.0297`

## Top 15 local ridge features

- `lag_10__CT_place_VENDING`: coefficient `-0.003885`, |coef| `0.003885`
- `lag_00__CT_kills_last_3s`: coefficient `0.002541`, |coef| `0.002541`
- `lag_07__T_place_MINI`: coefficient `-0.002229`, |coef| `0.002229`
- `lag_12__T_place_MINI`: coefficient `0.002229`, |coef| `0.002229`
- `lag_00__kill_diff_last_3s`: coefficient `0.002026`, |coef| `0.002026`
- `lag_10__CT_place_LOBBY`: coefficient `0.001962`, |coef| `0.001962`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001947`, |coef| `0.001947`
- `lag_02__T_place_MINI`: coefficient `0.001788`, |coef| `0.001788`
- `lag_00__CT_place_RAFTERS`: coefficient `0.001505`, |coef| `0.001505`
- `lag_00__CT_damage_last_5s`: coefficient `0.001383`, |coef| `0.001383`
- `lag_01__T_place_MINI`: coefficient `0.001348`, |coef| `0.001348`
- `lag_00__damage_diff_last_5s`: coefficient `0.001326`, |coef| `0.001326`
- `lag_07__T_flashed_players`: coefficient `0.001270`, |coef| `0.001270`
- `lag_08__T5__flash_duration`: coefficient `-0.001210`, |coef| `0.001210`
- `lag_04__T_place_MINI`: coefficient `-0.001201`, |coef| `0.001201`

## Top 10 utility ridge features

- `lag_08__T5__flash_duration`: coefficient `-0.001210` (lowers CT win probability)
- `lag_00__T2__flash`: coefficient `-0.000707` (lowers CT win probability)
- `lag_07__T2__flash_duration`: coefficient `0.000646` (raises CT win probability)
- `lag_11__T_A_site_active_infernos`: coefficient `0.000515` (raises CT win probability)
- `lag_00__T2__utility_total`: coefficient `-0.000500` (lowers CT win probability)
- `lag_11__T_B_site_active_infernos`: coefficient `0.000489` (raises CT win probability)
- `lag_02__T2__flash_duration`: coefficient `-0.000484` (lowers CT win probability)
- `lag_06__T2__flash_duration`: coefficient `0.000481` (raises CT win probability)
- `lag_15__T1__molly`: coefficient `-0.000480` (lowers CT win probability)
- `lag_12__CT_A_site_active_infernos`: coefficient `0.000473` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_10__CT_place_VENDING`: coefficient `-0.003885` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002541` (raises CT win probability)
- `lag_07__T_place_MINI`: coefficient `-0.002229` (lowers CT win probability)
- `lag_12__T_place_MINI`: coefficient `0.002229` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002026` (raises CT win probability)
- `lag_10__CT_place_LOBBY`: coefficient `0.001962` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001947` (raises CT win probability)
- `lag_02__T_place_MINI`: coefficient `0.001788` (raises CT win probability)
- `lag_00__CT_place_RAFTERS`: coefficient `0.001505` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001383` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `39838`, seconds `83.00`, LSTM delta `+0.3123`

Top all feature movements:
- `lag_10__CT_place_VENDING`: contribution `+0.066582`
- `lag_07__T_place_MINI`: contribution `+0.031011`
- `lag_12__T_place_MINI`: contribution `+0.031009`
- `lag_04__T_place_MINI`: contribution `+0.016715`
- `lag_10__CT_place_LOBBY`: contribution `+0.016057`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `36158`, seconds `25.50`, LSTM delta `+0.1416`

Top all feature movements:
- `lag_08__T5__flash_duration`: contribution `+0.008607`
- `lag_00__CT_kills_last_3s`: contribution `+0.007337`
- `lag_00__CT_shots_fired_sum`: contribution `+0.006764`
- `lag_06__CT_place_MINI`: contribution `+0.006011`
- `lag_00__T_place_SQUEAKY`: contribution `+0.005732`

Top utility-only movements:
- `lag_08__T5__flash_duration`: contribution `+0.008607`

### tick `39038`, seconds `70.50`, LSTM delta `+0.1232`

Top all feature movements:
- `lag_01__T_place_MINI`: contribution `+0.018752`
- `lag_03__T_place_MINI`: contribution `+0.012793`
- `lag_00__CT_place_RAFTERS`: contribution `+0.008044`
- `lag_00__CT_kills_last_3s`: contribution `+0.007337`
- `lag_12__CT_place_LOCKERROOM`: contribution `+0.005604`

Top utility-only movements:
- `lag_07__T2__flash_duration`: contribution `+0.001838`

### tick `39102`, seconds `71.50`, LSTM delta `-0.0654`

Top all feature movements:
- `lag_02__T_place_MINI`: contribution `-0.024881`
- `lag_00__CT_place_RAFTERS`: contribution `-0.016089`
- `lag_03__T_place_MINI`: contribution `+0.012793`
- `lag_05__T_place_MINI`: contribution `-0.012309`
- `lag_00__CT_shots_fired_sum`: contribution `-0.005411`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `39230`, seconds `73.50`, LSTM delta `-0.0644`

Top all feature movements:
- `lag_07__T_place_MINI`: contribution `-0.031011`
- `lag_04__T_place_MINI`: contribution `-0.016715`
- `lag_04__CT_place_RAFTERS`: contribution `-0.007844`
- `lag_09__T_place_MINI`: contribution `-0.007834`
- `lag_00__CT_kills_last_3s`: contribution `-0.007337`

Top utility-only movements:
- `lag_08__T2__flash_duration`: contribution `-0.000916`
