# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-vitality-bo3-3MYCYJWfx_8le7ueost7BH/furia-vs-vitality-m1-nuke.csv`
- round_num: `21`

## Largest probability jumps

- tick `185768`, seconds `79.00`, LSTM `0.0494`, delta `-0.1660`
- tick `182120`, seconds `22.00`, LSTM `0.6770`, delta `+0.1332`
- tick `185736`, seconds `78.50`, LSTM `0.2154`, delta `-0.1123`
- tick `182824`, seconds `33.00`, LSTM `0.5335`, delta `-0.1089`
- tick `185640`, seconds `77.00`, LSTM `0.3734`, delta `-0.0930`
- tick `185608`, seconds `76.50`, LSTM `0.4665`, delta `-0.0519`
- tick `183272`, seconds `40.00`, LSTM `0.5267`, delta `+0.0456`
- tick `182344`, seconds `25.50`, LSTM `0.6663`, delta `-0.0366`
- tick `184232`, seconds `55.00`, LSTM `0.4437`, delta `+0.0306`
- tick `181704`, seconds `15.50`, LSTM `0.5610`, delta `-0.0256`

## Top 15 local ridge features

- `lag_15__T_place_SQUEAKY`: coefficient `-0.001529`, |coef| `0.001529`
- `lag_00__damage_diff_last_5s`: coefficient `0.001397`, |coef| `0.001397`
- `lag_05__CT_place_LOCKERROOM`: coefficient `0.001340`, |coef| `0.001340`
- `lag_00__kill_diff_last_3s`: coefficient `0.001316`, |coef| `0.001316`
- `lag_00__T_place_VENTS`: coefficient `-0.001253`, |coef| `0.001253`
- `lag_09__CT_place_LOBBY`: coefficient `-0.001218`, |coef| `0.001218`
- `lag_08__CT_place_LOBBY`: coefficient `-0.001156`, |coef| `0.001156`
- `lag_00__CT_place_LOBBY`: coefficient `0.001074`, |coef| `0.001074`
- `lag_08__CT_place_HUT`: coefficient `0.000996`, |coef| `0.000996`
- `lag_00__T_kills_last_3s`: coefficient `-0.000962`, |coef| `0.000962`
- `lag_09__CT_place_HUT`: coefficient `0.000961`, |coef| `0.000961`
- `lag_04__CT_place_LOCKERROOM`: coefficient `0.000949`, |coef| `0.000949`
- `lag_00__CT_place_MINI`: coefficient `0.000938`, |coef| `0.000938`
- `lag_01__CT1__is_scoped`: coefficient `-0.000929`, |coef| `0.000929`
- `lag_10__CT_place_CONTROL`: coefficient `0.000919`, |coef| `0.000919`

## Top 10 utility ridge features

- `lag_14__CT_A_site_active_infernos`: coefficient `-0.000912` (lowers CT win probability)
- `lag_13__CT_B_site_active_infernos`: coefficient `-0.000904` (lowers CT win probability)
- `lag_14__CT_B_site_active_infernos`: coefficient `-0.000901` (lowers CT win probability)
- `lag_13__CT_A_site_active_infernos`: coefficient `-0.000868` (lowers CT win probability)
- `lag_00__CT4__utility_total`: coefficient `0.000736` (raises CT win probability)
- `lag_00__CT4__smoke`: coefficient `0.000698` (raises CT win probability)
- `lag_13__CT_active_infernos`: coefficient `-0.000661` (lowers CT win probability)
- `lag_07__T_A_site_active_infernos`: coefficient `-0.000619` (lowers CT win probability)
- `lag_07__T5__flash_duration`: coefficient `-0.000596` (lowers CT win probability)
- `lag_00__CT_utility_inv`: coefficient `0.000594` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_15__T_place_SQUEAKY`: coefficient `-0.001529` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001397` (raises CT win probability)
- `lag_05__CT_place_LOCKERROOM`: coefficient `0.001340` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001316` (raises CT win probability)
- `lag_00__T_place_VENTS`: coefficient `-0.001253` (lowers CT win probability)
- `lag_09__CT_place_LOBBY`: coefficient `-0.001218` (lowers CT win probability)
- `lag_08__CT_place_LOBBY`: coefficient `-0.001156` (lowers CT win probability)
- `lag_00__CT_place_LOBBY`: coefficient `0.001074` (raises CT win probability)
- `lag_08__CT_place_HUT`: coefficient `0.000996` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.000962` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `185768`, seconds `79.00`, LSTM delta `-0.1660`

Top all feature movements:
- `lag_05__CT_place_LOCKERROOM`: contribution `-0.016677`
- `lag_15__T_place_DECON`: contribution `-0.013437`
- `lag_09__CT_place_LOBBY`: contribution `-0.009974`
- `lag_09__CT_place_HUT`: contribution `-0.009376`
- `lag_00__CT_place_LOBBY`: contribution `-0.008794`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `182120`, seconds `22.00`, LSTM delta `+0.1332`

Top all feature movements:
- `lag_15__T_place_SQUEAKY`: contribution `+0.009522`
- `lag_08__CT1__is_scoped`: contribution `+0.003914`
- `lag_12__T4__is_scoped`: contribution `+0.003878`
- `lag_00__CT1__is_scoped`: contribution `+0.003828`
- `lag_14__CT_A_site_active_infernos`: contribution `+0.003219`

Top utility-only movements:
- `lag_14__CT_A_site_active_infernos`: contribution `+0.003219`
- `lag_13__CT_B_site_active_infernos`: contribution `+0.003106`
- `lag_14__CT_B_site_active_infernos`: contribution `+0.003094`
- `lag_13__CT_A_site_active_infernos`: contribution `+0.003064`
- `lag_07__T_A_site_active_infernos`: contribution `+0.001841`

### tick `185736`, seconds `78.50`, LSTM delta `-0.1123`

Top all feature movements:
- `lag_04__CT_place_LOCKERROOM`: contribution `-0.011817`
- `lag_08__CT_place_HUT`: contribution `-0.009717`
- `lag_14__T_place_DECON`: contribution `-0.009463`
- `lag_08__CT_place_LOBBY`: contribution `-0.009461`
- `lag_09__T_place_DECON`: contribution `-0.009030`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `182824`, seconds `33.00`, LSTM delta `-0.1089`

Top all feature movements:
- `lag_01__CT_place_MINI`: contribution `-0.005558`
- `lag_14__CT_place_HUTROOF`: contribution `-0.005274`
- `lag_04__T_shots_fired_sum`: contribution `-0.004353`
- `lag_01__T_place_TROPHY`: contribution `-0.004263`
- `lag_13__T_shots_fired_sum`: contribution `-0.003786`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `185640`, seconds `77.00`, LSTM delta `-0.0930`

Top all feature movements:
- `lag_13__CT_place_TROPHY`: contribution `-0.009925`
- `lag_10__CT_place_CONTROL`: contribution `-0.009539`
- `lag_01__CT_place_LOCKERROOM`: contribution `-0.008365`
- `lag_11__T_place_DECON`: contribution `-0.008348`
- `lag_05__CT_place_LOBBY`: contribution `-0.007265`

Top utility-only movements:
- `lag_13__CT_active_infernos`: contribution `-0.001523`
