# Local Round Explainability

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-vitality-vs-faze-bo3-hDX5yjYYbla4cw8aPwAYi3/vitality-vs-faze-m1-nuke.csv`
- round_num: `8`

## Largest probability jumps

- tick `55060`, seconds `34.50`, LSTM `0.1114`, delta `-0.2028`
- tick `53716`, seconds `13.50`, LSTM `0.2824`, delta `+0.1666`
- tick `54580`, seconds `27.00`, LSTM `0.3450`, delta `-0.1031`
- tick `53780`, seconds `14.50`, LSTM `0.4011`, delta `+0.1006`
- tick `54900`, seconds `32.00`, LSTM `0.3361`, delta `+0.0869`
- tick `54548`, seconds `26.50`, LSTM `0.4480`, delta `-0.0777`
- tick `53844`, seconds `15.50`, LSTM `0.5257`, delta `+0.0708`
- tick `54036`, seconds `18.50`, LSTM `0.5373`, delta `-0.0617`
- tick `54804`, seconds `30.50`, LSTM `0.2033`, delta `-0.0583`
- tick `53812`, seconds `15.00`, LSTM `0.4549`, delta `+0.0538`

## Top 15 local ridge features

- `lag_04__CT_place_HUT`: coefficient `0.001695`, |coef| `0.001695`
- `lag_13__T_place_DECON`: coefficient `0.001409`, |coef| `0.001409`
- `lag_05__CT_place_HUT`: coefficient `0.001311`, |coef| `0.001311`
- `lag_12__T_place_DECON`: coefficient `0.001299`, |coef| `0.001299`
- `lag_15__T_place_MINI`: coefficient `-0.001261`, |coef| `0.001261`
- `lag_13__CT_place_HUT`: coefficient `0.001112`, |coef| `0.001112`
- `lag_01__T_place_SQUEAKY`: coefficient `-0.001104`, |coef| `0.001104`
- `lag_15__CT_place_HEAVEN`: coefficient `0.001088`, |coef| `0.001088`
- `lag_00__T_place_VENTS`: coefficient `0.001012`, |coef| `0.001012`
- `lag_14__T_place_DECON`: coefficient `0.001010`, |coef| `0.001010`
- `lag_08__CT_place_VENDING`: coefficient `-0.000920`, |coef| `0.000920`
- `lag_12__CT_place_SQUEAKY`: coefficient `-0.000900`, |coef| `0.000900`
- `lag_03__T_place_VENTS`: coefficient `0.000897`, |coef| `0.000897`
- `lag_01__CT_place_HUT`: coefficient `0.000887`, |coef| `0.000887`
- `lag_14__CT_place_HEAVEN`: coefficient `0.000851`, |coef| `0.000851`

## Top 10 utility ridge features

- `lag_00__T_mollies_last_5s`: coefficient `-0.000510` (lowers CT win probability)
- `lag_03__T1__flash_duration`: coefficient `0.000459` (raises CT win probability)
- `lag_05__CT1__flash_duration`: coefficient `0.000443` (raises CT win probability)
- `lag_02__CT_A_site_active_infernos`: coefficient `0.000413` (raises CT win probability)
- `lag_02__CT_B_site_active_infernos`: coefficient `0.000402` (raises CT win probability)
- `lag_07__T_A_site_active_infernos`: coefficient `0.000385` (raises CT win probability)
- `lag_03__T_A_site_active_infernos`: coefficient `0.000375` (raises CT win probability)
- `lag_07__T_B_site_active_infernos`: coefficient `0.000365` (raises CT win probability)
- `lag_15__T_mollies_last_5s`: coefficient `-0.000358` (lowers CT win probability)
- `lag_07__CT1__flash_duration`: coefficient `0.000356` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_04__CT_place_HUT`: coefficient `0.001695` (raises CT win probability)
- `lag_13__T_place_DECON`: coefficient `0.001409` (raises CT win probability)
- `lag_05__CT_place_HUT`: coefficient `0.001311` (raises CT win probability)
- `lag_12__T_place_DECON`: coefficient `0.001299` (raises CT win probability)
- `lag_15__T_place_MINI`: coefficient `-0.001261` (lowers CT win probability)
- `lag_13__CT_place_HUT`: coefficient `0.001112` (raises CT win probability)
- `lag_01__T_place_SQUEAKY`: coefficient `-0.001104` (lowers CT win probability)
- `lag_15__CT_place_HEAVEN`: coefficient `0.001088` (raises CT win probability)
- `lag_00__T_place_VENTS`: coefficient `0.001012` (raises CT win probability)
- `lag_14__T_place_DECON`: coefficient `0.001010` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `55060`, seconds `34.50`, LSTM delta `-0.2028`

Top all feature movements:
- `lag_15__T_place_MINI`: contribution `-0.017541`
- `lag_08__CT_place_VENDING`: contribution `-0.015774`
- `lag_12__CT_place_SQUEAKY`: contribution `-0.011966`
- `lag_05__CT_place_VENDING`: contribution `-0.011587`
- `lag_13__CT_place_HUT`: contribution `-0.010847`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `53716`, seconds `13.50`, LSTM delta `+0.1666`

Top all feature movements:
- `lag_04__CT_place_HUT`: contribution `+0.016535`
- `lag_01__T_place_SQUEAKY`: contribution `+0.006876`
- `lag_14__CT_place_HEAVEN`: contribution `+0.004594`
- `lag_02__CT_place_MINI`: contribution `+0.004472`
- `lag_04__CT_place_HEAVEN`: contribution `+0.004394`

Top utility-only movements:
- `lag_03__T1__flash_duration`: contribution `+0.002389`
- `lag_05__CT1__flash_duration`: contribution `+0.002065`

### tick `54580`, seconds `27.00`, LSTM delta `-0.1031`

Top all feature movements:
- `lag_13__T_place_DECON`: contribution `-0.022630`
- `lag_05__CT_place_HUT`: contribution `-0.012783`
- `lag_00__T_place_MINI`: contribution `-0.006205`
- `lag_05__CT_place_LOBBY`: contribution `-0.005817`
- `lag_15__T_place_DECON`: contribution `-0.005731`

Top utility-only movements:
- `lag_03__T_A_site_active_infernos`: contribution `-0.001117`
- `lag_03__T_B_site_active_infernos`: contribution `-0.001006`

### tick `53780`, seconds `14.50`, LSTM delta `+0.1006`

Top all feature movements:
- `lag_00__T_place_VENTS`: contribution `+0.013655`
- `lag_01__T_place_SQUEAKY`: contribution `+0.006876`
- `lag_15__CT_place_HEAVEN`: contribution `+0.005876`
- `lag_06__CT_place_HUT`: contribution `+0.004755`
- `lag_06__CT_place_HEAVEN`: contribution `+0.003735`

Top utility-only movements:
- `lag_05__T1__flash_duration`: contribution `+0.001786`
- `lag_07__CT1__flash_duration`: contribution `+0.001659`

### tick `54900`, seconds `32.00`, LSTM delta `+0.0869`

Top all feature movements:
- `lag_13__CT_place_HUT`: contribution `+0.010847`
- `lag_01__CT_place_HUT`: contribution `+0.008649`
- `lag_00__CT_place_VENDING`: contribution `+0.008095`
- `lag_03__CT_place_VENDING`: contribution `+0.006992`
- `lag_07__CT_place_SQUEAKY`: contribution `+0.006976`

Top utility-only movements:
- No utility movement among the top local contributors.
