# Local Round Explainability

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-the-mongolz-vs-3dmax-bo3-NhOpC3bR-AJd86c-60IeuJ/the-mongolz-vs-3dmax-m1-nuke.csv`
- round_num: `8`

## Largest probability jumps

- tick `53147`, seconds `49.50`, LSTM `0.8479`, delta `+0.1617`
- tick `53403`, seconds `53.50`, LSTM `0.6214`, delta `-0.1422`
- tick `52987`, seconds `47.00`, LSTM `0.6694`, delta `+0.0957`
- tick `54267`, seconds `67.00`, LSTM `0.8622`, delta `+0.0704`
- tick `53563`, seconds `56.00`, LSTM `0.6355`, delta `+0.0604`
- tick `53787`, seconds `59.50`, LSTM `0.7669`, delta `+0.0558`
- tick `53691`, seconds `58.00`, LSTM `0.7085`, delta `+0.0438`
- tick `53371`, seconds `53.00`, LSTM `0.7636`, delta `-0.0417`
- tick `53883`, seconds `61.00`, LSTM `0.7163`, delta `-0.0372`
- tick `53947`, seconds `62.00`, LSTM `0.7484`, delta `+0.0291`

## Top 15 local ridge features

- `lag_02__CT_place_TROPHY`: coefficient `0.001493`, |coef| `0.001493`
- `lag_03__CT_place_HUT`: coefficient `0.001068`, |coef| `0.001068`
- `lag_00__CT_place_MINI`: coefficient `0.000999`, |coef| `0.000999`
- `lag_10__CT_place_TROPHY`: coefficient `-0.000992`, |coef| `0.000992`
- `lag_05__CT_place_VENDING`: coefficient `-0.000974`, |coef| `0.000974`
- `lag_12__CT_place_CONTROL`: coefficient `0.000896`, |coef| `0.000896`
- `lag_02__CT_place_CONTROL`: coefficient `-0.000848`, |coef| `0.000848`
- `lag_07__CT_place_CONTROL`: coefficient `0.000839`, |coef| `0.000839`
- `lag_13__CT_place_SILO`: coefficient `0.000806`, |coef| `0.000806`
- `lag_00__T4__shots_fired`: coefficient `0.000734`, |coef| `0.000734`
- `lag_04__CT_place_VENDING`: coefficient `-0.000706`, |coef| `0.000706`
- `lag_10__T_place_SQUEAKY`: coefficient `0.000701`, |coef| `0.000701`
- `lag_06__T_A_site_active_infernos`: coefficient `0.000680`, |coef| `0.000680`
- `lag_05__T_place_SQUEAKY`: coefficient `0.000660`, |coef| `0.000660`
- `lag_00__CT_kills_last_3s`: coefficient `0.000657`, |coef| `0.000657`

## Top 10 utility ridge features

- `lag_06__T_A_site_active_infernos`: coefficient `0.000680` (raises CT win probability)
- `lag_06__T_B_site_active_infernos`: coefficient `0.000646` (raises CT win probability)
- `lag_06__T_active_infernos`: coefficient `0.000470` (raises CT win probability)
- `lag_01__T4__flash_duration`: coefficient `0.000412` (raises CT win probability)
- `lag_01__T_A_site_active_infernos`: coefficient `0.000383` (raises CT win probability)
- `lag_01__T_B_site_active_infernos`: coefficient `0.000363` (raises CT win probability)
- `lag_11__T_A_site_active_infernos`: coefficient `0.000349` (raises CT win probability)
- `lag_00__T1__flash`: coefficient `-0.000344` (lowers CT win probability)
- `lag_11__T_B_site_active_infernos`: coefficient `0.000331` (raises CT win probability)
- `lag_06__active_infernos_total`: coefficient `0.000330` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_02__CT_place_TROPHY`: coefficient `0.001493` (raises CT win probability)
- `lag_03__CT_place_HUT`: coefficient `0.001068` (raises CT win probability)
- `lag_00__CT_place_MINI`: coefficient `0.000999` (raises CT win probability)
- `lag_10__CT_place_TROPHY`: coefficient `-0.000992` (lowers CT win probability)
- `lag_05__CT_place_VENDING`: coefficient `-0.000974` (lowers CT win probability)
- `lag_12__CT_place_CONTROL`: coefficient `0.000896` (raises CT win probability)
- `lag_02__CT_place_CONTROL`: coefficient `-0.000848` (lowers CT win probability)
- `lag_07__CT_place_CONTROL`: coefficient `0.000839` (raises CT win probability)
- `lag_13__CT_place_SILO`: coefficient `0.000806` (raises CT win probability)
- `lag_00__T4__shots_fired`: coefficient `0.000734` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `53147`, seconds `49.50`, LSTM delta `+0.1617`

Top all feature movements:
- `lag_02__CT_place_TROPHY`: contribution `+0.022048`
- `lag_03__CT_place_HUT`: contribution `+0.010415`
- `lag_12__CT_place_CONTROL`: contribution `+0.009304`
- `lag_02__CT_place_CONTROL`: contribution `+0.008799`
- `lag_10__T_place_SQUEAKY`: contribution `+0.004366`

Top utility-only movements:
- `lag_06__T_A_site_active_infernos`: contribution `+0.002025`
- `lag_06__T_B_site_active_infernos`: contribution `+0.001828`
- `lag_01__T4__flash_duration`: contribution `+0.001462`

### tick `53403`, seconds `53.50`, LSTM delta `-0.1422`

Top all feature movements:
- `lag_05__CT_place_VENDING`: contribution `-0.016686`
- `lag_10__CT_place_TROPHY`: contribution `-0.014658`
- `lag_03__CT_place_HUT`: contribution `-0.010415`
- `lag_05__CT_place_TROPHY`: contribution `-0.008597`
- `lag_00__T4__shots_fired`: contribution `-0.008156`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `52987`, seconds `47.00`, LSTM delta `+0.0957`

Top all feature movements:
- `lag_07__CT_place_CONTROL`: contribution `+0.008704`
- `lag_00__CT_place_MINI`: contribution `+0.006123`
- `lag_05__T_place_SQUEAKY`: contribution `+0.004110`
- `lag_06__T_A_site_active_infernos`: contribution `+0.002025`
- `lag_00__CT_kills_last_3s`: contribution `+0.001896`

Top utility-only movements:
- `lag_06__T_A_site_active_infernos`: contribution `+0.002025`
- `lag_06__T_B_site_active_infernos`: contribution `+0.001828`
- `lag_01__T_A_site_active_infernos`: contribution `+0.001139`
- `lag_01__T_B_site_active_infernos`: contribution `+0.001028`

### tick `54267`, seconds `67.00`, LSTM delta `+0.0704`

Top all feature movements:
- `lag_13__CT_place_SILO`: contribution `+0.053939`
- `lag_13__CT_place_ROOF`: contribution `+0.002641`
- `lag_00__CT_kills_last_3s`: contribution `+0.001896`
- `lag_10__CT_place_LOCKERROOM`: contribution `+0.001860`
- `lag_15__CT_place_SQUEAKY`: contribution `-0.001723`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `53563`, seconds `56.00`, LSTM delta `+0.0604`

Top all feature movements:
- `lag_10__CT_place_TROPHY`: contribution `+0.014658`
- `lag_00__CT_place_ROOF`: contribution `+0.012181`
- `lag_04__CT_place_VENDING`: contribution `+0.012104`
- `lag_15__CT_place_CONTROL`: contribution `+0.004558`
- `lag_10__CT_place_VENDING`: contribution `+0.004079`

Top utility-only movements:
- `lag_05__T_A_site_active_infernos`: contribution `-0.000872`
