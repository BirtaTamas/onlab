# Local Round Explainability

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-faze-vs-bcgame-bo3-daIGc_M_y7Qq42fF9AoQhi/faze-vs-bcgame-m2-anubis.csv`
- round_num: `3`

## Largest probability jumps

- tick `14264`, seconds `42.00`, LSTM `0.8546`, delta `+0.2029`
- tick `12632`, seconds `16.50`, LSTM `0.6249`, delta `+0.1296`
- tick `14296`, seconds `42.50`, LSTM `0.9341`, delta `+0.0795`
- tick `15288`, seconds `58.00`, LSTM `0.9687`, delta `+0.0532`
- tick `12664`, seconds `17.00`, LSTM `0.6778`, delta `+0.0529`
- tick `14232`, seconds `41.50`, LSTM `0.6517`, delta `-0.0434`
- tick `13944`, seconds `37.00`, LSTM `0.7220`, delta `+0.0380`
- tick `12696`, seconds `17.50`, LSTM `0.7093`, delta `+0.0315`
- tick `12824`, seconds `19.50`, LSTM `0.7170`, delta `-0.0284`
- tick `14072`, seconds `39.00`, LSTM `0.7329`, delta `+0.0259`

## Top 15 local ridge features

- `lag_00__CT_place_CTSIDEUPPER`: coefficient `-0.001697`, |coef| `0.001697`
- `lag_03__T_place_WALKWAY`: coefficient `0.001026`, |coef| `0.001026`
- `lag_01__CT_A_site_active_infernos`: coefficient `0.000975`, |coef| `0.000975`
- `lag_14__CT_place_OUTSIDELONG`: coefficient `0.000933`, |coef| `0.000933`
- `lag_07__CT_place_BRICKS`: coefficient `-0.000850`, |coef| `0.000850`
- `lag_01__CT_active_infernos`: coefficient `0.000839`, |coef| `0.000839`
- `lag_09__CT_place_OUTSIDELONG`: coefficient `0.000818`, |coef| `0.000818`
- `lag_00__CT_shots_fired_sum`: coefficient `0.000813`, |coef| `0.000813`
- `lag_00__T1__shots_fired`: coefficient `0.000808`, |coef| `0.000808`
- `lag_01__CT_place_HEAVEN`: coefficient `0.000791`, |coef| `0.000791`
- `lag_03__CT_place_HEAVEN`: coefficient `0.000773`, |coef| `0.000773`
- `lag_00__CT_kills_last_3s`: coefficient `0.000766`, |coef| `0.000766`
- `lag_14__T_place_STREET`: coefficient `-0.000718`, |coef| `0.000718`
- `lag_13__T_place_TSTAIRS`: coefficient `0.000717`, |coef| `0.000717`
- `lag_11__CT_place_TUNNEL`: coefficient `0.000716`, |coef| `0.000716`

## Top 10 utility ridge features

- `lag_01__CT_A_site_active_infernos`: coefficient `0.000975` (raises CT win probability)
- `lag_01__CT_active_infernos`: coefficient `0.000839` (raises CT win probability)
- `lag_06__T4__flash_duration`: coefficient `0.000604` (raises CT win probability)
- `lag_02__CT_A_site_active_infernos`: coefficient `0.000549` (raises CT win probability)
- `lag_01__active_infernos_total`: coefficient `0.000522` (raises CT win probability)
- `lag_00__T4__flash_duration`: coefficient `-0.000518` (lowers CT win probability)
- `lag_14__CT4__molly`: coefficient `0.000425` (raises CT win probability)
- `lag_00__T2__molly`: coefficient `-0.000414` (lowers CT win probability)
- `lag_00__T2__smoke`: coefficient `-0.000408` (lowers CT win probability)
- `lag_00__T4__molly`: coefficient `-0.000382` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_CTSIDEUPPER`: coefficient `-0.001697` (lowers CT win probability)
- `lag_03__T_place_WALKWAY`: coefficient `0.001026` (raises CT win probability)
- `lag_14__CT_place_OUTSIDELONG`: coefficient `0.000933` (raises CT win probability)
- `lag_07__CT_place_BRICKS`: coefficient `-0.000850` (lowers CT win probability)
- `lag_09__CT_place_OUTSIDELONG`: coefficient `0.000818` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.000813` (raises CT win probability)
- `lag_00__T1__shots_fired`: coefficient `0.000808` (raises CT win probability)
- `lag_01__CT_place_HEAVEN`: coefficient `0.000791` (raises CT win probability)
- `lag_03__CT_place_HEAVEN`: coefficient `0.000773` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000766` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `14264`, seconds `42.00`, LSTM delta `+0.2029`

Top all feature movements:
- `lag_07__CT_place_BRICKS`: contribution `+0.016315`
- `lag_03__T_place_WALKWAY`: contribution `+0.013957`
- `lag_08__CT_place_BRICKS`: contribution `+0.013400`
- `lag_10__CT_place_TUNNEL`: contribution `+0.009444`
- `lag_03__CT_place_TUNNELSTAIRS`: contribution `+0.009043`

Top utility-only movements:
- `lag_01__CT_A_site_active_infernos`: contribution `+0.006883`
- `lag_01__CT_active_infernos`: contribution `+0.003867`

### tick `12632`, seconds `16.50`, LSTM delta `+0.1296`

Top all feature movements:
- `lag_14__CT_place_OUTSIDELONG`: contribution `+0.009460`
- `lag_09__CT_place_OUTSIDELONG`: contribution `+0.008294`
- `lag_13__CT_place_OUTSIDELONG`: contribution `+0.005687`
- `lag_02__CT_place_OUTSIDELONG`: contribution `+0.005443`
- `lag_13__T_place_TSTAIRS`: contribution `+0.004064`

Top utility-only movements:
- `lag_06__T4__flash_duration`: contribution `+0.003345`
- `lag_00__T4__flash_duration`: contribution `+0.002872`
- `lag_01__CT_active_infernos`: contribution `+0.001933`

### tick `14296`, seconds `42.50`, LSTM delta `+0.0795`

Top all feature movements:
- `lag_08__CT_place_BRICKS`: contribution `-0.013400`
- `lag_11__CT_place_TUNNEL`: contribution `+0.011496`
- `lag_04__T_place_WALKWAY`: contribution `+0.007881`
- `lag_09__CT_place_BRICKS`: contribution `+0.007759`
- `lag_00__T_place_WALKWAY`: contribution `+0.006440`

Top utility-only movements:
- `lag_02__CT_A_site_active_infernos`: contribution `+0.003872`

### tick `15288`, seconds `58.00`, LSTM delta `+0.0532`

Top all feature movements:
- `lag_11__CT_place_TUNNEL`: contribution `+0.011496`
- `lag_00__T_place_WALKWAY`: contribution `+0.006440`
- `lag_07__CT_place_TUNNELSTAIRS`: contribution `+0.005631`
- `lag_07__CT_place_TUNNEL`: contribution `+0.004718`
- `lag_10__T_place_WALKWAY`: contribution `+0.003324`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `12664`, seconds `17.00`, LSTM delta `+0.0529`

Top all feature movements:
- `lag_14__CT_place_OUTSIDELONG`: contribution `+0.009460`
- `lag_13__CT_place_OUTSIDELONG`: contribution `-0.005687`
- `lag_10__CT_place_OUTSIDELONG`: contribution `+0.004292`
- `lag_15__CT_place_OUTSIDELONG`: contribution `+0.004174`
- `lag_14__T_place_STREET`: contribution `+0.003945`

Top utility-only movements:
- `lag_01__T4__flash_duration`: contribution `+0.002092`
- `lag_07__T4__flash_duration`: contribution `+0.001914`
