# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-astralis-vs-natus-vincere-bo3-4-6Sb81TUo41h9OxcK0xKz/astralis-vs-natus-vincere-m3-nuke.csv`
- round_num: `4`

## Largest probability jumps

- tick `48185`, seconds `101.50`, LSTM `0.1392`, delta `-0.3307`
- tick `48153`, seconds `101.00`, LSTM `0.4699`, delta `-0.1969`
- tick `45241`, seconds `55.50`, LSTM `0.7756`, delta `+0.1423`
- tick `44697`, seconds `47.00`, LSTM `0.7261`, delta `+0.1016`
- tick `47737`, seconds `94.50`, LSTM `0.9487`, delta `+0.0911`
- tick `47833`, seconds `96.00`, LSTM `0.8502`, delta `-0.0838`
- tick `47865`, seconds `96.50`, LSTM `0.7705`, delta `-0.0797`
- tick `47929`, seconds `97.50`, LSTM `0.6801`, delta `-0.0630`
- tick `47353`, seconds `88.50`, LSTM `0.7587`, delta `-0.0611`
- tick `44665`, seconds `46.50`, LSTM `0.6245`, delta `+0.0610`

## Top 15 local ridge features

- `lag_12__T_place_HEAVEN`: coefficient `-0.002787`, |coef| `0.002787`
- `lag_11__T_place_HEAVEN`: coefficient `-0.002637`, |coef| `0.002637`
- `lag_00__kill_diff_last_3s`: coefficient `0.002329`, |coef| `0.002329`
- `lag_09__T_shots_fired_sum`: coefficient `0.002148`, |coef| `0.002148`
- `lag_00__T_place_ADMIN`: coefficient `0.002108`, |coef| `0.002108`
- `lag_00__T_bomb_zone_count`: coefficient `-0.001900`, |coef| `0.001900`
- `lag_09__T4__shots_fired`: coefficient `0.001687`, |coef| `0.001687`
- `lag_11__T_shots_fired_sum`: coefficient `-0.001643`, |coef| `0.001643`
- `lag_10__T_place_HEAVEN`: coefficient `-0.001637`, |coef| `0.001637`
- `lag_01__CT_place_VENTS`: coefficient `0.001618`, |coef| `0.001618`
- `lag_10__T_shots_fired_sum`: coefficient `-0.001580`, |coef| `0.001580`
- `lag_00__T_kills_last_3s`: coefficient `-0.001549`, |coef| `0.001549`
- `lag_08__T4__shots_fired`: coefficient `0.001522`, |coef| `0.001522`
- `lag_14__T_place_HEAVEN`: coefficient `-0.001487`, |coef| `0.001487`
- `lag_13__T_place_HEAVEN`: coefficient `-0.001471`, |coef| `0.001471`

## Top 10 utility ridge features

- `lag_14__T3__flash`: coefficient `0.000977` (raises CT win probability)
- `lag_13__T3__flash`: coefficient `0.000658` (raises CT win probability)
- `lag_01__CT4__flash`: coefficient `0.000562` (raises CT win probability)
- `lag_00__T_active_infernos`: coefficient `-0.000545` (lowers CT win probability)
- `lag_14__T3__utility_total`: coefficient `0.000543` (raises CT win probability)
- `lag_07__CT_B_site_active_infernos`: coefficient `0.000542` (raises CT win probability)
- `lag_00__T_A_site_active_infernos`: coefficient `-0.000480` (lowers CT win probability)
- `lag_07__CT_A_site_active_infernos`: coefficient `0.000468` (raises CT win probability)
- `lag_00__CT4__flash`: coefficient `0.000461` (raises CT win probability)
- `lag_00__T_B_site_active_infernos`: coefficient `-0.000456` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_12__T_place_HEAVEN`: coefficient `-0.002787` (lowers CT win probability)
- `lag_11__T_place_HEAVEN`: coefficient `-0.002637` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002329` (raises CT win probability)
- `lag_09__T_shots_fired_sum`: coefficient `0.002148` (raises CT win probability)
- `lag_00__T_place_ADMIN`: coefficient `0.002108` (raises CT win probability)
- `lag_00__T_bomb_zone_count`: coefficient `-0.001900` (lowers CT win probability)
- `lag_09__T4__shots_fired`: coefficient `0.001687` (raises CT win probability)
- `lag_11__T_shots_fired_sum`: coefficient `-0.001643` (lowers CT win probability)
- `lag_10__T_place_HEAVEN`: coefficient `-0.001637` (lowers CT win probability)
- `lag_01__CT_place_VENTS`: coefficient `0.001618` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `48185`, seconds `101.50`, LSTM delta `-0.3307`

Top all feature movements:
- `lag_12__T_place_HEAVEN`: contribution `-0.034197`
- `lag_09__T_shots_fired_sum`: contribution `-0.024159`
- `lag_12__T_place_HELL`: contribution `-0.020271`
- `lag_09__T4__shots_fired`: contribution `-0.015631`
- `lag_01__CT_place_VENTS`: contribution `-0.013577`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `48153`, seconds `101.00`, LSTM delta `-0.1969`

Top all feature movements:
- `lag_11__T_place_HEAVEN`: contribution `-0.032355`
- `lag_11__T_place_HELL`: contribution `-0.014967`
- `lag_08__T4__shots_fired`: contribution `-0.014104`
- `lag_08__T_shots_fired_sum`: contribution `-0.013387`
- `lag_09__T_shots_fired_sum`: contribution `+0.009664`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `45241`, seconds `55.50`, LSTM delta `+0.1423`

Top all feature movements:
- `lag_08__CT_place_DECON`: contribution `+0.020170`
- `lag_11__T_shots_fired_sum`: contribution `+0.014780`
- `lag_09__T_shots_fired_sum`: contribution `+0.008053`
- `lag_01__CT_place_MINI`: contribution `+0.007523`
- `lag_10__T_place_SILO`: contribution `+0.007516`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `44697`, seconds `47.00`, LSTM delta `+0.1016`

Top all feature movements:
- `lag_15__CT_place_MINI`: contribution `+0.007976`
- `lag_00__kill_diff_last_3s`: contribution `+0.005606`
- `lag_12__CT_place_VENTS`: contribution `+0.005355`
- `lag_00__CT_kills_last_3s`: contribution `+0.003989`
- `lag_01__T_place_SECRET`: contribution `+0.003720`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `47737`, seconds `94.50`, LSTM delta `+0.0911`

Top all feature movements:
- `lag_06__T_place_HELL`: contribution `+0.020569`
- `lag_10__T_place_SQUEAKY`: contribution `+0.006989`
- `lag_15__T_place_SQUEAKY`: contribution `-0.006489`
- `lag_00__kill_diff_last_3s`: contribution `+0.005606`
- `lag_00__CT_kills_last_3s`: contribution `+0.003989`

Top utility-only movements:
- No utility movement among the top local contributors.
