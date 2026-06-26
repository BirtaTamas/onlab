# Local Round Explainability

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-the-huns-vs-ninja-bo3-8zmdVWrC356tnVH1OFLf2Y/the-huns-vs-ninja-m2-anubis.csv`
- round_num: `8`

## Largest probability jumps

- tick `49030`, seconds `55.50`, LSTM `0.5833`, delta `+0.3378`
- tick `49638`, seconds `65.00`, LSTM `0.1373`, delta `-0.3286`
- tick `47046`, seconds `24.50`, LSTM `0.2814`, delta `-0.2855`
- tick `49382`, seconds `61.00`, LSTM `0.5042`, delta `-0.2070`
- tick `48870`, seconds `53.00`, LSTM `0.2090`, delta `+0.0985`
- tick `47078`, seconds `25.00`, LSTM `0.2082`, delta `-0.0732`
- tick `49222`, seconds `58.50`, LSTM `0.7223`, delta `+0.0716`
- tick `47110`, seconds `25.50`, LSTM `0.1423`, delta `-0.0658`
- tick `49254`, seconds `59.00`, LSTM `0.6633`, delta `-0.0590`
- tick `47142`, seconds `26.00`, LSTM `0.0910`, delta `-0.0514`

## Top 15 local ridge features

- `lag_00__CT_place_BRICKS`: coefficient `0.003066`, |coef| `0.003066`
- `lag_09__CT_place_BRICKS`: coefficient `-0.002953`, |coef| `0.002953`
- `lag_08__CT_place_BRICKS`: coefficient `0.002562`, |coef| `0.002562`
- `lag_04__CT_place_BRICKS`: coefficient `0.002463`, |coef| `0.002463`
- `lag_05__T_shots_fired_sum`: coefficient `0.002454`, |coef| `0.002454`
- `lag_06__CT_place_SNIPERSNEST`: coefficient `0.002433`, |coef| `0.002433`
- `lag_00__T_shots_fired_sum`: coefficient `-0.002397`, |coef| `0.002397`
- `lag_05__CT_damage_last_5s`: coefficient `0.002387`, |coef| `0.002387`
- `lag_00__damage_diff_last_5s`: coefficient `0.002220`, |coef| `0.002220`
- `lag_00__kill_diff_last_3s`: coefficient `0.002127`, |coef| `0.002127`
- `lag_13__CT_place_BRICKS`: coefficient `0.002103`, |coef| `0.002103`
- `lag_01__T_shots_fired_sum`: coefficient `0.002094`, |coef| `0.002094`
- `lag_01__T1__shots_fired`: coefficient `0.002020`, |coef| `0.002020`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001981`, |coef| `0.001981`
- `lag_06__T2__is_walking`: coefficient `-0.001951`, |coef| `0.001951`

## Top 10 utility ridge features

- `lag_11__CT_B_site_active_infernos`: coefficient `-0.001370` (lowers CT win probability)
- `lag_00__CT_B_site_active_infernos`: coefficient `0.001348` (raises CT win probability)
- `lag_10__T1__flash_duration`: coefficient `-0.001324` (lowers CT win probability)
- `lag_10__CT_flashes_last_5s`: coefficient `-0.001115` (lowers CT win probability)
- `lag_13__CT_B_site_active_infernos`: coefficient `-0.001063` (lowers CT win probability)
- `lag_07__CT3__flash`: coefficient `-0.001042` (lowers CT win probability)
- `lag_15__T1__flash_duration`: coefficient `-0.000955` (lowers CT win probability)
- `lag_11__active_infernos_total`: coefficient `-0.000921` (lowers CT win probability)
- `lag_11__CT_active_infernos`: coefficient `-0.000904` (lowers CT win probability)
- `lag_00__CT_active_infernos`: coefficient `0.000904` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_BRICKS`: coefficient `0.003066` (raises CT win probability)
- `lag_09__CT_place_BRICKS`: coefficient `-0.002953` (lowers CT win probability)
- `lag_08__CT_place_BRICKS`: coefficient `0.002562` (raises CT win probability)
- `lag_04__CT_place_BRICKS`: coefficient `0.002463` (raises CT win probability)
- `lag_05__T_shots_fired_sum`: coefficient `0.002454` (raises CT win probability)
- `lag_06__CT_place_SNIPERSNEST`: coefficient `0.002433` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.002397` (lowers CT win probability)
- `lag_05__CT_damage_last_5s`: coefficient `0.002387` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002220` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002127` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `49030`, seconds `55.50`, LSTM delta `+0.3378`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.016171`
- `lag_06__CT_place_SNIPERSNEST`: contribution `+0.013032`
- `lag_05__T_shots_fired_sum`: contribution `+0.011038`
- `lag_05__CT_damage_last_5s`: contribution `+0.009628`
- `lag_01__T_shots_fired_sum`: contribution `+0.007848`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `49638`, seconds `65.00`, LSTM delta `-0.3286`

Top all feature movements:
- `lag_09__CT_place_BRICKS`: contribution `-0.056705`
- `lag_08__CT_place_BRICKS`: contribution `-0.049190`
- `lag_13__CT_place_BRICKS`: contribution `-0.040390`
- `lag_15__CT_place_SNIPERSNEST`: contribution `-0.007400`
- `lag_11__T_place_CONNECTOR`: contribution `-0.006285`

Top utility-only movements:
- `lag_11__CT_B_site_active_infernos`: contribution `-0.004707`
- `lag_00__CT_B_site_active_infernos`: contribution `-0.004630`
- `lag_13__CT_B_site_active_infernos`: contribution `-0.003654`

### tick `47046`, seconds `24.50`, LSTM delta `-0.2855`

Top all feature movements:
- `lag_00__CT_place_BRICKS`: contribution `-0.058876`
- `lag_01__CT_place_FOUNTAIN`: contribution `-0.014365`
- `lag_11__CT_place_FOUNTAIN`: contribution `-0.011725`
- `lag_07__CT_place_FOUNTAIN`: contribution `-0.009662`
- `lag_10__T1__flash_duration`: contribution `-0.008921`

Top utility-only movements:
- `lag_10__T1__flash_duration`: contribution `-0.008921`
- `lag_12__CT4__flash_duration`: contribution `-0.004714`

### tick `49382`, seconds `61.00`, LSTM delta `-0.2070`

Top all feature movements:
- `lag_00__CT_place_BRICKS`: contribution `-0.058876`
- `lag_09__CT_place_BRICKS`: contribution `-0.056705`
- `lag_11__T_shots_fired_sum`: contribution `-0.007608`
- `lag_00__kill_diff_last_3s`: contribution `-0.005120`
- `lag_10__CT_shots_fired_sum`: contribution `-0.004769`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `48870`, seconds `53.00`, LSTM delta `+0.0985`

Top all feature movements:
- `lag_12__CT_place_TUNNEL`: contribution `+0.019200`
- `lag_00__T_shots_fired_sum`: contribution `-0.010781`
- `lag_12__CT_place_LOWERTUNNEL`: contribution `+0.007496`
- `lag_00__damage_diff_last_5s`: contribution `+0.007063`
- `lag_15__T1__flash_duration`: contribution `+0.006276`

Top utility-only movements:
- `lag_15__T1__flash_duration`: contribution `+0.006276`
- `lag_11__T_B_site_active_infernos`: contribution `+0.002404`
