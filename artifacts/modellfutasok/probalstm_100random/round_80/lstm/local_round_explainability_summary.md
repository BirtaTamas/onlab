# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-falcons-vs-vitality-bo3-948Z-JwufPJ8ROXkhPE5QF/falcons-vs-vitality-m2-nuke.csv`
- round_num: `15`

## Largest probability jumps

- tick `134072`, seconds `83.00`, LSTM `0.0935`, delta `-0.2633`
- tick `133944`, seconds `81.00`, LSTM `0.1424`, delta `-0.2590`
- tick `133976`, seconds `81.50`, LSTM `0.3831`, delta `+0.2407`
- tick `133912`, seconds `80.50`, LSTM `0.4014`, delta `-0.1217`
- tick `134040`, seconds `82.50`, LSTM `0.3568`, delta `-0.0601`
- tick `134008`, seconds `82.00`, LSTM `0.4169`, delta `+0.0338`
- tick `129432`, seconds `10.50`, LSTM `0.6210`, delta `+0.0288`
- tick `134360`, seconds `87.50`, LSTM `0.0320`, delta `-0.0229`
- tick `134104`, seconds `83.50`, LSTM `0.0708`, delta `-0.0227`
- tick `129720`, seconds `15.00`, LSTM `0.6064`, delta `-0.0211`

## Top 15 local ridge features

- `lag_13__CT_place_LOCKERROOM`: coefficient `0.002494`, |coef| `0.002494`
- `lag_05__CT_place_HUT`: coefficient `-0.002433`, |coef| `0.002433`
- `lag_04__CT_place_HUT`: coefficient `0.002148`, |coef| `0.002148`
- `lag_14__CT_place_LOCKERROOM`: coefficient `-0.002110`, |coef| `0.002110`
- `lag_00__T_shots_fired_sum`: coefficient `-0.002025`, |coef| `0.002025`
- `lag_00__T_kills_last_3s`: coefficient `-0.001916`, |coef| `0.001916`
- `lag_00__kill_diff_last_3s`: coefficient `0.001912`, |coef| `0.001912`
- `lag_00__CT_place_LOBBY`: coefficient `0.001787`, |coef| `0.001787`
- `lag_03__CT_place_VENTS`: coefficient `-0.001745`, |coef| `0.001745`
- `lag_08__CT_place_LOBBY`: coefficient `-0.001673`, |coef| `0.001673`
- `lag_00__T_place_HUT`: coefficient `-0.001648`, |coef| `0.001648`
- `lag_05__CT_place_LOBBY`: coefficient `0.001595`, |coef| `0.001595`
- `lag_00__T_damage_last_5s`: coefficient `-0.001568`, |coef| `0.001568`
- `lag_00__CT4__shots_fired`: coefficient `-0.001537`, |coef| `0.001537`
- `lag_14__CT_place_HELL`: coefficient `0.001533`, |coef| `0.001533`

## Top 10 utility ridge features

- `lag_14__T_A_site_active_infernos`: coefficient `-0.001112` (lowers CT win probability)
- `lag_14__T_B_site_active_infernos`: coefficient `-0.001052` (lowers CT win probability)
- `lag_10__T_A_site_active_infernos`: coefficient `-0.000848` (lowers CT win probability)
- `lag_00__T_A_site_active_infernos`: coefficient `0.000844` (raises CT win probability)
- `lag_15__T_A_site_active_infernos`: coefficient `-0.000838` (lowers CT win probability)
- `lag_00__T_B_site_active_infernos`: coefficient `0.000802` (raises CT win probability)
- `lag_10__T_B_site_active_infernos`: coefficient `-0.000800` (lowers CT win probability)
- `lag_15__T_B_site_active_infernos`: coefficient `-0.000791` (lowers CT win probability)
- `lag_14__T_active_infernos`: coefficient `-0.000752` (lowers CT win probability)
- `lag_01__T_A_site_active_infernos`: coefficient `0.000720` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_13__CT_place_LOCKERROOM`: coefficient `0.002494` (raises CT win probability)
- `lag_05__CT_place_HUT`: coefficient `-0.002433` (lowers CT win probability)
- `lag_04__CT_place_HUT`: coefficient `0.002148` (raises CT win probability)
- `lag_14__CT_place_LOCKERROOM`: coefficient `-0.002110` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.002025` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001916` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001912` (raises CT win probability)
- `lag_00__CT_place_LOBBY`: coefficient `0.001787` (raises CT win probability)
- `lag_03__CT_place_VENTS`: coefficient `-0.001745` (lowers CT win probability)
- `lag_08__CT_place_LOBBY`: coefficient `-0.001673` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `134072`, seconds `83.00`, LSTM delta `-0.2633`

Top all feature movements:
- `lag_00__T_place_HUT`: contribution `-0.015365`
- `lag_03__CT_place_VENTS`: contribution `-0.014645`
- `lag_08__CT_place_LOBBY`: contribution `-0.013692`
- `lag_08__CT_place_HUT`: contribution `-0.012865`
- `lag_00__T_shots_fired_sum`: contribution `-0.012143`

Top utility-only movements:
- `lag_14__T_A_site_active_infernos`: contribution `-0.003310`

### tick `133944`, seconds `81.00`, LSTM delta `-0.2590`

Top all feature movements:
- `lag_13__CT_place_LOCKERROOM`: contribution `-0.031048`
- `lag_04__CT_place_HUT`: contribution `-0.020945`
- `lag_00__CT_place_LOBBY`: contribution `-0.014625`
- `lag_10__CT_place_HEAVEN`: contribution `-0.007867`
- `lag_00__T_kills_last_3s`: contribution `-0.006070`

Top utility-only movements:
- `lag_10__T_A_site_active_infernos`: contribution `-0.002525`

### tick `133976`, seconds `81.50`, LSTM delta `+0.2407`

Top all feature movements:
- `lag_14__CT_place_LOCKERROOM`: contribution `+0.026266`
- `lag_05__CT_place_HUT`: contribution `+0.023724`
- `lag_05__CT_place_LOBBY`: contribution `+0.013058`
- `lag_00__CT_place_VENTS`: contribution `+0.012687`
- `lag_01__CT_place_LOBBY`: contribution `+0.012494`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `133912`, seconds `80.50`, LSTM delta `-0.1217`

Top all feature movements:
- `lag_12__CT_place_LOCKERROOM`: contribution `-0.015484`
- `lag_00__T_shots_fired_sum`: contribution `-0.007589`
- `lag_00__T_kills_last_3s`: contribution `-0.006070`
- `lag_00__kill_diff_last_3s`: contribution `-0.004601`
- `lag_03__CT_place_LOBBY`: contribution `-0.004355`

Top utility-only movements:
- `lag_14__T_A_site_active_infernos`: contribution `-0.003310`
- `lag_14__T_B_site_active_infernos`: contribution `-0.002975`
- `lag_00__T_A_site_active_infernos`: contribution `-0.002513`
- `lag_00__T_B_site_active_infernos`: contribution `-0.002268`

### tick `134040`, seconds `82.50`, LSTM delta `-0.0601`

Top all feature movements:
- `lag_07__CT_place_LOBBY`: contribution `-0.005115`
- `lag_02__CT_place_VENTS`: contribution `-0.004918`
- `lag_03__CT_place_LOBBY`: contribution `+0.004355`
- `lag_07__CT_place_HUT`: contribution `-0.003079`
- `lag_11__T_place_LOBBY`: contribution `-0.002996`

Top utility-only movements:
- No utility movement among the top local contributors.
