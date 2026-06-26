# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-b8-vs-flyquest-bo3-ROTxQXIIApwC88KHLMMjQT/b8-vs-flyquest-m3-inferno.csv`
- round_num: `8`

## Largest probability jumps

- tick `54606`, seconds `26.00`, LSTM `0.0656`, delta `-0.1313`
- tick `52974`, seconds `0.50`, LSTM `0.0782`, delta `-0.0567`
- tick `54126`, seconds `18.50`, LSTM `0.1493`, delta `+0.0348`
- tick `53838`, seconds `14.00`, LSTM `0.1783`, delta `-0.0323`
- tick `54862`, seconds `30.00`, LSTM `0.0248`, delta `-0.0315`
- tick `54062`, seconds `17.50`, LSTM `0.1300`, delta `-0.0308`
- tick `53774`, seconds `13.00`, LSTM `0.2073`, delta `+0.0249`
- tick `53646`, seconds `11.00`, LSTM `0.1655`, delta `+0.0243`
- tick `54414`, seconds `23.00`, LSTM `0.2008`, delta `+0.0222`
- tick `54510`, seconds `24.50`, LSTM `0.1859`, delta `-0.0213`

## Top 15 local ridge features

- `lag_00__CT_place_BALCONY`: coefficient `0.001382`, |coef| `0.001382`
- `lag_13__CT_place_ARCH`: coefficient `0.000899`, |coef| `0.000899`
- `lag_15__T_shots_fired_sum`: coefficient `0.000879`, |coef| `0.000879`
- `lag_00__T3__is_scoped`: coefficient `0.000875`, |coef| `0.000875`
- `lag_00__CT_he_last_5s`: coefficient `-0.000849`, |coef| `0.000849`
- `lag_00__T_kills_last_3s`: coefficient `-0.000848`, |coef| `0.000848`
- `lag_10__CT_place_RUINS`: coefficient `0.000777`, |coef| `0.000777`
- `lag_14__T_place_BACKALLEY`: coefficient `0.000762`, |coef| `0.000762`
- `lag_09__T_place_TOPOFMID`: coefficient `-0.000680`, |coef| `0.000680`
- `lag_00__T_damage_last_5s`: coefficient `-0.000674`, |coef| `0.000674`
- `lag_00__kill_diff_last_3s`: coefficient `0.000653`, |coef| `0.000653`
- `lag_15__T4__shots_fired`: coefficient `0.000642`, |coef| `0.000642`
- `lag_03__CT_place_RUINS`: coefficient `0.000631`, |coef| `0.000631`
- `lag_00__CT3__alive`: coefficient `0.000628`, |coef| `0.000628`
- `lag_00__CT3__hp`: coefficient `0.000620`, |coef| `0.000620`

## Top 10 utility ridge features

- `lag_00__CT_he_last_5s`: coefficient `-0.000849` (lowers CT win probability)
- `lag_12__CT_he_last_5s`: coefficient `0.000607` (raises CT win probability)
- `lag_00__CT3__smoke`: coefficient `0.000574` (raises CT win probability)
- `lag_13__CT_he_last_5s`: coefficient `0.000392` (raises CT win probability)
- `lag_09__CT_he_last_5s`: coefficient `0.000368` (raises CT win probability)
- `lag_15__CT_he_last_5s`: coefficient `-0.000358` (lowers CT win probability)
- `lag_07__CT_he_last_5s`: coefficient `0.000266` (raises CT win probability)
- `lag_06__CT_he_last_5s`: coefficient `0.000260` (raises CT win probability)
- `lag_00__CT3__utility_total`: coefficient `0.000247` (raises CT win probability)
- `lag_01__CT_he_last_5s`: coefficient `-0.000238` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_BALCONY`: coefficient `0.001382` (raises CT win probability)
- `lag_13__CT_place_ARCH`: coefficient `0.000899` (raises CT win probability)
- `lag_15__T_shots_fired_sum`: coefficient `0.000879` (raises CT win probability)
- `lag_00__T3__is_scoped`: coefficient `0.000875` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.000848` (lowers CT win probability)
- `lag_10__CT_place_RUINS`: coefficient `0.000777` (raises CT win probability)
- `lag_14__T_place_BACKALLEY`: coefficient `0.000762` (raises CT win probability)
- `lag_09__T_place_TOPOFMID`: coefficient `-0.000680` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.000674` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000653` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `54606`, seconds `26.00`, LSTM delta `-0.1313`

Top all feature movements:
- `lag_00__CT_place_BALCONY`: contribution `-0.008866`
- `lag_15__T_shots_fired_sum`: contribution `-0.005931`
- `lag_00__T3__is_scoped`: contribution `-0.005615`
- `lag_13__CT_place_ARCH`: contribution `-0.003667`
- `lag_15__T4__shots_fired`: contribution `-0.003571`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `52974`, seconds `0.50`, LSTM delta `-0.0567`

Top all feature movements:
- `lag_00__CT_he_last_5s`: contribution `-0.015587`
- `lag_01__CT_place_CTSPAWN`: contribution `-0.001693`
- `lag_01__T_closest_enemy_dist`: contribution `-0.001335`
- `lag_01__centroid_distance_xy`: contribution `-0.001263`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.001235`

Top utility-only movements:
- `lag_00__CT_he_last_5s`: contribution `-0.015587`
- `lag_01__T_smoke_inv`: contribution `-0.000476`
- `lag_01__T_molly_inv`: contribution `-0.000469`
- `lag_01__molly_inv_diff`: contribution `-0.000448`
- `lag_01__utility_inv_diff`: contribution `-0.000441`

### tick `54126`, seconds `18.50`, LSTM delta `+0.0348`

Top all feature movements:
- `lag_01__T3__is_scoped`: contribution `+0.003655`
- `lag_00__T_shots_fired_sum`: contribution `+0.002587`
- `lag_13__CT_place_BALCONY`: contribution `+0.002351`
- `lag_02__CT_place_RUINS`: contribution `+0.001752`
- `lag_00__T4__shots_fired`: contribution `+0.001699`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `53838`, seconds `14.00`, LSTM delta `-0.0323`

Top all feature movements:
- `lag_15__CT_he_last_5s`: contribution `-0.006562`
- `lag_00__T3__is_scoped`: contribution `-0.005615`
- `lag_10__CT_place_RUINS`: contribution `-0.002714`
- `lag_05__CT_he_last_5s`: contribution `-0.002227`
- `lag_06__CT1__is_walking`: contribution `-0.001394`

Top utility-only movements:
- `lag_15__CT_he_last_5s`: contribution `-0.006562`
- `lag_05__CT_he_last_5s`: contribution `-0.002227`

### tick `54862`, seconds `30.00`, LSTM delta `-0.0315`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.002687`
- `lag_08__CT_place_BALCONY`: contribution `-0.002466`
- `lag_11__CT_place_RUINS`: contribution `-0.002092`
- `lag_06__CT_place_LIBRARY`: contribution `-0.001845`
- `lag_00__T_damage_last_5s`: contribution `-0.001615`

Top utility-only movements:
- No utility movement among the top local contributors.
