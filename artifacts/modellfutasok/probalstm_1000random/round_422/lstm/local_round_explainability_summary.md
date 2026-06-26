# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-spirit-vs-vitality-bo3-KtWhzrlsNkWCCS0U9BIlr3/spirit-vs-vitality-m2-nuke.csv`
- round_num: `23`

## Largest probability jumps

- tick `206589`, seconds `33.50`, LSTM `0.1397`, delta `-0.0629`
- tick `204861`, seconds `6.50`, LSTM `0.1195`, delta `-0.0529`
- tick `204477`, seconds `0.50`, LSTM `0.0538`, delta `-0.0528`
- tick `205597`, seconds `18.00`, LSTM `0.2467`, delta `+0.0508`
- tick `205629`, seconds `18.50`, LSTM `0.2901`, delta `+0.0434`
- tick `204541`, seconds `1.50`, LSTM `0.0862`, delta `+0.0405`
- tick `205309`, seconds `13.50`, LSTM `0.1063`, delta `-0.0391`
- tick `205213`, seconds `12.00`, LSTM `0.1187`, delta `+0.0387`
- tick `208477`, seconds `63.00`, LSTM `0.0693`, delta `-0.0369`
- tick `204605`, seconds `2.50`, LSTM `0.1427`, delta `+0.0314`

## Top 15 local ridge features

- `lag_00__T_place_ROOF`: coefficient `-0.001714`, |coef| `0.001714`
- `lag_01__T_place_ROOF`: coefficient `-0.001140`, |coef| `0.001140`
- `lag_03__CT_place_SECRET`: coefficient `-0.001030`, |coef| `0.001030`
- `lag_07__CT_place_SECRET`: coefficient `-0.000881`, |coef| `0.000881`
- `lag_01__CT_place_SECRET`: coefficient `-0.000874`, |coef| `0.000874`
- `lag_00__CT_place_HEAVEN`: coefficient `0.000845`, |coef| `0.000845`
- `lag_02__CT_place_SECRET`: coefficient `-0.000837`, |coef| `0.000837`
- `lag_00__CT_flashes_last_5s`: coefficient `-0.000817`, |coef| `0.000817`
- `lag_03__CT_place_VENTS`: coefficient `0.000795`, |coef| `0.000795`
- `lag_10__CT_place_VENTS`: coefficient `0.000791`, |coef| `0.000791`
- `lag_04__CT_place_SECRET`: coefficient `-0.000783`, |coef| `0.000783`
- `lag_00__CT_place_SECRET`: coefficient `-0.000776`, |coef| `0.000776`
- `lag_10__CT_place_RAFTERS`: coefficient `0.000751`, |coef| `0.000751`
- `lag_08__CT_place_RAFTERS`: coefficient `0.000707`, |coef| `0.000707`
- `lag_00__CT_B_site_active_smokes`: coefficient `0.000701`, |coef| `0.000701`

## Top 10 utility ridge features

- `lag_00__CT_flashes_last_5s`: coefficient `-0.000817` (lowers CT win probability)
- `lag_00__CT_B_site_active_smokes`: coefficient `0.000701` (raises CT win probability)
- `lag_00__CT_A_site_active_smokes`: coefficient `0.000680` (raises CT win probability)
- `lag_00__CT_he_last_5s`: coefficient `0.000647` (raises CT win probability)
- `lag_00__CT_smokes_last_5s`: coefficient `0.000610` (raises CT win probability)
- `lag_13__CT_B_site_active_smokes`: coefficient `0.000538` (raises CT win probability)
- `lag_01__CT4__smoke`: coefficient `0.000529` (raises CT win probability)
- `lag_13__CT_A_site_active_smokes`: coefficient `0.000521` (raises CT win probability)
- `lag_04__CT4__smoke`: coefficient `0.000520` (raises CT win probability)
- `lag_00__CT4__smoke`: coefficient `0.000505` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_ROOF`: coefficient `-0.001714` (lowers CT win probability)
- `lag_01__T_place_ROOF`: coefficient `-0.001140` (lowers CT win probability)
- `lag_03__CT_place_SECRET`: coefficient `-0.001030` (lowers CT win probability)
- `lag_07__CT_place_SECRET`: coefficient `-0.000881` (lowers CT win probability)
- `lag_01__CT_place_SECRET`: coefficient `-0.000874` (lowers CT win probability)
- `lag_00__CT_place_HEAVEN`: coefficient `0.000845` (raises CT win probability)
- `lag_02__CT_place_SECRET`: coefficient `-0.000837` (lowers CT win probability)
- `lag_03__CT_place_VENTS`: coefficient `0.000795` (raises CT win probability)
- `lag_10__CT_place_VENTS`: coefficient `0.000791` (raises CT win probability)
- `lag_04__CT_place_SECRET`: coefficient `-0.000783` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `206589`, seconds `33.50`, LSTM delta `-0.0629`

Top all feature movements:
- `lag_00__T_place_ROOF`: contribution `-0.009706`
- `lag_07__CT_place_SECRET`: contribution `-0.009070`
- `lag_13__CT_place_HEAVEN`: contribution `-0.003712`
- `lag_01__CT1__duck_amount`: contribution `-0.002580`
- `lag_15__CT_place_RAFTERS`: contribution `-0.001725`

Top utility-only movements:
- `lag_00__CT_B_site_active_smokes`: contribution `-0.001165`
- `lag_04__CT4__smoke`: contribution `-0.001134`
- `lag_00__CT_A_site_active_smokes`: contribution `-0.001094`

### tick `204861`, seconds `6.50`, LSTM delta `-0.0529`

Top all feature movements:
- `lag_00__CT_he_last_5s`: contribution `-0.011871`
- `lag_00__CT_smokes_last_5s`: contribution `-0.010536`
- `lag_10__CT_he_last_5s`: contribution `-0.006498`
- `lag_10__CT_smokes_last_5s`: contribution `-0.005767`
- `lag_00__CT_place_HEAVEN`: contribution `+0.004565`

Top utility-only movements:
- `lag_00__CT_he_last_5s`: contribution `-0.011871`
- `lag_00__CT_smokes_last_5s`: contribution `-0.010536`
- `lag_10__CT_he_last_5s`: contribution `-0.006498`
- `lag_10__CT_smokes_last_5s`: contribution `-0.005767`
- `lag_02__CT_flashes_last_5s`: contribution `-0.004280`

### tick `204477`, seconds `0.50`, LSTM delta `-0.0528`

Top all feature movements:
- `lag_00__CT_flashes_last_5s`: contribution `-0.008982`
- `lag_01__CT_place_CTSPAWN`: contribution `-0.002239`
- `lag_01__T_place_TSPAWN`: contribution `-0.002087`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.001809`
- `lag_01__T_closest_enemy_dist`: contribution `-0.001573`

Top utility-only movements:
- `lag_00__CT_flashes_last_5s`: contribution `-0.008982`
- `lag_01__T1__flash`: contribution `-0.000614`
- `lag_01__smoke_inv_diff`: contribution `-0.000603`
- `lag_01__T5__utility_total`: contribution `-0.000596`
- `lag_01__utility_inv_diff`: contribution `-0.000561`

### tick `205597`, seconds `18.00`, LSTM delta `+0.0508`

Top all feature movements:
- `lag_00__T_place_ROOF`: contribution `+0.009706`
- `lag_02__CT_place_VENTS`: contribution `+0.004909`
- `lag_10__CT_place_RAFTERS`: contribution `+0.004014`
- `lag_15__CT_place_RAFTERS`: contribution `-0.001725`
- `lag_10__CT_place_HEAVEN`: contribution `+0.001703`

Top utility-only movements:
- `lag_15__T_A_site_active_infernos`: contribution `+0.001063`
- `lag_15__T_B_site_active_infernos`: contribution `+0.000957`
- `lag_13__CT_B_site_active_smokes`: contribution `+0.000893`
- `lag_01__T_A_site_active_infernos`: contribution `+0.000879`
- `lag_13__CT_A_site_active_smokes`: contribution `+0.000838`

### tick `205629`, seconds `18.50`, LSTM delta `+0.0434`

Top all feature movements:
- `lag_03__CT_place_VENTS`: contribution `+0.006667`
- `lag_01__T_place_ROOF`: contribution `+0.006457`
- `lag_11__CT_place_RAFTERS`: contribution `+0.003674`
- `lag_11__CT1__is_walking`: contribution `+0.001435`
- `lag_06__CT3__is_walking`: contribution `+0.000946`

Top utility-only movements:
- `lag_13__CT_B_site_active_smokes`: contribution `+0.000893`
- `lag_02__T_A_site_active_infernos`: contribution `+0.000877`
- `lag_13__CT_A_site_active_smokes`: contribution `+0.000838`
- `lag_02__T_B_site_active_infernos`: contribution `+0.000790`
- `lag_13__T5__flash`: contribution `+0.000765`
