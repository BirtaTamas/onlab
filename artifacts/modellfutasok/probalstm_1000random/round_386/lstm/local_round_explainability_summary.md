# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-furia-vs-saw-bo3-hxORpk_jCtMpGRLo1Voi3p/furia-vs-saw-m2-dust2.csv`
- round_num: `14`

## Largest probability jumps

- tick `117885`, seconds `0.50`, LSTM `0.0174`, delta `-0.0313`
- tick `119645`, seconds `28.00`, LSTM `0.0137`, delta `-0.0183`
- tick `119165`, seconds `20.50`, LSTM `0.0327`, delta `+0.0091`
- tick `119101`, seconds `19.50`, LSTM `0.0254`, delta `-0.0075`
- tick `119613`, seconds `27.50`, LSTM `0.0319`, delta `-0.0057`
- tick `118141`, seconds `4.50`, LSTM `0.0240`, delta `+0.0051`
- tick `119005`, seconds `18.00`, LSTM `0.0302`, delta `-0.0051`
- tick `118653`, seconds `12.50`, LSTM `0.0244`, delta `+0.0049`
- tick `117917`, seconds `1.00`, LSTM `0.0129`, delta `-0.0045`
- tick `119293`, seconds `22.50`, LSTM `0.0307`, delta `+0.0040`

## Top 15 local ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000329`, |coef| `0.000329`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000273`, |coef| `0.000273`
- `lag_00__T_velocity_mean`: coefficient `-0.000254`, |coef| `0.000254`
- `lag_00__CT_velocity_mean`: coefficient `-0.000227`, |coef| `0.000227`
- `lag_01__armor_diff`: coefficient `0.000198`, |coef| `0.000198`
- `lag_01__smoke_inv_diff`: coefficient `0.000195`, |coef| `0.000195`
- `lag_01__utility_inv_diff`: coefficient `0.000190`, |coef| `0.000190`
- `lag_01__T5__has_bomb`: coefficient `-0.000176`, |coef| `0.000176`
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000164`, |coef| `0.000164`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000164`, |coef| `0.000164`
- `lag_01__CT_armor_sum`: coefficient `0.000161`, |coef| `0.000161`
- `lag_00__CT_place_SHORTSTAIRS`: coefficient `0.000160`, |coef| `0.000160`
- `lag_01__centroid_distance_xy`: coefficient `-0.000151`, |coef| `0.000151`
- `lag_11__T_place_LOWERTUNNEL`: coefficient `0.000145`, |coef| `0.000145`
- `lag_01__T5__utility_total`: coefficient `-0.000144`, |coef| `0.000144`

## Top 10 utility ridge features

- `lag_01__smoke_inv_diff`: coefficient `0.000195` (raises CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000190` (raises CT win probability)
- `lag_01__T5__utility_total`: coefficient `-0.000144` (lowers CT win probability)
- `lag_01__flash_inv_diff`: coefficient `0.000138` (raises CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000137` (raises CT win probability)
- `lag_01__T5__flash`: coefficient `-0.000134` (lowers CT win probability)
- `lag_01__T_smoke_inv`: coefficient `-0.000132` (lowers CT win probability)
- `lag_01__T_utility_inv`: coefficient `-0.000114` (lowers CT win probability)
- `lag_01__T5__molly`: coefficient `-0.000110` (lowers CT win probability)
- `lag_01__T2__molly`: coefficient `-0.000107` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000329` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000273` (lowers CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000254` (lowers CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000227` (lowers CT win probability)
- `lag_01__armor_diff`: coefficient `0.000198` (raises CT win probability)
- `lag_01__T5__has_bomb`: coefficient `-0.000176` (lowers CT win probability)
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000164` (lowers CT win probability)
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000164` (lowers CT win probability)
- `lag_01__CT_armor_sum`: coefficient `0.000161` (raises CT win probability)
- `lag_00__CT_place_SHORTSTAIRS`: coefficient `0.000160` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `117885`, seconds `0.50`, LSTM delta `-0.0313`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.001575`
- `lag_01__T_place_TSPAWN`: contribution `-0.001211`
- `lag_00__T_velocity_mean`: contribution `-0.000933`
- `lag_00__CT_velocity_mean`: contribution `-0.000799`
- `lag_01__smoke_inv_diff`: contribution `-0.000622`

Top utility-only movements:
- `lag_01__smoke_inv_diff`: contribution `-0.000622`
- `lag_01__utility_inv_diff`: contribution `-0.000584`
- `lag_01__T5__utility_total`: contribution `-0.000334`
- `lag_01__flash_inv_diff`: contribution `-0.000312`
- `lag_01__T_smoke_inv`: contribution `-0.000301`

### tick `119645`, seconds `28.00`, LSTM delta `-0.0183`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.001372`
- `lag_11__T_place_TUNNELSTAIRS`: contribution `-0.000969`
- `lag_00__CT_place_SHORTSTAIRS`: contribution `-0.000894`
- `lag_11__T_place_LOWERTUNNEL`: contribution `-0.000628`
- `lag_13__CT_place_BDOORS`: contribution `-0.000441`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `119165`, seconds `20.50`, LSTM delta `+0.0091`

Top all feature movements:
- `lag_05__CT_place_HOLE`: contribution `+0.001234`
- `lag_00__T_shots_fired_sum`: contribution `+0.000882`
- `lag_10__T_place_TUNNELSTAIRS`: contribution `+0.000842`
- `lag_03__T_place_TUNNELSTAIRS`: contribution `+0.000543`
- `lag_15__T_place_TUNNELSTAIRS`: contribution `+0.000453`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `119101`, seconds `19.50`, LSTM delta `-0.0075`

Top all feature movements:
- `lag_03__CT_place_HOLE`: contribution `-0.000873`
- `lag_02__T_place_TUNNELSTAIRS`: contribution `-0.000485`
- `lag_08__CT_place_HOLE`: contribution `-0.000484`
- `lag_00__T_shots_fired_sum`: contribution `-0.000392`
- `lag_03__CT5__duck_amount`: contribution `+0.000383`

Top utility-only movements:
- `lag_12__T1__flash_duration`: contribution `-0.000151`

### tick `119613`, seconds `27.50`, LSTM delta `-0.0057`

Top all feature movements:
- `lag_10__T_place_TUNNELSTAIRS`: contribution `-0.000842`
- `lag_00__T_shots_fired_sum`: contribution `-0.000490`
- `lag_01__T1__shots_fired`: contribution `-0.000407`
- `lag_03__CT5__duck_amount`: contribution `-0.000383`
- `lag_10__T_place_LOWERTUNNEL`: contribution `-0.000354`

Top utility-only movements:
- No utility movement among the top local contributors.
