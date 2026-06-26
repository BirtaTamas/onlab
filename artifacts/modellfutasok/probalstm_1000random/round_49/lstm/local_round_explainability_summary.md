# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-legacy-vs-gentle-mates-bo3-EYv8hp-oY0glsojznK6Qby/legacy-vs-gentle-mates-m2-mirage.csv`
- round_num: `11`

## Largest probability jumps

- tick `76571`, seconds `94.50`, LSTM `0.8494`, delta `+0.2024`
- tick `76507`, seconds `93.50`, LSTM `0.6534`, delta `-0.1745`
- tick `72571`, seconds `32.00`, LSTM `0.6465`, delta `+0.0953`
- tick `75483`, seconds `77.50`, LSTM `0.7950`, delta `+0.0755`
- tick `77051`, seconds `102.00`, LSTM `0.9339`, delta `+0.0543`
- tick `75195`, seconds `73.00`, LSTM `0.7262`, delta `+0.0392`
- tick `75003`, seconds `70.00`, LSTM `0.7156`, delta `+0.0346`
- tick `77115`, seconds `103.00`, LSTM `0.9735`, delta `+0.0343`
- tick `75963`, seconds `85.00`, LSTM `0.8216`, delta `+0.0336`
- tick `77019`, seconds `101.50`, LSTM `0.8795`, delta `+0.0314`

## Top 15 local ridge features

- `lag_00__CT_place_STAIRS`: coefficient `0.002967`, |coef| `0.002967`
- `lag_13__T_place_SNIPERSNEST`: coefficient `0.002311`, |coef| `0.002311`
- `lag_00__kill_diff_last_3s`: coefficient `0.002024`, |coef| `0.002024`
- `lag_00__T_place_JUNGLE`: coefficient `0.001973`, |coef| `0.001973`
- `lag_15__T_place_SNIPERSNEST`: coefficient `-0.001894`, |coef| `0.001894`
- `lag_02__CT_place_STAIRS`: coefficient `-0.001690`, |coef| `0.001690`
- `lag_00__CT_kills_last_3s`: coefficient `0.001555`, |coef| `0.001555`
- `lag_11__T5__duck_amount`: coefficient `0.001545`, |coef| `0.001545`
- `lag_00__damage_diff_last_5s`: coefficient `0.001503`, |coef| `0.001503`
- `lag_09__T5__duck_amount`: coefficient `-0.001488`, |coef| `0.001488`
- `lag_07__T4__is_walking`: coefficient `-0.001395`, |coef| `0.001395`
- `lag_10__CT3__duck_amount`: coefficient `0.001322`, |coef| `0.001322`
- `lag_15__T_place_JUNGLE`: coefficient `0.001283`, |coef| `0.001283`
- `lag_02__T_place_CONNECTOR`: coefficient `0.001236`, |coef| `0.001236`
- `lag_10__T_place_SNIPERSNEST`: coefficient `0.001229`, |coef| `0.001229`

## Top 10 utility ridge features

- `lag_00__CT_utility_damage_last_5s`: coefficient `0.000872` (raises CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000746` (raises CT win probability)
- `lag_00__T_B_site_active_infernos`: coefficient `-0.000638` (lowers CT win probability)
- `lag_00__T1__molly`: coefficient `-0.000634` (lowers CT win probability)
- `lag_00__T1__smoke`: coefficient `-0.000618` (lowers CT win probability)
- `lag_04__T4__molly`: coefficient `0.000544` (raises CT win probability)
- `lag_06__T4__molly`: coefficient `-0.000543` (lowers CT win probability)
- `lag_00__T1__utility_total`: coefficient `-0.000533` (lowers CT win probability)
- `lag_15__CT_utility_damage_last_5s`: coefficient `0.000532` (raises CT win probability)
- `lag_02__T_B_site_active_infernos`: coefficient `0.000516` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_STAIRS`: coefficient `0.002967` (raises CT win probability)
- `lag_13__T_place_SNIPERSNEST`: coefficient `0.002311` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002024` (raises CT win probability)
- `lag_00__T_place_JUNGLE`: coefficient `0.001973` (raises CT win probability)
- `lag_15__T_place_SNIPERSNEST`: coefficient `-0.001894` (lowers CT win probability)
- `lag_02__CT_place_STAIRS`: coefficient `-0.001690` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001555` (raises CT win probability)
- `lag_11__T5__duck_amount`: coefficient `0.001545` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001503` (raises CT win probability)
- `lag_09__T5__duck_amount`: coefficient `-0.001488` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `76571`, seconds `94.50`, LSTM delta `+0.2024`

Top all feature movements:
- `lag_15__T_place_SNIPERSNEST`: contribution `+0.033648`
- `lag_00__T_place_JUNGLE`: contribution `+0.025556`
- `lag_02__CT_place_STAIRS`: contribution `+0.013156`
- `lag_02__T_place_CONNECTOR`: contribution `+0.005985`
- `lag_11__T5__duck_amount`: contribution `+0.005868`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `76507`, seconds `93.50`, LSTM delta `-0.1745`

Top all feature movements:
- `lag_13__T_place_SNIPERSNEST`: contribution `-0.041070`
- `lag_00__CT_place_STAIRS`: contribution `-0.023095`
- `lag_09__T5__duck_amount`: contribution `-0.005649`
- `lag_00__T_place_CONNECTOR`: contribution `-0.005340`
- `lag_00__kill_diff_last_3s`: contribution `-0.004871`

Top utility-only movements:
- `lag_00__T_B_site_active_infernos`: contribution `-0.001805`

### tick `72571`, seconds `32.00`, LSTM delta `+0.0953`

Top all feature movements:
- `lag_10__CT3__duck_amount`: contribution `+0.004919`
- `lag_00__kill_diff_last_3s`: contribution `+0.004871`
- `lag_00__CT_kills_last_3s`: contribution `+0.004490`
- `lag_00__CT1__is_scoped`: contribution `+0.004363`
- `lag_15__CT5__duck_amount`: contribution `+0.003991`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `75483`, seconds `77.50`, LSTM delta `+0.0755`

Top all feature movements:
- `lag_00__CT_place_STAIRS`: contribution `+0.023095`
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.007586`
- `lag_09__T5__duck_amount`: contribution `+0.005649`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.005318`
- `lag_07__T4__is_walking`: contribution `+0.003220`

Top utility-only movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.007586`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.005318`

### tick `77051`, seconds `102.00`, LSTM delta `+0.0543`

Top all feature movements:
- `lag_15__T_place_JUNGLE`: contribution `+0.016614`
- `lag_13__T_place_JUNGLE`: contribution `+0.007433`
- `lag_00__kill_diff_last_3s`: contribution `+0.004871`
- `lag_00__CT_kills_last_3s`: contribution `+0.004490`
- `lag_15__T_place_CTSPAWN`: contribution `-0.004135`

Top utility-only movements:
- No utility movement among the top local contributors.
