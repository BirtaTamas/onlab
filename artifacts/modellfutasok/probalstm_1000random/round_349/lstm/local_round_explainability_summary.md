# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-natus-vincere-vs-tyloo-bo3-u9zlDGjnIy0eSohnO5P-Xx/natus-vincere-vs-tyloo-m3-ancient.csv`
- round_num: `12`

## Largest probability jumps

- tick `77006`, seconds `51.50`, LSTM `0.8353`, delta `+0.1529`
- tick `79342`, seconds `88.00`, LSTM `0.8220`, delta `+0.1371`
- tick `77102`, seconds `53.00`, LSTM `0.7030`, delta `-0.0949`
- tick `79310`, seconds `87.50`, LSTM `0.6848`, delta `+0.0792`
- tick `80110`, seconds `100.00`, LSTM `0.9358`, delta `+0.0724`
- tick `78606`, seconds `76.50`, LSTM `0.6787`, delta `+0.0404`
- tick `79022`, seconds `83.00`, LSTM `0.6377`, delta `-0.0380`
- tick `76974`, seconds `51.00`, LSTM `0.6824`, delta `+0.0328`
- tick `77038`, seconds `52.00`, LSTM `0.8085`, delta `-0.0268`
- tick `77166`, seconds `54.00`, LSTM `0.6598`, delta `-0.0246`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.002461`, |coef| `0.002461`
- `lag_00__kill_diff_last_3s`: coefficient `0.002428`, |coef| `0.002428`
- `lag_00__T_place_SIDEENTRANCE`: coefficient `-0.002302`, |coef| `0.002302`
- `lag_00__damage_diff_last_5s`: coefficient `0.002160`, |coef| `0.002160`
- `lag_01__CT_shots_fired_sum`: coefficient `0.001953`, |coef| `0.001953`
- `lag_01__damage_diff_last_5s`: coefficient `0.001923`, |coef| `0.001923`
- `lag_00__CT_damage_last_5s`: coefficient `0.001835`, |coef| `0.001835`
- `lag_00__T2__is_walking`: coefficient `-0.001767`, |coef| `0.001767`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001707`, |coef| `0.001707`
- `lag_01__CT_damage_last_5s`: coefficient `0.001621`, |coef| `0.001621`
- `lag_00__CT_place_SIDEENTRANCE`: coefficient `0.001446`, |coef| `0.001446`
- `lag_03__CT2__is_walking`: coefficient `0.001424`, |coef| `0.001424`
- `lag_02__CT4__duck_amount`: coefficient `-0.001408`, |coef| `0.001408`
- `lag_01__T_place_SIDEENTRANCE`: coefficient `-0.001372`, |coef| `0.001372`
- `lag_00__T2__alive`: coefficient `-0.001365`, |coef| `0.001365`

## Top 10 utility ridge features

- `lag_11__CT_B_site_active_smokes`: coefficient `-0.001060` (lowers CT win probability)
- `lag_10__CT_B_site_active_smokes`: coefficient `-0.001046` (lowers CT win probability)
- `lag_09__CT_B_site_active_smokes`: coefficient `-0.000775` (lowers CT win probability)
- `lag_10__CT_active_smokes`: coefficient `-0.000774` (lowers CT win probability)
- `lag_11__CT_active_smokes`: coefficient `-0.000769` (lowers CT win probability)
- `lag_12__CT_B_site_active_smokes`: coefficient `-0.000761` (lowers CT win probability)
- `lag_00__T4__utility_total`: coefficient `-0.000711` (lowers CT win probability)
- `lag_13__CT_B_site_active_smokes`: coefficient `-0.000696` (lowers CT win probability)
- `lag_00__T4__molly`: coefficient `-0.000664` (lowers CT win probability)
- `lag_00__T4__smoke`: coefficient `-0.000663` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.002461` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002428` (raises CT win probability)
- `lag_00__T_place_SIDEENTRANCE`: coefficient `-0.002302` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002160` (raises CT win probability)
- `lag_01__CT_shots_fired_sum`: coefficient `0.001953` (raises CT win probability)
- `lag_01__damage_diff_last_5s`: coefficient `0.001923` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001835` (raises CT win probability)
- `lag_00__T2__is_walking`: coefficient `-0.001767` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001707` (raises CT win probability)
- `lag_01__CT_damage_last_5s`: coefficient `0.001621` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `77006`, seconds `51.50`, LSTM delta `+0.1529`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.007107`
- `lag_01__CT_shots_fired_sum`: contribution `+0.006784`
- `lag_00__CT_shots_fired_sum`: contribution `+0.005929`
- `lag_00__kill_diff_last_3s`: contribution `+0.005843`
- `lag_02__CT3__duck_amount`: contribution `+0.005007`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `79342`, seconds `88.00`, LSTM delta `+0.1371`

Top all feature movements:
- `lag_00__T_place_SIDEENTRANCE`: contribution `+0.011234`
- `lag_00__CT_kills_last_3s`: contribution `+0.007107`
- `lag_00__kill_diff_last_3s`: contribution `+0.005843`
- `lag_08__CT4__duck_amount`: contribution `+0.004841`
- `lag_01__CT_shots_fired_sum`: contribution `+0.004070`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `77102`, seconds `53.00`, LSTM delta `-0.0949`

Top all feature movements:
- `lag_02__CT_shots_fired_sum`: contribution `-0.006130`
- `lag_00__kill_diff_last_3s`: contribution `-0.005843`
- `lag_00__CT_place_SIDEENTRANCE`: contribution `-0.005820`
- `lag_02__CT3__duck_amount`: contribution `-0.005007`
- `lag_08__CT_place_SIDEHALL`: contribution `-0.003434`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `79310`, seconds `87.50`, LSTM delta `+0.0792`

Top all feature movements:
- `lag_00__damage_diff_last_5s`: contribution `+0.004094`
- `lag_00__T2__is_walking`: contribution `+0.004059`
- `lag_01__CT4__duck_amount`: contribution `+0.003716`
- `lag_00__CT_shots_fired_sum`: contribution `+0.003558`
- `lag_00__CT_damage_last_5s`: contribution `+0.003359`

Top utility-only movements:
- `lag_10__CT_B_site_active_smokes`: contribution `+0.001738`

### tick `80110`, seconds `100.00`, LSTM delta `+0.0724`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.007107`
- `lag_00__kill_diff_last_3s`: contribution `+0.005843`
- `lag_06__T_place_SIDEHALL`: contribution `+0.005783`
- `lag_05__T_place_SIDEHALL`: contribution `+0.005211`
- `lag_00__T_place_SIDEHALL`: contribution `+0.004934`

Top utility-only movements:
- No utility movement among the top local contributors.
