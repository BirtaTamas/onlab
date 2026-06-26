# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-falcons-bo3-yayytstbo8IxTFlUpfbUPR/mouz-vs-falcons-m1-train.csv`
- round_num: `3`

## Largest probability jumps

- tick `15199`, seconds `73.50`, LSTM `0.0324`, delta `-0.0538`
- tick `10527`, seconds `0.50`, LSTM `0.0210`, delta `-0.0334`
- tick `14719`, seconds `66.00`, LSTM `0.1016`, delta `+0.0322`
- tick `13535`, seconds `47.50`, LSTM `0.0300`, delta `-0.0311`
- tick `14687`, seconds `65.50`, LSTM `0.0694`, delta `+0.0257`
- tick `15007`, seconds `70.50`, LSTM `0.0937`, delta `-0.0218`
- tick `14783`, seconds `67.00`, LSTM `0.1163`, delta `+0.0216`
- tick `12063`, seconds `24.50`, LSTM `0.0545`, delta `-0.0145`
- tick `11583`, seconds `17.00`, LSTM `0.0638`, delta `+0.0137`
- tick `14879`, seconds `68.50`, LSTM `0.1161`, delta `-0.0127`

## Top 15 local ridge features

- `lag_02__CT_place_LONGDOG`: coefficient `-0.000586`, |coef| `0.000586`
- `lag_00__damage_diff_last_5s`: coefficient `0.000516`, |coef| `0.000516`
- `lag_00__kill_diff_last_3s`: coefficient `0.000483`, |coef| `0.000483`
- `lag_13__CT5__duck_amount`: coefficient `0.000469`, |coef| `0.000469`
- `lag_01__CT_place_LONGDOG`: coefficient `-0.000453`, |coef| `0.000453`
- `lag_04__CT4__duck_amount`: coefficient `0.000450`, |coef| `0.000450`
- `lag_00__CT5__duck_amount`: coefficient `-0.000421`, |coef| `0.000421`
- `lag_10__bomb_events_last_5s`: coefficient `0.000415`, |coef| `0.000415`
- `lag_08__T_place_TMAIN`: coefficient `-0.000407`, |coef| `0.000407`
- `lag_04__T_place_IVY`: coefficient `0.000371`, |coef| `0.000371`
- `lag_00__CT_place_LONGDOG`: coefficient `0.000371`, |coef| `0.000371`
- `lag_03__T_place_IVY`: coefficient `0.000367`, |coef| `0.000367`
- `lag_00__T_kills_last_3s`: coefficient `-0.000362`, |coef| `0.000362`
- `lag_03__CT4__duck_amount`: coefficient `0.000359`, |coef| `0.000359`
- `lag_01__T1__duck_amount`: coefficient `-0.000357`, |coef| `0.000357`

## Top 10 utility ridge features

- `lag_02__CT_A_site_active_smokes`: coefficient `0.000235` (raises CT win probability)
- `lag_09__T1__molly`: coefficient `-0.000234` (lowers CT win probability)
- `lag_01__T5__molly`: coefficient `-0.000220` (lowers CT win probability)
- `lag_01__T5__smoke`: coefficient `-0.000213` (lowers CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000205` (raises CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000181` (raises CT win probability)
- `lag_02__CT_active_smokes`: coefficient `0.000178` (raises CT win probability)
- `lag_01__T_molly_inv`: coefficient `-0.000173` (lowers CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000173` (raises CT win probability)
- `lag_01__CT_A_site_active_smokes`: coefficient `0.000159` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_02__CT_place_LONGDOG`: coefficient `-0.000586` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000516` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000483` (raises CT win probability)
- `lag_13__CT5__duck_amount`: coefficient `0.000469` (raises CT win probability)
- `lag_01__CT_place_LONGDOG`: coefficient `-0.000453` (lowers CT win probability)
- `lag_04__CT4__duck_amount`: coefficient `0.000450` (raises CT win probability)
- `lag_00__CT5__duck_amount`: coefficient `-0.000421` (lowers CT win probability)
- `lag_10__bomb_events_last_5s`: coefficient `0.000415` (raises CT win probability)
- `lag_08__T_place_TMAIN`: coefficient `-0.000407` (lowers CT win probability)
- `lag_04__T_place_IVY`: coefficient `0.000371` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `15199`, seconds `73.50`, LSTM delta `-0.0538`

Top all feature movements:
- `lag_02__CT_place_LONGDOG`: contribution `-0.003826`
- `lag_01__CT_place_LONGDOG`: contribution `-0.002956`
- `lag_00__CT_place_LONGDOG`: contribution `-0.002419`
- `lag_13__CT5__duck_amount`: contribution `-0.001770`
- `lag_04__CT4__duck_amount`: contribution `-0.001653`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `10527`, seconds `0.50`, LSTM delta `-0.0334`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.001451`
- `lag_01__T_place_TSPAWN`: contribution `-0.001072`
- `lag_01__T_closest_enemy_dist`: contribution `-0.000920`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.000918`
- `lag_00__CT_velocity_mean`: contribution `-0.000865`

Top utility-only movements:
- `lag_01__molly_inv_diff`: contribution `-0.000571`
- `lag_01__utility_inv_diff`: contribution `-0.000477`
- `lag_01__smoke_inv_diff`: contribution `-0.000440`
- `lag_01__T_molly_inv`: contribution `-0.000392`
- `lag_01__T5__molly`: contribution `-0.000348`

### tick `14719`, seconds `66.00`, LSTM delta `+0.0322`

Top all feature movements:
- `lag_04__T_place_IVY`: contribution `+0.001984`
- `lag_00__CT5__duck_amount`: contribution `+0.001589`
- `lag_00__T3__duck_amount`: contribution `+0.001252`
- `lag_01__T_place_LONGDOG`: contribution `+0.001201`
- `lag_15__T_place_TMAIN`: contribution `+0.001113`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `13535`, seconds `47.50`, LSTM delta `-0.0311`

Top all feature movements:
- `lag_09__CT_place_ELECTRICALBOX`: contribution `-0.002839`
- `lag_15__CT_place_TMAIN`: contribution `-0.002031`
- `lag_00__CT_place_TMAIN`: contribution `-0.001854`
- `lag_00__T_shots_fired_sum`: contribution `-0.001523`
- `lag_01__T_place_LONGDOG`: contribution `-0.001201`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `14687`, seconds `65.50`, LSTM delta `+0.0257`

Top all feature movements:
- `lag_03__T_place_IVY`: contribution `+0.001963`
- `lag_13__CT5__duck_amount`: contribution `+0.001770`
- `lag_10__bomb_events_last_5s`: contribution `+0.001733`
- `lag_00__damage_diff_last_5s`: contribution `+0.001164`
- `lag_00__kill_diff_last_3s`: contribution `+0.001163`

Top utility-only movements:
- No utility movement among the top local contributors.
