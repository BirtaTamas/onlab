# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-aurora-vs-faze-bo3-ZgdBOa3Yi0KCkwa_Ap1ef3/aurora-vs-faze-m2-train.csv`
- round_num: `15`

## Largest probability jumps

- tick `108970`, seconds `54.50`, LSTM `0.4243`, delta `-0.2465`
- tick `107050`, seconds `24.50`, LSTM `0.8571`, delta `+0.2429`
- tick `109322`, seconds `60.00`, LSTM `0.8125`, delta `+0.1939`
- tick `109258`, seconds `59.00`, LSTM `0.5503`, delta `+0.1454`
- tick `109226`, seconds `58.50`, LSTM `0.4049`, delta `+0.0939`
- tick `110634`, seconds `80.50`, LSTM `0.8945`, delta `+0.0896`
- tick `110314`, seconds `75.50`, LSTM `0.7162`, delta `-0.0830`
- tick `108682`, seconds `50.00`, LSTM `0.7079`, delta `+0.0790`
- tick `110602`, seconds `80.00`, LSTM `0.8049`, delta `+0.0684`
- tick `109290`, seconds `59.50`, LSTM `0.6186`, delta `+0.0683`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004216`, |coef| `0.004216`
- `lag_00__CT_kills_last_3s`: coefficient `0.003996`, |coef| `0.003996`
- `lag_00__damage_diff_last_5s`: coefficient `0.003332`, |coef| `0.003332`
- `lag_00__CT_shots_fired_sum`: coefficient `0.003044`, |coef| `0.003044`
- `lag_00__CT_damage_last_5s`: coefficient `0.002578`, |coef| `0.002578`
- `lag_02__CT_shots_fired_sum`: coefficient `0.002314`, |coef| `0.002314`
- `lag_03__CT4__is_walking`: coefficient `-0.002134`, |coef| `0.002134`
- `lag_10__T3__flash_duration`: coefficient `0.002070`, |coef| `0.002070`
- `lag_01__damage_diff_last_5s`: coefficient `0.001998`, |coef| `0.001998`
- `lag_06__T_shots_fired_sum`: coefficient `0.001904`, |coef| `0.001904`
- `lag_01__CT4__is_walking`: coefficient `-0.001842`, |coef| `0.001842`
- `lag_00__T_bomb_zone_count`: coefficient `-0.001823`, |coef| `0.001823`
- `lag_14__CT_place_BACKOFB`: coefficient `0.001800`, |coef| `0.001800`
- `lag_01__CT_kills_last_3s`: coefficient `0.001788`, |coef| `0.001788`
- `lag_09__CT_place_LONGDOG`: coefficient `-0.001752`, |coef| `0.001752`

## Top 10 utility ridge features

- `lag_10__T3__flash_duration`: coefficient `0.002070` (raises CT win probability)
- `lag_09__T3__flash_duration`: coefficient `0.001494` (raises CT win probability)
- `lag_15__T3__flash_duration`: coefficient `-0.001465` (lowers CT win probability)
- `lag_06__CT_A_site_active_infernos`: coefficient `-0.001288` (lowers CT win probability)
- `lag_02__CT2__flash_duration`: coefficient `0.001272` (raises CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `0.001036` (raises CT win probability)
- `lag_01__T3__flash_duration`: coefficient `0.000988` (raises CT win probability)
- `lag_08__CT_A_site_active_infernos`: coefficient `-0.000945` (lowers CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.000924` (lowers CT win probability)
- `lag_07__CT5__molly`: coefficient `0.000904` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004216` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003996` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003332` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.003044` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002578` (raises CT win probability)
- `lag_02__CT_shots_fired_sum`: coefficient `0.002314` (raises CT win probability)
- `lag_03__CT4__is_walking`: coefficient `-0.002134` (lowers CT win probability)
- `lag_01__damage_diff_last_5s`: coefficient `0.001998` (raises CT win probability)
- `lag_06__T_shots_fired_sum`: coefficient `0.001904` (raises CT win probability)
- `lag_01__CT4__is_walking`: coefficient `-0.001842` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `108970`, seconds `54.50`, LSTM delta `-0.2465`

Top all feature movements:
- `lag_09__T_place_ELECTRICALBOX`: contribution `-0.030483`
- `lag_06__T_shots_fired_sum`: contribution `-0.012845`
- `lag_00__kill_diff_last_3s`: contribution `-0.010148`
- `lag_06__T4__shots_fired`: contribution `-0.008591`
- `lag_00__CT_place_LONGDOG`: contribution `-0.007277`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `107050`, seconds `24.50`, LSTM delta `+0.2429`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.023072`
- `lag_00__kill_diff_last_3s`: contribution `+0.020295`
- `lag_01__T_place_DUMPSTER`: contribution `+0.015460`
- `lag_00__CT_shots_fired_sum`: contribution `+0.010573`
- `lag_00__damage_diff_last_5s`: contribution `+0.009246`

Top utility-only movements:
- `lag_02__CT2__flash_duration`: contribution `+0.006662`
- `lag_06__CT_A_site_active_infernos`: contribution `+0.004545`
- `lag_08__CT_A_site_active_infernos`: contribution `+0.003335`

### tick `109322`, seconds `60.00`, LSTM delta `+0.1939`

Top all feature movements:
- `lag_03__T_place_ELECTRICALBOX`: contribution `+0.015410`
- `lag_00__CT_kills_last_3s`: contribution `+0.011536`
- `lag_00__CT_shots_fired_sum`: contribution `+0.010573`
- `lag_00__kill_diff_last_3s`: contribution `+0.010148`
- `lag_00__T_place_ELECTRICALBOX`: contribution `+0.008784`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `109258`, seconds `59.00`, LSTM delta `+0.1454`

Top all feature movements:
- `lag_01__T_place_ELECTRICALBOX`: contribution `+0.017137`
- `lag_00__CT_kills_last_3s`: contribution `+0.011536`
- `lag_09__CT_place_LONGDOG`: contribution `+0.011431`
- `lag_00__kill_diff_last_3s`: contribution `+0.010148`
- `lag_15__T_shots_fired_sum`: contribution `+0.008954`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `109226`, seconds `58.50`, LSTM delta `+0.0939`

Top all feature movements:
- `lag_14__T_shots_fired_sum`: contribution `+0.011376`
- `lag_08__CT_place_LONGDOG`: contribution `+0.009482`
- `lag_00__T_place_ELECTRICALBOX`: contribution `-0.008784`
- `lag_14__T4__shots_fired`: contribution `+0.007170`
- `lag_15__CT3__duck_amount`: contribution `+0.005428`

Top utility-only movements:
- No utility movement among the top local contributors.
