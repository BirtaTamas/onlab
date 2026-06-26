# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-faze-vs-aurora-bo3-OcxcOl9bFIHQQ2588nwUWG/faze-vs-aurora-m1-inferno.csv`
- round_num: `18`

## Largest probability jumps

- tick `130767`, seconds `42.00`, LSTM `0.7722`, delta `+0.1832`
- tick `131087`, seconds `47.00`, LSTM `0.6418`, delta `-0.1191`
- tick `131151`, seconds `48.00`, LSTM `0.7695`, delta `+0.1190`
- tick `131279`, seconds `50.00`, LSTM `0.8838`, delta `+0.0878`
- tick `131407`, seconds `52.00`, LSTM `0.9432`, delta `+0.0833`
- tick `129551`, seconds `23.00`, LSTM `0.6691`, delta `+0.0468`
- tick `131023`, seconds `46.00`, LSTM `0.7545`, delta `-0.0397`
- tick `129871`, seconds `28.00`, LSTM `0.6512`, delta `-0.0276`
- tick `129455`, seconds `21.50`, LSTM `0.6392`, delta `+0.0225`
- tick `128687`, seconds `9.50`, LSTM `0.6318`, delta `+0.0220`

## Top 15 local ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.001925`, |coef| `0.001925`
- `lag_01__T_shots_fired_sum`: coefficient `-0.001831`, |coef| `0.001831`
- `lag_00__CT2__duck_amount`: coefficient `0.001459`, |coef| `0.001459`
- `lag_01__T1__duck_amount`: coefficient `-0.001442`, |coef| `0.001442`
- `lag_00__CT_kills_last_3s`: coefficient `0.001428`, |coef| `0.001428`
- `lag_00__kill_diff_last_3s`: coefficient `0.001376`, |coef| `0.001376`
- `lag_00__CT_damage_last_5s`: coefficient `0.001356`, |coef| `0.001356`
- `lag_00__damage_diff_last_5s`: coefficient `0.001257`, |coef| `0.001257`
- `lag_07__T_place_SECONDMID`: coefficient `-0.001222`, |coef| `0.001222`
- `lag_01__T3__shots_fired`: coefficient `-0.001169`, |coef| `0.001169`
- `lag_07__T5__duck_amount`: coefficient `-0.001134`, |coef| `0.001134`
- `lag_00__T4__has_bomb`: coefficient `-0.001128`, |coef| `0.001128`
- `lag_00__bomb_events_last_5s`: coefficient `0.001105`, |coef| `0.001105`
- `lag_12__T5__is_walking`: coefficient `0.001101`, |coef| `0.001101`
- `lag_02__CT_place_BANANA`: coefficient `-0.001084`, |coef| `0.001084`

## Top 10 utility ridge features

- `lag_00__T4__molly`: coefficient `-0.000948` (lowers CT win probability)
- `lag_08__T4__smoke`: coefficient `-0.000897` (lowers CT win probability)
- `lag_08__T3__smoke`: coefficient `-0.000897` (lowers CT win probability)
- `lag_02__CT2__flash_duration`: coefficient `-0.000840` (lowers CT win probability)
- `lag_01__T5__molly`: coefficient `-0.000832` (lowers CT win probability)
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.000705` (raises CT win probability)
- `lag_02__CT_utility_damage_last_5s`: coefficient `0.000681` (raises CT win probability)
- `lag_02__CT_flash_duration_sum`: coefficient `-0.000679` (lowers CT win probability)
- `lag_00__CT5__flash_duration`: coefficient `-0.000650` (lowers CT win probability)
- `lag_01__CT_utility_damage_last_5s`: coefficient `0.000645` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.001925` (raises CT win probability)
- `lag_01__T_shots_fired_sum`: coefficient `-0.001831` (lowers CT win probability)
- `lag_00__CT2__duck_amount`: coefficient `0.001459` (raises CT win probability)
- `lag_01__T1__duck_amount`: coefficient `-0.001442` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001428` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001376` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001356` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001257` (raises CT win probability)
- `lag_07__T_place_SECONDMID`: coefficient `-0.001222` (lowers CT win probability)
- `lag_01__T3__shots_fired`: coefficient `-0.001169` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `130767`, seconds `42.00`, LSTM delta `+0.1832`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.010698`
- `lag_01__T_shots_fired_sum`: contribution `+0.008237`
- `lag_00__CT2__duck_amount`: contribution `+0.005557`
- `lag_07__T5__duck_amount`: contribution `+0.004307`
- `lag_01__T3__shots_fired`: contribution `+0.004247`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `131087`, seconds `47.00`, LSTM delta `-0.1191`

Top all feature movements:
- `lag_07__T_place_BALCONY`: contribution `-0.011184`
- `lag_00__CT_shots_fired_sum`: contribution `-0.008024`
- `lag_01__T_shots_fired_sum`: contribution `-0.006864`
- `lag_00__CT2__duck_amount`: contribution `-0.005557`
- `lag_02__CT2__flash_duration`: contribution `-0.005457`

Top utility-only movements:
- `lag_02__CT2__flash_duration`: contribution `-0.005457`
- `lag_02__CT_flash_duration_sum`: contribution `-0.004808`
- `lag_02__CT5__flash_duration`: contribution `-0.003248`

### tick `131151`, seconds `48.00`, LSTM delta `+0.1190`

Top all feature movements:
- `lag_01__T_shots_fired_sum`: contribution `+0.015102`
- `lag_09__T_place_BALCONY`: contribution `+0.009446`
- `lag_00__CT_shots_fired_sum`: contribution `+0.006686`
- `lag_02__CT2__flash_duration`: contribution `+0.005457`
- `lag_04__CT_place_QUAD`: contribution `+0.004247`

Top utility-only movements:
- `lag_02__CT2__flash_duration`: contribution `+0.005457`
- `lag_04__CT5__flash_duration`: contribution `+0.002594`
- `lag_02__CT_flash_duration_sum`: contribution `+0.002002`
- `lag_04__CT_flash_duration_sum`: contribution `+0.001624`

### tick `131279`, seconds `50.00`, LSTM delta `+0.0878`

Top all feature movements:
- `lag_13__T_place_BALCONY`: contribution `+0.010464`
- `lag_00__CT_kills_last_3s`: contribution `+0.004122`
- `lag_00__kill_diff_last_3s`: contribution `+0.003313`
- `lag_08__CT5__flash_duration`: contribution `+0.002861`
- `lag_08__CT_flash_duration_sum`: contribution `+0.002640`

Top utility-only movements:
- `lag_08__CT5__flash_duration`: contribution `+0.002861`
- `lag_08__CT_flash_duration_sum`: contribution `+0.002640`
- `lag_08__CT2__flash_duration`: contribution `+0.002152`
- `lag_06__CT2__flash_duration`: contribution `+0.001580`

### tick `131407`, seconds `52.00`, LSTM delta `+0.0833`

Top all feature movements:
- `lag_04__T_place_BALCONY`: contribution `+0.004778`
- `lag_00__CT5__flash_duration`: contribution `+0.004293`
- `lag_04__CT_place_QUAD`: contribution `+0.004247`
- `lag_00__CT_kills_last_3s`: contribution `+0.004122`
- `lag_12__CT5__flash_duration`: contribution `+0.003763`

Top utility-only movements:
- `lag_00__CT5__flash_duration`: contribution `+0.004293`
- `lag_12__CT5__flash_duration`: contribution `+0.003763`
- `lag_12__CT_flash_duration_sum`: contribution `+0.002919`
- `lag_12__CT2__flash_duration`: contribution `+0.001587`
- `lag_10__CT2__flash_duration`: contribution `+0.001489`
