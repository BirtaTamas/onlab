# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-lynn-vision-vs-vitality-bo3-ZpOL0o26IrRvvgFRbFxVou/lynn-vision-vs-vitality-m1-dust2.csv`
- round_num: `11`

## Largest probability jumps

- tick `81397`, seconds `64.50`, LSTM `0.7669`, delta `-0.1183`
- tick `82453`, seconds `81.00`, LSTM `0.9141`, delta `+0.1124`
- tick `82517`, seconds `82.00`, LSTM `0.8245`, delta `-0.0914`
- tick `83637`, seconds `99.50`, LSTM `0.9425`, delta `+0.0708`
- tick `78325`, seconds `16.50`, LSTM `0.7884`, delta `+0.0685`
- tick `83605`, seconds `99.00`, LSTM `0.8717`, delta `+0.0552`
- tick `81045`, seconds `59.00`, LSTM `0.8794`, delta `+0.0514`
- tick `81557`, seconds `67.00`, LSTM `0.7424`, delta `-0.0400`
- tick `82357`, seconds `79.50`, LSTM `0.7812`, delta `-0.0365`
- tick `77301`, seconds `0.50`, LSTM `0.7781`, delta `+0.0357`

## Top 15 local ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.002189`, |coef| `0.002189`
- `lag_00__kill_diff_last_3s`: coefficient `0.001839`, |coef| `0.001839`
- `lag_00__damage_diff_last_5s`: coefficient `0.001528`, |coef| `0.001528`
- `lag_12__T1__duck_amount`: coefficient `0.001493`, |coef| `0.001493`
- `lag_15__CT_place_ARAMP`: coefficient `-0.001358`, |coef| `0.001358`
- `lag_15__T5__is_walking`: coefficient `-0.001225`, |coef| `0.001225`
- `lag_00__CT_kills_last_3s`: coefficient `0.001199`, |coef| `0.001199`
- `lag_02__CT_place_HOLE`: coefficient `-0.001183`, |coef| `0.001183`
- `lag_01__T1__shots_fired`: coefficient `0.001146`, |coef| `0.001146`
- `lag_00__T_kills_last_3s`: coefficient `-0.001105`, |coef| `0.001105`
- `lag_03__CT5__duck_amount`: coefficient `-0.001103`, |coef| `0.001103`
- `lag_11__T1__duck_amount`: coefficient `0.001024`, |coef| `0.001024`
- `lag_00__T_place_LONGDOORS`: coefficient `-0.000977`, |coef| `0.000977`
- `lag_07__CT_place_EXTENDEDA`: coefficient `0.000955`, |coef| `0.000955`
- `lag_10__CT_place_EXTENDEDA`: coefficient `0.000930`, |coef| `0.000930`

## Top 10 utility ridge features

- `lag_09__CT_A_site_active_infernos`: coefficient `0.000890` (raises CT win probability)
- `lag_14__T5__flash_duration`: coefficient `0.000863` (raises CT win probability)
- `lag_00__T4__flash`: coefficient `-0.000814` (lowers CT win probability)
- `lag_07__CT_A_site_active_infernos`: coefficient `-0.000778` (lowers CT win probability)
- `lag_00__T5__flash_duration`: coefficient `-0.000767` (lowers CT win probability)
- `lag_15__CT1__flash`: coefficient `-0.000678` (lowers CT win probability)
- `lag_09__CT_active_infernos`: coefficient `0.000591` (raises CT win probability)
- `lag_02__T4__flash`: coefficient `0.000557` (raises CT win probability)
- `lag_12__CT2__smoke`: coefficient `0.000524` (raises CT win probability)
- `lag_07__CT_active_infernos`: coefficient `-0.000505` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.002189` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001839` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001528` (raises CT win probability)
- `lag_12__T1__duck_amount`: coefficient `0.001493` (raises CT win probability)
- `lag_15__CT_place_ARAMP`: coefficient `-0.001358` (lowers CT win probability)
- `lag_15__T5__is_walking`: coefficient `-0.001225` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001199` (raises CT win probability)
- `lag_02__CT_place_HOLE`: coefficient `-0.001183` (lowers CT win probability)
- `lag_01__T1__shots_fired`: coefficient `0.001146` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001105` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `81397`, seconds `64.50`, LSTM delta `-0.1183`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.015206`
- `lag_02__CT_place_HOLE`: contribution `-0.013212`
- `lag_00__kill_diff_last_3s`: contribution `-0.004427`
- `lag_15__CT_place_EXTENDEDA`: contribution `-0.004142`
- `lag_00__CT2__shots_fired`: contribution `-0.003922`

Top utility-only movements:
- `lag_14__T5__flash_duration`: contribution `-0.001999`

### tick `82453`, seconds `81.00`, LSTM delta `+0.1124`

Top all feature movements:
- `lag_15__CT_place_ARAMP`: contribution `+0.008461`
- `lag_00__CT_shots_fired_sum`: contribution `+0.007603`
- `lag_12__T1__duck_amount`: contribution `+0.005847`
- `lag_07__CT_place_EXTENDEDA`: contribution `+0.005361`
- `lag_00__kill_diff_last_3s`: contribution `+0.004427`

Top utility-only movements:
- `lag_07__CT_A_site_active_infernos`: contribution `+0.002745`
- `lag_00__T4__flash`: contribution `+0.002212`

### tick `82517`, seconds `82.00`, LSTM delta `-0.0914`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.004427`
- `lag_11__T1__duck_amount`: contribution `-0.004009`
- `lag_00__T_kills_last_3s`: contribution `-0.003500`
- `lag_09__CT_A_site_active_infernos`: contribution `-0.003141`
- `lag_05__CT5__duck_amount`: contribution `-0.002959`

Top utility-only movements:
- `lag_09__CT_A_site_active_infernos`: contribution `-0.003141`
- `lag_02__T4__flash`: contribution `-0.001512`

### tick `83637`, seconds `99.50`, LSTM delta `+0.0708`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.007603`
- `lag_00__T_shots_fired_sum`: contribution `+0.007159`
- `lag_00__kill_diff_last_3s`: contribution `+0.004427`
- `lag_00__CT_kills_last_3s`: contribution `+0.003462`
- `lag_01__T1__shots_fired`: contribution `+0.003424`

Top utility-only movements:
- `lag_13__T5__flash_duration`: contribution `+0.001954`

### tick `78325`, seconds `16.50`, LSTM delta `+0.0685`

Top all feature movements:
- `lag_01__CT_place_SIDE`: contribution `+0.024165`
- `lag_00__CT_shots_fired_sum`: contribution `+0.006082`
- `lag_00__kill_diff_last_3s`: contribution `+0.004427`
- `lag_00__CT_place_EXTENDEDA`: contribution `+0.004204`
- `lag_00__CT_kills_last_3s`: contribution `+0.003462`

Top utility-only movements:
- `lag_13__T_flashes_last_5s`: contribution `+0.002383`
- `lag_08__CT2__flash_duration`: contribution `+0.001766`
