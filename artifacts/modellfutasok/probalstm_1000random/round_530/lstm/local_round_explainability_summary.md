# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-aurora-vs-faze-bo3-ZgdBOa3Yi0KCkwa_Ap1ef3/aurora-vs-faze-m2-train.csv`
- round_num: `6`

## Largest probability jumps

- tick `33194`, seconds `60.00`, LSTM `0.1960`, delta `-0.2437`
- tick `34250`, seconds `76.50`, LSTM `0.0657`, delta `-0.2353`
- tick `34058`, seconds `73.50`, LSTM `0.1855`, delta `+0.0988`
- tick `33226`, seconds `60.50`, LSTM `0.1318`, delta `-0.0641`
- tick `34122`, seconds `74.50`, LSTM `0.2665`, delta `+0.0479`
- tick `33386`, seconds `63.00`, LSTM `0.1660`, delta `+0.0429`
- tick `33610`, seconds `66.50`, LSTM `0.0997`, delta `-0.0379`
- tick `34026`, seconds `73.00`, LSTM `0.0868`, delta `+0.0363`
- tick `34090`, seconds `74.00`, LSTM `0.2186`, delta `+0.0330`
- tick `34282`, seconds `77.00`, LSTM `0.0357`, delta `-0.0301`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002660`, |coef| `0.002660`
- `lag_00__CT_place_BACKOFB`: coefficient `0.002433`, |coef| `0.002433`
- `lag_00__T_kills_last_3s`: coefficient `-0.002241`, |coef| `0.002241`
- `lag_01__T3__duck_amount`: coefficient `-0.001940`, |coef| `0.001940`
- `lag_13__CT_place_BACKOFB`: coefficient `-0.001853`, |coef| `0.001853`
- `lag_00__T_damage_last_5s`: coefficient `-0.001663`, |coef| `0.001663`
- `lag_13__T_place_LONGDOG`: coefficient `0.001584`, |coef| `0.001584`
- `lag_00__damage_diff_last_5s`: coefficient `0.001422`, |coef| `0.001422`
- `lag_05__T_A_site_active_infernos`: coefficient `-0.001338`, |coef| `0.001338`
- `lag_00__CT3__molly`: coefficient `0.001323`, |coef| `0.001323`
- `lag_06__T_bomb_zone_count`: coefficient `-0.001311`, |coef| `0.001311`
- `lag_00__T_bomb_zone_count`: coefficient `0.001301`, |coef| `0.001301`
- `lag_00__CT3__alive`: coefficient `0.001299`, |coef| `0.001299`
- `lag_13__T_place_BACKOFB`: coefficient `-0.001297`, |coef| `0.001297`
- `lag_01__CT_place_BACKOFB`: coefficient `0.001294`, |coef| `0.001294`

## Top 10 utility ridge features

- `lag_05__T_A_site_active_infernos`: coefficient `-0.001338` (lowers CT win probability)
- `lag_00__CT3__molly`: coefficient `0.001323` (raises CT win probability)
- `lag_01__CT4__molly`: coefficient `0.001144` (raises CT win probability)
- `lag_12__T4__molly`: coefficient `0.001041` (raises CT win probability)
- `lag_11__T_B_site_active_infernos`: coefficient `0.001028` (raises CT win probability)
- `lag_14__CT1__smoke`: coefficient `0.001020` (raises CT win probability)
- `lag_00__CT_molly_inv`: coefficient `0.000795` (raises CT win probability)
- `lag_12__T1__molly`: coefficient `-0.000785` (lowers CT win probability)
- `lag_08__CT_A_site_active_smokes`: coefficient `0.000768` (raises CT win probability)
- `lag_05__T_B_site_active_infernos`: coefficient `-0.000742` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002660` (raises CT win probability)
- `lag_00__CT_place_BACKOFB`: coefficient `0.002433` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002241` (lowers CT win probability)
- `lag_01__T3__duck_amount`: coefficient `-0.001940` (lowers CT win probability)
- `lag_13__CT_place_BACKOFB`: coefficient `-0.001853` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001663` (lowers CT win probability)
- `lag_13__T_place_LONGDOG`: coefficient `0.001584` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001422` (raises CT win probability)
- `lag_06__T_bomb_zone_count`: coefficient `-0.001311` (lowers CT win probability)
- `lag_00__T_bomb_zone_count`: coefficient `0.001301` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `33194`, seconds `60.00`, LSTM delta `-0.2437`

Top all feature movements:
- `lag_00__CT_place_BACKOFB`: contribution `-0.013894`
- `lag_13__CT_place_BACKOFB`: contribution `-0.010580`
- `lag_13__T_place_LONGDOG`: contribution `-0.007373`
- `lag_01__T3__duck_amount`: contribution `-0.007316`
- `lag_00__T_kills_last_3s`: contribution `-0.007100`

Top utility-only movements:
- `lag_00__CT3__molly`: contribution `-0.003267`
- `lag_01__CT4__molly`: contribution `-0.002817`

### tick `34250`, seconds `76.50`, LSTM delta `-0.2353`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.012806`
- `lag_11__CT_place_ENTRANCE`: contribution `-0.008368`
- `lag_09__CT_place_ENTRANCE`: contribution `-0.008219`
- `lag_06__T_bomb_zone_count`: contribution `-0.007629`
- `lag_01__T3__duck_amount`: contribution `-0.007316`

Top utility-only movements:
- `lag_05__T_A_site_active_infernos`: contribution `-0.003983`
- `lag_11__T_B_site_active_infernos`: contribution `-0.002907`

### tick `34058`, seconds `73.50`, LSTM delta `+0.0988`

Top all feature movements:
- `lag_00__T_bomb_zone_count`: contribution `+0.007575`
- `lag_03__CT_place_ENTRANCE`: contribution `+0.007328`
- `lag_00__kill_diff_last_3s`: contribution `+0.006403`
- `lag_05__T_A_site_active_infernos`: contribution `+0.003983`
- `lag_15__T5__duck_amount`: contribution `+0.003555`

Top utility-only movements:
- `lag_05__T_A_site_active_infernos`: contribution `+0.003983`
- `lag_11__T_B_site_active_infernos`: contribution `+0.002907`
- `lag_05__T_B_site_active_infernos`: contribution `+0.002099`
- `lag_11__T_active_infernos`: contribution `+0.001531`
- `lag_15__CT_A_site_active_infernos`: contribution `+0.001497`

### tick `33226`, seconds `60.50`, LSTM delta `-0.0641`

Top all feature movements:
- `lag_01__CT_place_BACKOFB`: contribution `-0.007386`
- `lag_14__CT_place_BACKOFB`: contribution `-0.005835`
- `lag_00__T_shots_fired_sum`: contribution `+0.005628`
- `lag_01__T_shots_fired_sum`: contribution `-0.004747`
- `lag_14__T_place_LONGDOG`: contribution `-0.004386`

Top utility-only movements:
- `lag_00__CT_A_site_active_infernos`: contribution `-0.002118`

### tick `34122`, seconds `74.50`, LSTM delta `+0.0479`

Top all feature movements:
- `lag_00__CT4__duck_amount`: contribution `+0.004382`
- `lag_07__CT_place_ENTRANCE`: contribution `+0.003961`
- `lag_13__T_place_BACKOFB`: contribution `+0.003483`
- `lag_02__T_bomb_zone_count`: contribution `+0.003289`
- `lag_03__T4__duck_amount`: contribution `+0.003256`

Top utility-only movements:
- `lag_11__T3__flash_duration`: contribution `+0.001563`
