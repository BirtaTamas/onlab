# Local Round Explainability

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-vitality-vs-faze-bo3-hDX5yjYYbla4cw8aPwAYi3/vitality-vs-faze-m1-nuke.csv`
- round_num: `14`

## Largest probability jumps

- tick `115870`, seconds `100.50`, LSTM `0.9306`, delta `+0.1924`
- tick `111710`, seconds `35.50`, LSTM `0.7580`, delta `+0.1602`
- tick `115518`, seconds `95.00`, LSTM `0.8542`, delta `+0.1450`
- tick `115646`, seconds `97.00`, LSTM `0.8080`, delta `-0.1042`
- tick `111934`, seconds `39.00`, LSTM `0.6983`, delta `-0.0854`
- tick `115710`, seconds `98.00`, LSTM `0.7256`, delta `-0.0701`
- tick `111902`, seconds `38.50`, LSTM `0.7836`, delta `-0.0553`
- tick `111742`, seconds `36.00`, LSTM `0.8108`, delta `+0.0527`
- tick `112830`, seconds `53.00`, LSTM `0.7021`, delta `+0.0517`
- tick `112094`, seconds `41.50`, LSTM `0.6146`, delta `-0.0481`

## Top 15 local ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.003124`, |coef| `0.003124`
- `lag_00__CT_kills_last_3s`: coefficient `0.002739`, |coef| `0.002739`
- `lag_00__kill_diff_last_3s`: coefficient `0.002640`, |coef| `0.002640`
- `lag_04__CT_place_GARAGE`: coefficient `0.002320`, |coef| `0.002320`
- `lag_00__T_place_SQUEAKY`: coefficient `-0.002191`, |coef| `0.002191`
- `lag_02__CT_shots_fired_sum`: coefficient `0.001919`, |coef| `0.001919`
- `lag_00__damage_diff_last_5s`: coefficient `0.001846`, |coef| `0.001846`
- `lag_04__CT_place_CONTROL`: coefficient `0.001716`, |coef| `0.001716`
- `lag_00__CT_place_ADMIN`: coefficient `0.001695`, |coef| `0.001695`
- `lag_00__CT_damage_last_5s`: coefficient `0.001681`, |coef| `0.001681`
- `lag_00__CT3__duck_amount`: coefficient `0.001551`, |coef| `0.001551`
- `lag_05__T_place_HUT`: coefficient `0.001541`, |coef| `0.001541`
- `lag_08__CT_place_HUTROOF`: coefficient `-0.001465`, |coef| `0.001465`
- `lag_03__CT_shots_fired_sum`: coefficient `0.001394`, |coef| `0.001394`
- `lag_05__CT_place_LOCKERROOM`: coefficient `0.001350`, |coef| `0.001350`

## Top 10 utility ridge features

- `lag_12__T3__flash_duration`: coefficient `-0.001316` (lowers CT win probability)
- `lag_09__T_A_site_active_smokes`: coefficient `0.001032` (raises CT win probability)
- `lag_15__T5__smoke`: coefficient `-0.001020` (lowers CT win probability)
- `lag_06__CT2__molly`: coefficient `-0.000994` (lowers CT win probability)
- `lag_04__CT_active_infernos`: coefficient `0.000925` (raises CT win probability)
- `lag_15__T4__smoke`: coefficient `-0.000915` (lowers CT win probability)
- `lag_13__T3__flash_duration`: coefficient `-0.000899` (lowers CT win probability)
- `lag_09__CT3__smoke`: coefficient `-0.000880` (lowers CT win probability)
- `lag_11__T3__flash_duration`: coefficient `-0.000863` (lowers CT win probability)
- `lag_04__CT4__smoke`: coefficient `-0.000839` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.003124` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002739` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002640` (raises CT win probability)
- `lag_04__CT_place_GARAGE`: coefficient `0.002320` (raises CT win probability)
- `lag_00__T_place_SQUEAKY`: coefficient `-0.002191` (lowers CT win probability)
- `lag_02__CT_shots_fired_sum`: coefficient `0.001919` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001846` (raises CT win probability)
- `lag_04__CT_place_CONTROL`: coefficient `0.001716` (raises CT win probability)
- `lag_00__CT_place_ADMIN`: coefficient `0.001695` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001681` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `115870`, seconds `100.50`, LSTM delta `+0.1924`

Top all feature movements:
- `lag_04__CT_place_CONTROL`: contribution `+0.017812`
- `lag_00__CT_shots_fired_sum`: contribution `+0.017366`
- `lag_05__T_place_HUT`: contribution `+0.014364`
- `lag_00__CT_place_ADMIN`: contribution `+0.011778`
- `lag_00__CT_kills_last_3s`: contribution `+0.007909`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `111710`, seconds `35.50`, LSTM delta `+0.1602`

Top all feature movements:
- `lag_04__CT_place_GARAGE`: contribution `+0.016673`
- `lag_00__CT_shots_fired_sum`: contribution `+0.010854`
- `lag_00__CT_kills_last_3s`: contribution `+0.007909`
- `lag_00__kill_diff_last_3s`: contribution `+0.006353`
- `lag_01__T1__duck_amount`: contribution `+0.004709`

Top utility-only movements:
- `lag_09__T_A_site_active_smokes`: contribution `+0.002936`
- `lag_06__CT2__molly`: contribution `+0.002451`
- `lag_15__T5__smoke`: contribution `+0.002209`

### tick `115518`, seconds `95.00`, LSTM delta `+0.1450`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.015195`
- `lag_00__T_place_SQUEAKY`: contribution `+0.013639`
- `lag_08__CT_place_HUTROOF`: contribution `+0.010254`
- `lag_12__T3__flash_duration`: contribution `+0.008065`
- `lag_00__CT_kills_last_3s`: contribution `+0.007909`

Top utility-only movements:
- `lag_12__T3__flash_duration`: contribution `+0.008065`
- `lag_10__T_A_site_active_infernos`: contribution `+0.002329`
- `lag_10__T_B_site_active_infernos`: contribution `+0.002095`

### tick `115646`, seconds `97.00`, LSTM delta `-0.1042`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.019536`
- `lag_03__CT_shots_fired_sum`: contribution `-0.011619`
- `lag_04__T_place_SQUEAKY`: contribution `-0.007187`
- `lag_00__kill_diff_last_3s`: contribution `-0.006353`
- `lag_00__CT3__shots_fired`: contribution `-0.004912`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `111934`, seconds `39.00`, LSTM delta `-0.0854`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.028219`
- `lag_03__CT_shots_fired_sum`: contribution `-0.016460`
- `lag_02__CT_shots_fired_sum`: contribution `+0.006666`
- `lag_00__CT3__duck_amount`: contribution `-0.005772`
- `lag_03__CT1__shots_fired`: contribution `-0.004702`

Top utility-only movements:
- No utility movement among the top local contributors.
