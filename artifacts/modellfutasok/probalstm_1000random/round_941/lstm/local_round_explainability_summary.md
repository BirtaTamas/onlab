# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-lynn-vision-vs-vitality-bo3-ZpOL0o26IrRvvgFRbFxVou/lynn-vision-vs-vitality-m1-dust2.csv`
- round_num: `3`

## Largest probability jumps

- tick `20012`, seconds `90.50`, LSTM `0.8380`, delta `+0.2464`
- tick `19500`, seconds `82.50`, LSTM `0.7143`, delta `+0.1267`
- tick `20140`, seconds `92.50`, LSTM `0.9445`, delta `+0.0756`
- tick `19884`, seconds `88.50`, LSTM `0.6217`, delta `-0.0546`
- tick `16556`, seconds `36.50`, LSTM `0.4385`, delta `-0.0510`
- tick `18316`, seconds `64.00`, LSTM `0.4452`, delta `-0.0490`
- tick `19820`, seconds `87.50`, LSTM `0.6898`, delta `-0.0476`
- tick `19468`, seconds `82.00`, LSTM `0.5876`, delta `+0.0411`
- tick `19564`, seconds `83.50`, LSTM `0.7654`, delta `+0.0398`
- tick `21068`, seconds `107.00`, LSTM `0.9032`, delta `+0.0380`

## Top 15 local ridge features

- `lag_00__T_place_UNDERA`: coefficient `-0.003367`, |coef| `0.003367`
- `lag_04__T_place_UNDERA`: coefficient `0.002196`, |coef| `0.002196`
- `lag_00__CT_kills_last_3s`: coefficient `0.001727`, |coef| `0.001727`
- `lag_04__T_place_EXTENDEDA`: coefficient `-0.001643`, |coef| `0.001643`
- `lag_14__T1__duck_amount`: coefficient `0.001601`, |coef| `0.001601`
- `lag_00__kill_diff_last_3s`: coefficient `0.001498`, |coef| `0.001498`
- `lag_07__CT3__is_walking`: coefficient `-0.001457`, |coef| `0.001457`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001359`, |coef| `0.001359`
- `lag_00__damage_diff_last_5s`: coefficient `0.001356`, |coef| `0.001356`
- `lag_12__T_place_SHORTSTAIRS`: coefficient `0.001321`, |coef| `0.001321`
- `lag_08__T_place_UNDERA`: coefficient `0.001269`, |coef| `0.001269`
- `lag_01__T_place_LOWERTUNNEL`: coefficient `0.001210`, |coef| `0.001210`
- `lag_00__bomb_events_last_5s`: coefficient `0.001182`, |coef| `0.001182`
- `lag_10__T1__duck_amount`: coefficient `0.001159`, |coef| `0.001159`
- `lag_04__T3__has_bomb`: coefficient `0.001153`, |coef| `0.001153`

## Top 10 utility ridge features

- `lag_08__T_flashes_last_5s`: coefficient `-0.000881` (lowers CT win probability)
- `lag_02__T_flashes_last_5s`: coefficient `0.000863` (raises CT win probability)
- `lag_00__T5__smoke`: coefficient `-0.000772` (lowers CT win probability)
- `lag_09__T_flashes_last_5s`: coefficient `-0.000769` (lowers CT win probability)
- `lag_12__CT_active_infernos`: coefficient `-0.000757` (lowers CT win probability)
- `lag_10__CT1__molly`: coefficient `-0.000750` (lowers CT win probability)
- `lag_06__T_A_site_active_infernos`: coefficient `0.000722` (raises CT win probability)
- `lag_04__T_flashes_last_5s`: coefficient `-0.000701` (lowers CT win probability)
- `lag_08__CT_B_site_active_smokes`: coefficient `0.000659` (raises CT win probability)
- `lag_15__T4__smoke`: coefficient `-0.000629` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_UNDERA`: coefficient `-0.003367` (lowers CT win probability)
- `lag_04__T_place_UNDERA`: coefficient `0.002196` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001727` (raises CT win probability)
- `lag_04__T_place_EXTENDEDA`: coefficient `-0.001643` (lowers CT win probability)
- `lag_14__T1__duck_amount`: coefficient `0.001601` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001498` (raises CT win probability)
- `lag_07__CT3__is_walking`: coefficient `-0.001457` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001359` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001356` (raises CT win probability)
- `lag_12__T_place_SHORTSTAIRS`: coefficient `0.001321` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `20012`, seconds `90.50`, LSTM delta `+0.2464`

Top all feature movements:
- `lag_00__T_place_UNDERA`: contribution `+0.052622`
- `lag_04__T_place_UNDERA`: contribution `+0.034320`
- `lag_04__T_place_EXTENDEDA`: contribution `+0.008147`
- `lag_06__T_place_EXTENDEDA`: contribution `+0.005589`
- `lag_00__CT_place_ARAMP`: contribution `+0.005237`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `19500`, seconds `82.50`, LSTM delta `+0.1267`

Top all feature movements:
- `lag_08__CT_place_OUTSIDELONG`: contribution `+0.010403`
- `lag_14__T1__duck_amount`: contribution `+0.006267`
- `lag_00__CT_kills_last_3s`: contribution `+0.004987`
- `lag_15__CT_place_ARAMP`: contribution `+0.004471`
- `lag_00__kill_diff_last_3s`: contribution `+0.003606`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `20140`, seconds `92.50`, LSTM delta `+0.0756`

Top all feature movements:
- `lag_04__T_place_UNDERA`: contribution `-0.034320`
- `lag_08__T_place_UNDERA`: contribution `+0.019834`
- `lag_00__CT_shots_fired_sum`: contribution `+0.006607`
- `lag_00__CT_kills_last_3s`: contribution `+0.004987`
- `lag_10__T_place_EXTENDEDA`: contribution `+0.004180`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `19884`, seconds `88.50`, LSTM delta `-0.0546`

Top all feature movements:
- `lag_00__T_place_UNDERA`: contribution `-0.052622`
- `lag_07__CT3__is_walking`: contribution `-0.003479`
- `lag_12__T2__duck_amount`: contribution `-0.002841`
- `lag_07__CT_walking_count`: contribution `-0.002739`
- `lag_00__CT3__is_walking`: contribution `-0.002595`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `16556`, seconds `36.50`, LSTM delta `-0.0510`

Top all feature movements:
- `lag_04__T_place_EXTENDEDA`: contribution `-0.008147`
- `lag_08__T_flashes_last_5s`: contribution `-0.007984`
- `lag_01__T_place_SHORTSTAIRS`: contribution `-0.004402`
- `lag_00__CT1__flash_duration`: contribution `-0.003359`
- `lag_13__CT_place_PIT`: contribution `-0.002667`

Top utility-only movements:
- `lag_08__T_flashes_last_5s`: contribution `-0.007984`
- `lag_00__CT1__flash_duration`: contribution `-0.003359`
