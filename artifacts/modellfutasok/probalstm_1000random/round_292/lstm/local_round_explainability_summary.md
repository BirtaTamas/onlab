# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-tyloo-vs-falcons-bo3-MBKGKnSCeuy54EHzS5mmW8/tyloo-vs-falcons-m2-ancient.csv`
- round_num: `6`

## Largest probability jumps

- tick `43548`, seconds `68.50`, LSTM `0.0399`, delta `-0.3253`
- tick `39196`, seconds `0.50`, LSTM `0.1074`, delta `-0.0665`
- tick `40828`, seconds `26.00`, LSTM `0.3154`, delta `+0.0664`
- tick `43132`, seconds `62.00`, LSTM `0.2983`, delta `+0.0663`
- tick `43068`, seconds `61.00`, LSTM `0.2009`, delta `-0.0604`
- tick `42652`, seconds `54.50`, LSTM `0.2511`, delta `-0.0518`
- tick `40796`, seconds `25.50`, LSTM `0.2491`, delta `-0.0475`
- tick `43516`, seconds `68.00`, LSTM `0.3652`, delta `+0.0472`
- tick `41756`, seconds `40.50`, LSTM `0.2219`, delta `-0.0468`
- tick `42684`, seconds `55.00`, LSTM `0.2940`, delta `+0.0429`

## Top 15 local ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.002722`, |coef| `0.002722`
- `lag_01__T3__flash_duration`: coefficient `0.002136`, |coef| `0.002136`
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.002059`, |coef| `0.002059`
- `lag_14__T_place_RAMP`: coefficient `0.001984`, |coef| `0.001984`
- `lag_00__CT_place_MIDDLE`: coefficient `0.001828`, |coef| `0.001828`
- `lag_05__T5__flash_duration`: coefficient `-0.001787`, |coef| `0.001787`
- `lag_00__T4__is_walking`: coefficient `0.001717`, |coef| `0.001717`
- `lag_00__T_kills_last_3s`: coefficient `-0.001697`, |coef| `0.001697`
- `lag_01__CT_shots_fired_sum`: coefficient `0.001637`, |coef| `0.001637`
- `lag_00__T5__is_scoped`: coefficient `0.001565`, |coef| `0.001565`
- `lag_14__T_place_TSIDELOWER`: coefficient `-0.001537`, |coef| `0.001537`
- `lag_02__T3__flash_duration`: coefficient `-0.001520`, |coef| `0.001520`
- `lag_00__CT1__is_walking`: coefficient `0.001509`, |coef| `0.001509`
- `lag_00__T3__duck_amount`: coefficient `-0.001338`, |coef| `0.001338`
- `lag_00__T1__shots_fired`: coefficient `-0.001337`, |coef| `0.001337`

## Top 10 utility ridge features

- `lag_01__T3__flash_duration`: coefficient `0.002136` (raises CT win probability)
- `lag_05__T5__flash_duration`: coefficient `-0.001787` (lowers CT win probability)
- `lag_02__T3__flash_duration`: coefficient `-0.001520` (lowers CT win probability)
- `lag_04__CT2__flash_duration`: coefficient `-0.001006` (lowers CT win probability)
- `lag_13__CT1__flash_duration`: coefficient `-0.000922` (lowers CT win probability)
- `lag_02__T_flash_duration_sum`: coefficient `-0.000866` (lowers CT win probability)
- `lag_01__T3__molly`: coefficient `0.000772` (raises CT win probability)
- `lag_01__T3__smoke`: coefficient `0.000758` (raises CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `0.000708` (raises CT win probability)
- `lag_06__T5__flash_duration`: coefficient `-0.000676` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.002722` (lowers CT win probability)
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.002059` (raises CT win probability)
- `lag_14__T_place_RAMP`: coefficient `0.001984` (raises CT win probability)
- `lag_00__CT_place_MIDDLE`: coefficient `0.001828` (raises CT win probability)
- `lag_00__T4__is_walking`: coefficient `0.001717` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001697` (lowers CT win probability)
- `lag_01__CT_shots_fired_sum`: coefficient `0.001637` (raises CT win probability)
- `lag_00__T5__is_scoped`: coefficient `0.001565` (raises CT win probability)
- `lag_14__T_place_TSIDELOWER`: coefficient `-0.001537` (lowers CT win probability)
- `lag_00__CT1__is_walking`: coefficient `0.001509` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `43548`, seconds `68.50`, LSTM delta `-0.3253`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.020406`
- `lag_01__T3__flash_duration`: contribution `-0.015501`
- `lag_14__T_place_RAMP`: contribution `-0.014035`
- `lag_05__T5__flash_duration`: contribution `-0.013389`
- `lag_14__T_place_TSIDELOWER`: contribution `-0.011518`

Top utility-only movements:
- `lag_01__T3__flash_duration`: contribution `-0.015501`
- `lag_05__T5__flash_duration`: contribution `-0.013389`
- `lag_02__T3__flash_duration`: contribution `-0.011031`

### tick `39196`, seconds `0.50`, LSTM delta `-0.0665`

Top all feature movements:
- `lag_01__CT_place_UNKNOWN`: contribution `-0.045761`
- `lag_00__CT_he_last_5s`: contribution `-0.010147`
- `lag_00__CT_flashes_last_5s`: contribution `-0.002730`
- `lag_01__T2__is_walking`: contribution `-0.001242`
- `lag_01__T3__molly`: contribution `+0.001230`

Top utility-only movements:
- `lag_00__CT_he_last_5s`: contribution `-0.010147`
- `lag_00__CT_flashes_last_5s`: contribution `-0.002730`
- `lag_01__T3__molly`: contribution `+0.001230`
- `lag_01__T3__smoke`: contribution `+0.001146`
- `lag_01__T3__utility_total`: contribution `+0.000998`

### tick `40828`, seconds `26.00`, LSTM delta `+0.0664`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.016325`
- `lag_00__T1__shots_fired`: contribution `+0.006392`
- `lag_13__CT1__flash_duration`: contribution `+0.006302`
- `lag_04__CT_place_TSIDEUPPER`: contribution `+0.004338`
- `lag_00__T3__duck_amount`: contribution `+0.004090`

Top utility-only movements:
- `lag_13__CT1__flash_duration`: contribution `+0.006302`

### tick `43132`, seconds `62.00`, LSTM delta `+0.0663`

Top all feature movements:
- `lag_00__CT_place_TSIDEUPPER`: contribution `+0.015478`
- `lag_00__CT2__duck_amount`: contribution `+0.005065`
- `lag_15__CT_place_TSIDEUPPER`: contribution `-0.004548`
- `lag_00__CT_place_SIDEENTRANCE`: contribution `+0.004049`
- `lag_01__T_place_TSIDELOWER`: contribution `+0.003320`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `43068`, seconds `61.00`, LSTM delta `-0.0604`

Top all feature movements:
- `lag_00__T5__is_scoped`: contribution `-0.007465`
- `lag_13__CT_place_TSIDEUPPER`: contribution `+0.007292`
- `lag_00__T4__is_walking`: contribution `-0.003963`
- `lag_00__T_place_TSIDELOWER`: contribution `-0.003271`
- `lag_00__T_place_RAMP`: contribution `-0.002887`

Top utility-only movements:
- No utility movement among the top local contributors.
