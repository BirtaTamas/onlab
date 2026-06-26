# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-heroic-vs-3dmax-bo3-OVT4ch_FfOW2E26liKqT_k/heroic-vs-3dmax-m2-inferno.csv`
- round_num: `15`

## Largest probability jumps

- tick `108929`, seconds `25.00`, LSTM `0.7228`, delta `+0.2853`
- tick `108961`, seconds `25.50`, LSTM `0.4941`, delta `-0.2287`
- tick `109249`, seconds `30.00`, LSTM `0.3584`, delta `-0.0701`
- tick `109345`, seconds `31.50`, LSTM `0.2375`, delta `-0.0667`
- tick `108865`, seconds `24.00`, LSTM `0.4260`, delta `+0.0632`
- tick `107361`, seconds `0.50`, LSTM `0.2033`, delta `-0.0556`
- tick `108801`, seconds `23.00`, LSTM `0.3870`, delta `-0.0487`
- tick `110977`, seconds `57.00`, LSTM `0.2543`, delta `+0.0483`
- tick `107969`, seconds `10.00`, LSTM `0.2652`, delta `+0.0459`
- tick `111041`, seconds `58.00`, LSTM `0.1935`, delta `-0.0411`

## Top 15 local ridge features

- `lag_04__T_flashed_players`: coefficient `0.002681`, |coef| `0.002681`
- `lag_04__CT4__flash_duration`: coefficient `0.002272`, |coef| `0.002272`
- `lag_01__T_place_TOPOFMID`: coefficient `0.001684`, |coef| `0.001684`
- `lag_03__CT4__flash_duration`: coefficient `0.001680`, |coef| `0.001680`
- `lag_04__CT_flash_duration_sum`: coefficient `0.001375`, |coef| `0.001375`
- `lag_00__CT_place_LOWERMID`: coefficient `-0.001281`, |coef| `0.001281`
- `lag_04__CT_flashed_players`: coefficient `0.001280`, |coef| `0.001280`
- `lag_00__T_place_PIT`: coefficient `-0.001241`, |coef| `0.001241`
- `lag_02__CT4__flash_duration`: coefficient `-0.001198`, |coef| `0.001198`
- `lag_07__CT5__is_walking`: coefficient `0.001187`, |coef| `0.001187`
- `lag_03__T_shots_fired_sum`: coefficient `0.001168`, |coef| `0.001168`
- `lag_01__CT_place_LOWERMID`: coefficient `-0.001134`, |coef| `0.001134`
- `lag_00__CT_kills_last_3s`: coefficient `0.001083`, |coef| `0.001083`
- `lag_09__T2__is_walking`: coefficient `0.001083`, |coef| `0.001083`
- `lag_00__CT4__flash_duration`: coefficient `-0.001071`, |coef| `0.001071`

## Top 10 utility ridge features

- `lag_04__CT4__flash_duration`: coefficient `0.002272` (raises CT win probability)
- `lag_03__CT4__flash_duration`: coefficient `0.001680` (raises CT win probability)
- `lag_04__CT_flash_duration_sum`: coefficient `0.001375` (raises CT win probability)
- `lag_02__CT4__flash_duration`: coefficient `-0.001198` (lowers CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `-0.001071` (lowers CT win probability)
- `lag_04__T_flashes_last_5s`: coefficient `-0.000939` (lowers CT win probability)
- `lag_03__CT_flash_duration_sum`: coefficient `0.000920` (raises CT win probability)
- `lag_13__T_flashes_last_5s`: coefficient `-0.000908` (lowers CT win probability)
- `lag_06__T_flashes_last_5s`: coefficient `-0.000837` (lowers CT win probability)
- `lag_04__CT3__flash_duration`: coefficient `0.000799` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_04__T_flashed_players`: coefficient `0.002681` (raises CT win probability)
- `lag_01__T_place_TOPOFMID`: coefficient `0.001684` (raises CT win probability)
- `lag_00__CT_place_LOWERMID`: coefficient `-0.001281` (lowers CT win probability)
- `lag_04__CT_flashed_players`: coefficient `0.001280` (raises CT win probability)
- `lag_00__T_place_PIT`: coefficient `-0.001241` (lowers CT win probability)
- `lag_07__CT5__is_walking`: coefficient `0.001187` (raises CT win probability)
- `lag_03__T_shots_fired_sum`: coefficient `0.001168` (raises CT win probability)
- `lag_01__CT_place_LOWERMID`: coefficient `-0.001134` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001083` (raises CT win probability)
- `lag_09__T2__is_walking`: coefficient `0.001083` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `108929`, seconds `25.00`, LSTM delta `+0.2853`

Top all feature movements:
- `lag_04__T_flashed_players`: contribution `+0.020697`
- `lag_04__CT4__flash_duration`: contribution `+0.015874`
- `lag_02__CT4__flash_duration`: contribution `+0.008370`
- `lag_01__T_place_TOPOFMID`: contribution `+0.006858`
- `lag_04__CT_flash_duration_sum`: contribution `+0.006196`

Top utility-only movements:
- `lag_04__CT4__flash_duration`: contribution `+0.015874`
- `lag_02__CT4__flash_duration`: contribution `+0.008370`
- `lag_04__CT_flash_duration_sum`: contribution `+0.006196`
- `lag_04__CT3__flash_duration`: contribution `+0.002470`
- `lag_02__T3__flash`: contribution `+0.002328`

### tick `108961`, seconds `25.50`, LSTM delta `-0.2287`

Top all feature movements:
- `lag_04__T_flashed_players`: contribution `-0.015522`
- `lag_03__CT4__flash_duration`: contribution `-0.011738`
- `lag_05__T_flashed_players`: contribution `-0.006163`
- `lag_05__CT4__flash_duration`: contribution `-0.004984`
- `lag_01__T_place_TOPOFMID`: contribution `-0.003429`

Top utility-only movements:
- `lag_03__CT4__flash_duration`: contribution `-0.011738`
- `lag_05__CT4__flash_duration`: contribution `-0.004984`
- `lag_03__CT_flash_duration_sum`: contribution `-0.002873`
- `lag_05__CT_flash_duration_sum`: contribution `-0.002354`

### tick `109249`, seconds `30.00`, LSTM delta `-0.0701`

Top all feature movements:
- `lag_02__CT_damage_last_5s`: contribution `-0.003371`
- `lag_09__CT_place_TOPOFMID`: contribution `-0.003113`
- `lag_12__CT4__flash_duration`: contribution `-0.002787`
- `lag_12__CT_place_TOPOFMID`: contribution `-0.002627`
- `lag_00__CT_damage_last_5s`: contribution `-0.002595`

Top utility-only movements:
- `lag_12__CT4__flash_duration`: contribution `-0.002787`

### tick `109345`, seconds `31.50`, LSTM delta `-0.0667`

Top all feature movements:
- `lag_02__T_place_PIT`: contribution `-0.006446`
- `lag_12__CT_place_TOPOFMID`: contribution `-0.002627`
- `lag_01__CT_shots_fired_sum`: contribution `-0.002390`
- `lag_15__CT_place_TOPOFMID`: contribution `-0.002269`
- `lag_15__CT4__flash_duration`: contribution `-0.001811`

Top utility-only movements:
- `lag_15__CT4__flash_duration`: contribution `-0.001811`

### tick `108865`, seconds `24.00`, LSTM delta `+0.0632`

Top all feature movements:
- `lag_02__CT4__flash_duration`: contribution `-0.008370`
- `lag_00__CT4__flash_duration`: contribution `+0.007485`
- `lag_02__T_flashed_players`: contribution `-0.004003`
- `lag_00__CT_damage_last_5s`: contribution `+0.003624`
- `lag_01__T_place_TOPOFMID`: contribution `+0.003429`

Top utility-only movements:
- `lag_02__CT4__flash_duration`: contribution `-0.008370`
- `lag_00__CT4__flash_duration`: contribution `+0.007485`
- `lag_02__CT_flash_duration_sum`: contribution `-0.001553`
- `lag_02__CT3__flash_duration`: contribution `+0.001300`
