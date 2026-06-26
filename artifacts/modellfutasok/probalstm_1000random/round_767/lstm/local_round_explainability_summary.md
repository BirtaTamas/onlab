# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-mouz-bo3-Ko5VJMvyF1OsCx2TbVU9pb/vitality-vs-mouz-m1-inferno.csv`
- round_num: `13`

## Largest probability jumps

- tick `107196`, seconds `53.50`, LSTM `0.6413`, delta `+0.2245`
- tick `107228`, seconds `54.00`, LSTM `0.8172`, delta `+0.1759`
- tick `107292`, seconds `55.00`, LSTM `0.8877`, delta `+0.0743`
- tick `107132`, seconds `52.50`, LSTM `0.4173`, delta `-0.0539`
- tick `107388`, seconds `56.50`, LSTM `0.9558`, delta `+0.0527`
- tick `107068`, seconds `51.50`, LSTM `0.4869`, delta `-0.0376`
- tick `105308`, seconds `24.00`, LSTM `0.4801`, delta `-0.0185`
- tick `107100`, seconds `52.00`, LSTM `0.4712`, delta `-0.0157`
- tick `107420`, seconds `57.00`, LSTM `0.9684`, delta `+0.0126`
- tick `106876`, seconds `48.50`, LSTM `0.5114`, delta `+0.0122`

## Top 15 local ridge features

- `lag_07__CT3__flash_duration`: coefficient `0.001646`, |coef| `0.001646`
- `lag_02__CT_place_BALCONY`: coefficient `-0.001645`, |coef| `0.001645`
- `lag_07__T_flashed_players`: coefficient `0.001594`, |coef| `0.001594`
- `lag_00__T5__flash_duration`: coefficient `-0.001543`, |coef| `0.001543`
- `lag_02__T4__flash_duration`: coefficient `0.001540`, |coef| `0.001540`
- `lag_02__T_flash_duration_sum`: coefficient `0.001492`, |coef| `0.001492`
- `lag_00__T_flash_duration_sum`: coefficient `-0.001464`, |coef| `0.001464`
- `lag_04__CT1__flash_duration`: coefficient `0.001395`, |coef| `0.001395`
- `lag_00__CT_kills_last_3s`: coefficient `0.001364`, |coef| `0.001364`
- `lag_00__T1__flash_duration`: coefficient `-0.001356`, |coef| `0.001356`
- `lag_08__CT3__flash_duration`: coefficient `0.001270`, |coef| `0.001270`
- `lag_03__T_flash_duration_sum`: coefficient `0.001241`, |coef| `0.001241`
- `lag_00__damage_diff_last_5s`: coefficient `0.001225`, |coef| `0.001225`
- `lag_02__T3__flash_duration`: coefficient `0.001208`, |coef| `0.001208`
- `lag_00__CT_damage_last_5s`: coefficient `0.001191`, |coef| `0.001191`

## Top 10 utility ridge features

- `lag_07__CT3__flash_duration`: coefficient `0.001646` (raises CT win probability)
- `lag_00__T5__flash_duration`: coefficient `-0.001543` (lowers CT win probability)
- `lag_02__T4__flash_duration`: coefficient `0.001540` (raises CT win probability)
- `lag_02__T_flash_duration_sum`: coefficient `0.001492` (raises CT win probability)
- `lag_00__T_flash_duration_sum`: coefficient `-0.001464` (lowers CT win probability)
- `lag_04__CT1__flash_duration`: coefficient `0.001395` (raises CT win probability)
- `lag_00__T1__flash_duration`: coefficient `-0.001356` (lowers CT win probability)
- `lag_08__CT3__flash_duration`: coefficient `0.001270` (raises CT win probability)
- `lag_03__T_flash_duration_sum`: coefficient `0.001241` (raises CT win probability)
- `lag_02__T3__flash_duration`: coefficient `0.001208` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_02__CT_place_BALCONY`: coefficient `-0.001645` (lowers CT win probability)
- `lag_07__T_flashed_players`: coefficient `0.001594` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001364` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001225` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001191` (raises CT win probability)
- `lag_04__CT_place_BALCONY`: coefficient `0.001163` (raises CT win probability)
- `lag_08__T_flashed_players`: coefficient `0.001140` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001137` (raises CT win probability)
- `lag_07__CT_flashed_players`: coefficient `0.001033` (raises CT win probability)
- `lag_06__T_flashed_players`: coefficient `-0.000881` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `107196`, seconds `53.50`, LSTM delta `+0.2245`

Top all feature movements:
- `lag_02__T_flash_duration_sum`: contribution `+0.012445`
- `lag_07__T_flashed_players`: contribution `+0.012306`
- `lag_07__CT3__flash_duration`: contribution `+0.011449`
- `lag_02__CT_place_BALCONY`: contribution `+0.010560`
- `lag_02__T4__flash_duration`: contribution `+0.010033`

Top utility-only movements:
- `lag_02__T_flash_duration_sum`: contribution `+0.012445`
- `lag_07__CT3__flash_duration`: contribution `+0.011449`
- `lag_02__T4__flash_duration`: contribution `+0.010033`
- `lag_04__CT1__flash_duration`: contribution `+0.008227`
- `lag_02__T3__flash_duration`: contribution `+0.007323`

### tick `107228`, seconds `54.00`, LSTM delta `+0.1759`

Top all feature movements:
- `lag_00__T5__flash_duration`: contribution `+0.011485`
- `lag_03__T_flash_duration_sum`: contribution `+0.010354`
- `lag_08__CT3__flash_duration`: contribution `+0.008837`
- `lag_08__T_flashed_players`: contribution `+0.008802`
- `lag_03__T4__flash_duration`: contribution `+0.007741`

Top utility-only movements:
- `lag_00__T5__flash_duration`: contribution `+0.011485`
- `lag_03__T_flash_duration_sum`: contribution `+0.010354`
- `lag_08__CT3__flash_duration`: contribution `+0.008837`
- `lag_03__T4__flash_duration`: contribution `+0.007741`
- `lag_05__CT1__flash_duration`: contribution `+0.006345`

### tick `107292`, seconds `55.00`, LSTM delta `+0.0743`

Top all feature movements:
- `lag_07__T_flashed_players`: contribution `+0.009229`
- `lag_08__T_flashed_players`: contribution `-0.006601`
- `lag_05__T_flash_duration_sum`: contribution `+0.004695`
- `lag_05__CT_place_BALCONY`: contribution `-0.004527`
- `lag_02__T_flash_duration_sum`: contribution `-0.004510`

Top utility-only movements:
- `lag_05__T_flash_duration_sum`: contribution `+0.004695`
- `lag_02__T_flash_duration_sum`: contribution `-0.004510`
- `lag_00__T_flash_duration_sum`: contribution `+0.004413`
- `lag_00__T3__flash_duration`: contribution `+0.004390`
- `lag_10__CT3__flash_duration`: contribution `+0.003500`

### tick `107132`, seconds `52.50`, LSTM delta `-0.0539`

Top all feature movements:
- `lag_00__T_flash_duration_sum`: contribution `-0.012212`
- `lag_02__CT_place_BALCONY`: contribution `-0.010560`
- `lag_00__T1__flash_duration`: contribution `-0.007234`
- `lag_05__T_flashed_players`: contribution `-0.004221`
- `lag_02__T_flash_duration_sum`: contribution `+0.004136`

Top utility-only movements:
- `lag_00__T_flash_duration_sum`: contribution `-0.012212`
- `lag_00__T1__flash_duration`: contribution `-0.007234`
- `lag_02__T_flash_duration_sum`: contribution `+0.004136`
- `lag_00__T3__flash_duration`: contribution `-0.003679`
- `lag_00__T5__flash_duration`: contribution `-0.003599`

### tick `107388`, seconds `56.50`, LSTM delta `+0.0527`

Top all feature movements:
- `lag_03__T3__flash_duration`: contribution `-0.006138`
- `lag_08__T_flash_duration_sum`: contribution `+0.004820`
- `lag_08__T_flashed_players`: contribution `+0.004401`
- `lag_05__T5__flash_duration`: contribution `-0.003757`
- `lag_03__T_flash_duration_sum`: contribution `-0.003742`

Top utility-only movements:
- `lag_03__T3__flash_duration`: contribution `-0.006138`
- `lag_08__T_flash_duration_sum`: contribution `+0.004820`
- `lag_05__T5__flash_duration`: contribution `-0.003757`
- `lag_03__T_flash_duration_sum`: contribution `-0.003742`
- `lag_08__T3__flash_duration`: contribution `+0.002428`
