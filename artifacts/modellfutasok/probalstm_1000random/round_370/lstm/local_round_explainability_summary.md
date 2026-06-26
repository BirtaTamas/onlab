# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-spirit-vs-saw-bo3-_1uD70D_aUzOV8qHt5kBr9/spirit-vs-saw-m1-dust2.csv`
- round_num: `13`

## Largest probability jumps

- tick `126442`, seconds `51.00`, LSTM `0.5849`, delta `+0.2454`
- tick `126506`, seconds `52.00`, LSTM `0.8320`, delta `+0.2033`
- tick `126346`, seconds `49.50`, LSTM `0.3367`, delta `-0.1061`
- tick `126986`, seconds `59.50`, LSTM `0.9404`, delta `+0.0864`
- tick `126730`, seconds `55.50`, LSTM `0.8912`, delta `-0.0456`
- tick `126474`, seconds `51.50`, LSTM `0.6287`, delta `+0.0438`
- tick `126634`, seconds `54.00`, LSTM `0.9094`, delta `+0.0361`
- tick `126538`, seconds `52.50`, LSTM `0.8631`, delta `+0.0311`
- tick `126826`, seconds `57.00`, LSTM `0.8466`, delta `-0.0306`
- tick `126186`, seconds `47.00`, LSTM `0.4654`, delta `-0.0291`

## Top 15 local ridge features

- `lag_00__T_place_EXTENDEDA`: coefficient `-0.004956`, |coef| `0.004956`
- `lag_00__T_flashed_players`: coefficient `-0.003451`, |coef| `0.003451`
- `lag_03__T_place_EXTENDEDA`: coefficient `0.002429`, |coef| `0.002429`
- `lag_08__T_place_EXTENDEDA`: coefficient `0.002320`, |coef| `0.002320`
- `lag_03__T_flashed_players`: coefficient `0.002318`, |coef| `0.002318`
- `lag_06__T_place_EXTENDEDA`: coefficient `0.002138`, |coef| `0.002138`
- `lag_05__T_place_EXTENDEDA`: coefficient `0.001921`, |coef| `0.001921`
- `lag_05__T_flashed_players`: coefficient `0.001879`, |coef| `0.001879`
- `lag_00__CT_kills_last_3s`: coefficient `0.001767`, |coef| `0.001767`
- `lag_00__damage_diff_last_5s`: coefficient `0.001550`, |coef| `0.001550`
- `lag_00__CT_damage_last_5s`: coefficient `0.001514`, |coef| `0.001514`
- `lag_00__kill_diff_last_3s`: coefficient `0.001504`, |coef| `0.001504`
- `lag_01__T_place_EXTENDEDA`: coefficient `-0.001488`, |coef| `0.001488`
- `lag_00__T_place_SHORTSTAIRS`: coefficient `0.001360`, |coef| `0.001360`
- `lag_03__T2__flash_duration`: coefficient `0.001353`, |coef| `0.001353`

## Top 10 utility ridge features

- `lag_03__T2__flash_duration`: coefficient `0.001353` (raises CT win probability)
- `lag_00__T_flash_duration_sum`: coefficient `-0.001149` (lowers CT win probability)
- `lag_01__T_A_site_active_infernos`: coefficient `-0.001147` (lowers CT win probability)
- `lag_05__T2__flash_duration`: coefficient `0.001057` (raises CT win probability)
- `lag_07__T2__molly`: coefficient `-0.000861` (lowers CT win probability)
- `lag_00__T_A_site_active_infernos`: coefficient `-0.000858` (lowers CT win probability)
- `lag_01__CT1__smoke`: coefficient `-0.000846` (lowers CT win probability)
- `lag_01__T_active_infernos`: coefficient `-0.000818` (lowers CT win probability)
- `lag_00__T4__flash_duration`: coefficient `-0.000811` (lowers CT win probability)
- `lag_04__T2__flash_duration`: coefficient `0.000754` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_EXTENDEDA`: coefficient `-0.004956` (lowers CT win probability)
- `lag_00__T_flashed_players`: coefficient `-0.003451` (lowers CT win probability)
- `lag_03__T_place_EXTENDEDA`: coefficient `0.002429` (raises CT win probability)
- `lag_08__T_place_EXTENDEDA`: coefficient `0.002320` (raises CT win probability)
- `lag_03__T_flashed_players`: coefficient `0.002318` (raises CT win probability)
- `lag_06__T_place_EXTENDEDA`: coefficient `0.002138` (raises CT win probability)
- `lag_05__T_place_EXTENDEDA`: coefficient `0.001921` (raises CT win probability)
- `lag_05__T_flashed_players`: coefficient `0.001879` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001767` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001550` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `126442`, seconds `51.00`, LSTM delta `+0.2454`

Top all feature movements:
- `lag_00__T_place_EXTENDEDA`: contribution `+0.024569`
- `lag_03__T_place_EXTENDEDA`: contribution `+0.024083`
- `lag_03__T_flashed_players`: contribution `+0.017891`
- `lag_00__T_flashed_players`: contribution `+0.013318`
- `lag_08__T_place_EXTENDEDA`: contribution `+0.011504`

Top utility-only movements:
- `lag_03__T2__flash_duration`: contribution `+0.003562`
- `lag_01__T_A_site_active_infernos`: contribution `+0.003413`

### tick `126506`, seconds `52.00`, LSTM delta `+0.2033`

Top all feature movements:
- `lag_00__T_place_EXTENDEDA`: contribution `+0.024569`
- `lag_05__T_place_EXTENDEDA`: contribution `+0.019052`
- `lag_05__T_flashed_players`: contribution `+0.014501`
- `lag_08__T_place_EXTENDEDA`: contribution `+0.011504`
- `lag_00__T_flashed_players`: contribution `+0.006659`

Top utility-only movements:
- `lag_05__T2__flash_duration`: contribution `+0.002783`

### tick `126346`, seconds `49.50`, LSTM delta `-0.1061`

Top all feature movements:
- `lag_00__T_place_EXTENDEDA`: contribution `-0.049137`
- `lag_00__T_flashed_players`: contribution `-0.026636`
- `lag_03__T_place_EXTENDEDA`: contribution `+0.012042`
- `lag_00__T_place_SHORTSTAIRS`: contribution `-0.011433`
- `lag_05__T_place_EXTENDEDA`: contribution `+0.009526`

Top utility-only movements:
- `lag_00__T_flash_duration_sum`: contribution `-0.002670`
- `lag_00__T_A_site_active_infernos`: contribution `-0.002553`
- `lag_00__T2__flash_duration`: contribution `-0.001876`

### tick `126986`, seconds `59.50`, LSTM delta `+0.0864`

Top all feature movements:
- `lag_00__T_place_EXTENDEDA`: contribution `+0.024569`
- `lag_00__T_flashed_players`: contribution `+0.006659`
- `lag_00__CT_kills_last_3s`: contribution `+0.005101`
- `lag_08__CT4__flash_duration`: contribution `+0.004744`
- `lag_00__kill_diff_last_3s`: contribution `+0.003619`

Top utility-only movements:
- `lag_08__CT4__flash_duration`: contribution `+0.004744`
- `lag_00__T4__flash_duration`: contribution `+0.003148`
- `lag_08__T2__flash_duration`: contribution `+0.002779`
- `lag_00__T_flash_duration_sum`: contribution `+0.001814`
- `lag_15__T2__flash_duration`: contribution `-0.000943`

### tick `126730`, seconds `55.50`, LSTM delta `-0.0456`

Top all feature movements:
- `lag_03__T_place_EXTENDEDA`: contribution `-0.012042`
- `lag_00__T_flashed_players`: contribution `-0.006659`
- `lag_00__T4__flash_duration`: contribution `-0.004937`
- `lag_07__T_place_EXTENDEDA`: contribution `-0.004375`
- `lag_03__CT_place_HOLE`: contribution `-0.004209`

Top utility-only movements:
- `lag_00__T4__flash_duration`: contribution `-0.004937`
- `lag_00__T2__flash_duration`: contribution `-0.003878`
- `lag_00__T_flash_duration_sum`: contribution `-0.002845`
- `lag_00__CT4__flash_duration`: contribution `+0.002804`
