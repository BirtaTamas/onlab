# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-spirit-vs-saw-bo3-_1uD70D_aUzOV8qHt5kBr9/spirit-vs-saw-m1-dust2.csv`
- round_num: `10`

## Largest probability jumps

- tick `92553`, seconds `103.50`, LSTM `0.3242`, delta `-0.2420`
- tick `92233`, seconds `98.50`, LSTM `0.4410`, delta `+0.2291`
- tick `90857`, seconds `77.00`, LSTM `0.1917`, delta `-0.1676`
- tick `90793`, seconds `76.00`, LSTM `0.3477`, delta `+0.1292`
- tick `92585`, seconds `104.00`, LSTM `0.2551`, delta `-0.0690`
- tick `91465`, seconds `86.50`, LSTM `0.0561`, delta `-0.0684`
- tick `91625`, seconds `89.00`, LSTM `0.0798`, delta `+0.0676`
- tick `92393`, seconds `101.00`, LSTM `0.5747`, delta `+0.0650`
- tick `90473`, seconds `71.00`, LSTM `0.2781`, delta `-0.0564`
- tick `85961`, seconds `0.50`, LSTM `0.2413`, delta `-0.0506`

## Top 15 local ridge features

- `lag_00__CT_place_EXTENDEDA`: coefficient `0.002962`, |coef| `0.002962`
- `lag_00__kill_diff_last_3s`: coefficient `0.002865`, |coef| `0.002865`
- `lag_00__T_place_ARAMP`: coefficient `-0.002839`, |coef| `0.002839`
- `lag_03__T_place_ARAMP`: coefficient `-0.002716`, |coef| `0.002716`
- `lag_00__damage_diff_last_5s`: coefficient `0.002604`, |coef| `0.002604`
- `lag_13__T_place_ARAMP`: coefficient `0.002565`, |coef| `0.002565`
- `lag_02__T_place_ARAMP`: coefficient `-0.002444`, |coef| `0.002444`
- `lag_14__T_duck_amount_mean`: coefficient `-0.002357`, |coef| `0.002357`
- `lag_00__T_kills_last_3s`: coefficient `-0.002272`, |coef| `0.002272`
- `lag_01__damage_diff_last_5s`: coefficient `0.002147`, |coef| `0.002147`
- `lag_05__T_place_ARAMP`: coefficient `-0.002145`, |coef| `0.002145`
- `lag_04__T_place_ARAMP`: coefficient `-0.002049`, |coef| `0.002049`
- `lag_15__CT_place_EXTENDEDA`: coefficient `0.001900`, |coef| `0.001900`
- `lag_13__T4__duck_amount`: coefficient `-0.001871`, |coef| `0.001871`
- `lag_13__T_duck_amount_mean`: coefficient `-0.001843`, |coef| `0.001843`

## Top 10 utility ridge features

- `lag_00__T3__flash`: coefficient `-0.001488` (lowers CT win probability)
- `lag_02__CT1__flash_duration`: coefficient `-0.001360` (lowers CT win probability)
- `lag_10__T3__flash`: coefficient `0.001290` (raises CT win probability)
- `lag_12__T_A_site_active_infernos`: coefficient `-0.001231` (lowers CT win probability)
- `lag_00__T_flashes_last_5s`: coefficient `-0.000993` (lowers CT win probability)
- `lag_00__CT4__smoke`: coefficient `0.000987` (raises CT win probability)
- `lag_10__CT4__smoke`: coefficient `0.000973` (raises CT win probability)
- `lag_01__T3__flash`: coefficient `-0.000942` (lowers CT win probability)
- `lag_06__CT1__flash`: coefficient `0.000878` (raises CT win probability)
- `lag_00__T3__utility_total`: coefficient `-0.000827` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_EXTENDEDA`: coefficient `0.002962` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002865` (raises CT win probability)
- `lag_00__T_place_ARAMP`: coefficient `-0.002839` (lowers CT win probability)
- `lag_03__T_place_ARAMP`: coefficient `-0.002716` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002604` (raises CT win probability)
- `lag_13__T_place_ARAMP`: coefficient `0.002565` (raises CT win probability)
- `lag_02__T_place_ARAMP`: coefficient `-0.002444` (lowers CT win probability)
- `lag_14__T_duck_amount_mean`: coefficient `-0.002357` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002272` (lowers CT win probability)
- `lag_01__damage_diff_last_5s`: coefficient `0.002147` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `92553`, seconds `103.50`, LSTM delta `-0.2420`

Top all feature movements:
- `lag_13__T_place_ARAMP`: contribution `-0.023205`
- `lag_00__CT_place_EXTENDEDA`: contribution `-0.016626`
- `lag_10__T_place_ARAMP`: contribution `-0.013896`
- `lag_11__T_place_ARAMP`: contribution `-0.010383`
- `lag_02__T_duck_amount_mean`: contribution `-0.010248`

Top utility-only movements:
- `lag_10__T3__flash`: contribution `-0.003801`

### tick `92233`, seconds `98.50`, LSTM delta `+0.2291`

Top all feature movements:
- `lag_00__T_place_ARAMP`: contribution `+0.025690`
- `lag_03__T_place_ARAMP`: contribution `+0.024570`
- `lag_13__CT_place_SHORTSTAIRS`: contribution `+0.009036`
- `lag_13__CT_place_EXTENDEDA`: contribution `+0.007464`
- `lag_00__kill_diff_last_3s`: contribution `+0.006896`

Top utility-only movements:
- `lag_00__T3__flash`: contribution `+0.004387`
- `lag_12__T_A_site_active_infernos`: contribution `+0.003665`

### tick `90857`, seconds `77.00`, LSTM delta `-0.1676`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.007199`
- `lag_02__CT_shots_fired_sum`: contribution `-0.007069`
- `lag_13__T_flashed_players`: contribution `-0.007053`
- `lag_13__T4__duck_amount`: contribution `-0.006920`
- `lag_00__kill_diff_last_3s`: contribution `-0.006896`

Top utility-only movements:
- `lag_13__CT5__flash_duration`: contribution `-0.004112`
- `lag_13__CT_flash_duration_sum`: contribution `-0.003413`
- `lag_13__CT2__flash_duration`: contribution `-0.003295`
- `lag_13__T_flash_duration_sum`: contribution `-0.003188`

### tick `90793`, seconds `76.00`, LSTM delta `+0.1292`

Top all feature movements:
- `lag_11__T_flashed_players`: contribution `+0.007172`
- `lag_00__kill_diff_last_3s`: contribution `+0.006896`
- `lag_13__T_place_EXTENDEDA`: contribution `-0.006662`
- `lag_11__CT5__flash_duration`: contribution `+0.005388`
- `lag_01__CT_place_EXTENDEDA`: contribution `-0.004808`

Top utility-only movements:
- `lag_11__CT5__flash_duration`: contribution `+0.005388`
- `lag_11__CT2__flash_duration`: contribution `+0.004409`
- `lag_01__T1__flash_duration`: contribution `+0.004387`
- `lag_11__CT_flash_duration_sum`: contribution `+0.003890`

### tick `92585`, seconds `104.00`, LSTM delta `-0.0690`

Top all feature movements:
- `lag_14__T_place_ARAMP`: contribution `-0.013320`
- `lag_11__T_place_ARAMP`: contribution `+0.010383`
- `lag_12__T_place_ARAMP`: contribution `+0.007919`
- `lag_03__T_duck_amount_mean`: contribution `-0.006694`
- `lag_01__damage_diff_last_5s`: contribution `-0.006249`

Top utility-only movements:
- No utility movement among the top local contributors.
