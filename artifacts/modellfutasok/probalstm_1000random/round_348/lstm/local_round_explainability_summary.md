# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-natus-vincere-vs-tyloo-bo3-u9zlDGjnIy0eSohnO5P-Xx/natus-vincere-vs-tyloo-m3-ancient.csv`
- round_num: `1`

## Largest probability jumps

- tick `4186`, seconds `45.50`, LSTM `0.5126`, delta `+0.3089`
- tick `4122`, seconds `44.50`, LSTM `0.2574`, delta `-0.2416`
- tick `4026`, seconds `43.00`, LSTM `0.5246`, delta `+0.2118`
- tick `4218`, seconds `46.00`, LSTM `0.7107`, delta `+0.1981`
- tick `4346`, seconds `48.00`, LSTM `0.4981`, delta `-0.1866`
- tick `4442`, seconds `49.50`, LSTM `0.6784`, delta `+0.1754`
- tick `3866`, seconds `40.50`, LSTM `0.5036`, delta `-0.1636`
- tick `4506`, seconds `50.50`, LSTM `0.7036`, delta `+0.0783`
- tick `3898`, seconds `41.00`, LSTM `0.4419`, delta `-0.0617`
- tick `4090`, seconds `44.00`, LSTM `0.4989`, delta `-0.0568`

## Top 15 local ridge features

- `lag_08__CT_utility_damage_last_5s`: coefficient `-0.003016`, |coef| `0.003016`
- `lag_07__CT_utility_damage_last_5s`: coefficient `-0.002904`, |coef| `0.002904`
- `lag_08__utility_damage_diff_last_5s`: coefficient `-0.002473`, |coef| `0.002473`
- `lag_07__utility_damage_diff_last_5s`: coefficient `-0.002382`, |coef| `0.002382`
- `lag_05__CT_utility_damage_last_5s`: coefficient `0.002096`, |coef| `0.002096`
- `lag_06__CT_place_MAINHALL`: coefficient `0.001906`, |coef| `0.001906`
- `lag_03__T2__flash_duration`: coefficient `-0.001738`, |coef| `0.001738`
- `lag_05__utility_damage_diff_last_5s`: coefficient `0.001720`, |coef| `0.001720`
- `lag_03__T1__flash_duration`: coefficient `-0.001684`, |coef| `0.001684`
- `lag_00__damage_diff_last_5s`: coefficient `0.001680`, |coef| `0.001680`
- `lag_15__CT_utility_damage_last_5s`: coefficient `-0.001587`, |coef| `0.001587`
- `lag_12__CT_utility_damage_last_5s`: coefficient `0.001579`, |coef| `0.001579`
- `lag_00__kill_diff_last_3s`: coefficient `0.001502`, |coef| `0.001502`
- `lag_12__CT_place_HOUSE`: coefficient `-0.001466`, |coef| `0.001466`
- `lag_03__T_flash_duration_sum`: coefficient `-0.001464`, |coef| `0.001464`

## Top 10 utility ridge features

- `lag_08__CT_utility_damage_last_5s`: coefficient `-0.003016` (lowers CT win probability)
- `lag_07__CT_utility_damage_last_5s`: coefficient `-0.002904` (lowers CT win probability)
- `lag_08__utility_damage_diff_last_5s`: coefficient `-0.002473` (lowers CT win probability)
- `lag_07__utility_damage_diff_last_5s`: coefficient `-0.002382` (lowers CT win probability)
- `lag_05__CT_utility_damage_last_5s`: coefficient `0.002096` (raises CT win probability)
- `lag_03__T2__flash_duration`: coefficient `-0.001738` (lowers CT win probability)
- `lag_05__utility_damage_diff_last_5s`: coefficient `0.001720` (raises CT win probability)
- `lag_03__T1__flash_duration`: coefficient `-0.001684` (lowers CT win probability)
- `lag_15__CT_utility_damage_last_5s`: coefficient `-0.001587` (lowers CT win probability)
- `lag_12__CT_utility_damage_last_5s`: coefficient `0.001579` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_06__CT_place_MAINHALL`: coefficient `0.001906` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001680` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001502` (raises CT win probability)
- `lag_12__CT_place_HOUSE`: coefficient `-0.001466` (lowers CT win probability)
- `lag_14__T_flashed_players`: coefficient `-0.001451` (lowers CT win probability)
- `lag_15__CT_place_HOUSE`: coefficient `0.001419` (raises CT win probability)
- `lag_04__T_kills_last_3s`: coefficient `-0.001316` (lowers CT win probability)
- `lag_11__T_place_MAINHALL`: coefficient `-0.001208` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001120` (raises CT win probability)
- `lag_03__T_flashed_players`: coefficient `-0.001108` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `4186`, seconds `45.50`, LSTM delta `+0.3089`

Top all feature movements:
- `lag_07__CT_utility_damage_last_5s`: contribution `+0.065205`
- `lag_07__utility_damage_diff_last_5s`: contribution `+0.043879`
- `lag_06__CT_place_MAINHALL`: contribution `+0.015775`
- `lag_03__T2__flash_duration`: contribution `+0.012722`
- `lag_03__T1__flash_duration`: contribution `+0.011948`

Top utility-only movements:
- `lag_07__CT_utility_damage_last_5s`: contribution `+0.065205`
- `lag_07__utility_damage_diff_last_5s`: contribution `+0.043879`
- `lag_03__T2__flash_duration`: contribution `+0.012722`
- `lag_03__T1__flash_duration`: contribution `+0.011948`
- `lag_03__T_flash_duration_sum`: contribution `+0.008709`

### tick `4122`, seconds `44.50`, LSTM delta `-0.2416`

Top all feature movements:
- `lag_05__CT_utility_damage_last_5s`: contribution `-0.047065`
- `lag_05__utility_damage_diff_last_5s`: contribution `-0.031685`
- `lag_14__T_flashed_players`: contribution `-0.011201`
- `lag_01__T2__flash_duration`: contribution `-0.008383`
- `lag_01__T1__flash_duration`: contribution `-0.007873`

Top utility-only movements:
- `lag_05__CT_utility_damage_last_5s`: contribution `-0.047065`
- `lag_05__utility_damage_diff_last_5s`: contribution `-0.031685`
- `lag_01__T2__flash_duration`: contribution `-0.008383`
- `lag_01__T1__flash_duration`: contribution `-0.007873`
- `lag_14__T_flash_duration_sum`: contribution `-0.006475`

### tick `4026`, seconds `43.00`, LSTM delta `+0.2118`

Top all feature movements:
- `lag_13__CT_utility_damage_last_5s`: contribution `+0.026177`
- `lag_13__utility_damage_diff_last_5s`: contribution `+0.017621`
- `lag_02__CT_utility_damage_last_5s`: contribution `+0.013316`
- `lag_02__utility_damage_diff_last_5s`: contribution `+0.008986`
- `lag_11__T_flashed_players`: contribution `+0.006531`

Top utility-only movements:
- `lag_13__CT_utility_damage_last_5s`: contribution `+0.026177`
- `lag_13__utility_damage_diff_last_5s`: contribution `+0.017621`
- `lag_02__CT_utility_damage_last_5s`: contribution `+0.013316`
- `lag_02__utility_damage_diff_last_5s`: contribution `+0.008986`
- `lag_11__T_flash_duration_sum`: contribution `+0.002995`

### tick `4218`, seconds `46.00`, LSTM delta `+0.1981`

Top all feature movements:
- `lag_08__CT_utility_damage_last_5s`: contribution `+0.067726`
- `lag_08__utility_damage_diff_last_5s`: contribution `+0.045553`
- `lag_07__CT_place_MAINHALL`: contribution `+0.008104`
- `lag_11__T_place_MAINHALL`: contribution `+0.004362`
- `lag_00__damage_diff_last_5s`: contribution `+0.003789`

Top utility-only movements:
- `lag_08__CT_utility_damage_last_5s`: contribution `+0.067726`
- `lag_08__utility_damage_diff_last_5s`: contribution `+0.045553`
- `lag_04__T2__flash_duration`: contribution `+0.002988`
- `lag_04__T1__flash_duration`: contribution `+0.002807`
- `lag_13__T_A_site_active_infernos`: contribution `+0.001941`

### tick `4346`, seconds `48.00`, LSTM delta `-0.1866`

Top all feature movements:
- `lag_12__CT_utility_damage_last_5s`: contribution `-0.035462`
- `lag_12__utility_damage_diff_last_5s`: contribution `-0.023893`
- `lag_11__CT_place_MAINHALL`: contribution `-0.007145`
- `lag_12__CT_place_HOUSE`: contribution `-0.005180`
- `lag_00__CT_place_MAINHALL`: contribution `-0.004828`

Top utility-only movements:
- `lag_12__CT_utility_damage_last_5s`: contribution `-0.035462`
- `lag_12__utility_damage_diff_last_5s`: contribution `-0.023893`
- `lag_08__T2__flash_duration`: contribution `-0.003829`
- `lag_08__T1__flash_duration`: contribution `-0.003596`
- `lag_08__T_flash_duration_sum`: contribution `-0.001785`
