# Local Round Explainability

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-tyloo-vs-rare-atom-bo3-8GB1HWZtKOlh9_707n2A62/tyloo-vs-rare-atom-m2-inferno.csv`
- round_num: `14`

## Largest probability jumps

- tick `122181`, seconds `64.00`, LSTM `0.1277`, delta `-0.2907`
- tick `122341`, seconds `66.50`, LSTM `0.0252`, delta `-0.1876`
- tick `122021`, seconds `61.50`, LSTM `0.7163`, delta `+0.1773`
- tick `123205`, seconds `80.00`, LSTM `0.4882`, delta `+0.1752`
- tick `122085`, seconds `62.50`, LSTM `0.5123`, delta `-0.1416`
- tick `122117`, seconds `63.00`, LSTM `0.3848`, delta `-0.1274`
- tick `123173`, seconds `79.50`, LSTM `0.3130`, delta `+0.1263`
- tick `122245`, seconds `65.00`, LSTM `0.1622`, delta `+0.0796`
- tick `122053`, seconds `62.00`, LSTM `0.6538`, delta `-0.0625`
- tick `122565`, seconds `70.00`, LSTM `0.0594`, delta `+0.0461`

## Top 15 local ridge features

- `lag_00__T_place_PIT`: coefficient `-0.002993`, |coef| `0.002993`
- `lag_00__T_bomb_zone_count`: coefficient `-0.002806`, |coef| `0.002806`
- `lag_10__T_place_QUAD`: coefficient `-0.002269`, |coef| `0.002269`
- `lag_13__CT_place_LIBRARY`: coefficient `-0.002078`, |coef| `0.002078`
- `lag_01__T_place_PIT`: coefficient `-0.002027`, |coef| `0.002027`
- `lag_00__kill_diff_last_3s`: coefficient `0.001950`, |coef| `0.001950`
- `lag_05__T_duck_amount_mean`: coefficient `0.001834`, |coef| `0.001834`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001832`, |coef| `0.001832`
- `lag_14__CT_place_LIBRARY`: coefficient `-0.001751`, |coef| `0.001751`
- `lag_15__T_bomb_zone_count`: coefficient `0.001688`, |coef| `0.001688`
- `lag_00__CT_kills_last_3s`: coefficient `0.001558`, |coef| `0.001558`
- `lag_00__T3__has_bomb`: coefficient `-0.001543`, |coef| `0.001543`
- `lag_04__T_place_BALCONY`: coefficient `-0.001438`, |coef| `0.001438`
- `lag_08__T_place_QUAD`: coefficient `-0.001436`, |coef| `0.001436`
- `lag_14__T3__duck_amount`: coefficient `0.001420`, |coef| `0.001420`

## Top 10 utility ridge features

- `lag_15__CT3__smoke`: coefficient `0.001215` (raises CT win probability)
- `lag_07__T5__smoke`: coefficient `-0.001145` (lowers CT win probability)
- `lag_08__T5__smoke`: coefficient `-0.001054` (lowers CT win probability)
- `lag_05__CT3__smoke`: coefficient `-0.001015` (lowers CT win probability)
- `lag_00__CT_active_smokes`: coefficient `-0.001009` (lowers CT win probability)
- `lag_00__active_smokes_total`: coefficient `-0.000959` (lowers CT win probability)
- `lag_06__CT3__smoke`: coefficient `-0.000956` (lowers CT win probability)
- `lag_00__CT_B_site_active_smokes`: coefficient `-0.000879` (lowers CT win probability)
- `lag_14__CT3__smoke`: coefficient `0.000829` (raises CT win probability)
- `lag_09__T_flash_duration_sum`: coefficient `-0.000794` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_PIT`: coefficient `-0.002993` (lowers CT win probability)
- `lag_00__T_bomb_zone_count`: coefficient `-0.002806` (lowers CT win probability)
- `lag_10__T_place_QUAD`: coefficient `-0.002269` (lowers CT win probability)
- `lag_13__CT_place_LIBRARY`: coefficient `-0.002078` (lowers CT win probability)
- `lag_01__T_place_PIT`: coefficient `-0.002027` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001950` (raises CT win probability)
- `lag_05__T_duck_amount_mean`: coefficient `0.001834` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001832` (raises CT win probability)
- `lag_14__CT_place_LIBRARY`: coefficient `-0.001751` (lowers CT win probability)
- `lag_15__T_bomb_zone_count`: coefficient `0.001688` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `122181`, seconds `64.00`, LSTM delta `-0.2907`

Top all feature movements:
- `lag_10__T_place_QUAD`: contribution `-0.054652`
- `lag_06__T_place_QUAD`: contribution `-0.026071`
- `lag_04__T_place_BALCONY`: contribution `-0.019775`
- `lag_01__T_place_PIT`: contribution `-0.012793`
- `lag_09__T_flash_duration_sum`: contribution `-0.007004`

Top utility-only movements:
- `lag_09__T_flash_duration_sum`: contribution `-0.007004`
- `lag_09__T4__flash_duration`: contribution `-0.005190`
- `lag_09__T1__flash_duration`: contribution `-0.005101`
- `lag_09__T3__flash_duration`: contribution `-0.004739`
- `lag_03__CT1__flash_duration`: contribution `-0.003612`

### tick `122341`, seconds `66.50`, LSTM delta `-0.1876`

Top all feature movements:
- `lag_15__T_place_QUAD`: contribution `-0.025950`
- `lag_00__CT_shots_fired_sum`: contribution `-0.024178`
- `lag_06__T_place_BALCONY`: contribution `-0.008904`
- `lag_09__T_place_BALCONY`: contribution `-0.007797`
- `lag_10__CT_shots_fired_sum`: contribution `-0.005824`

Top utility-only movements:
- `lag_14__T4__flash_duration`: contribution `-0.004141`
- `lag_14__T_flash_duration_sum`: contribution `-0.004007`
- `lag_14__T1__flash_duration`: contribution `-0.002494`
- `lag_14__T3__flash_duration`: contribution `-0.002406`

### tick `122021`, seconds `61.50`, LSTM delta `+0.1773`

Top all feature movements:
- `lag_05__T_place_QUAD`: contribution `+0.030798`
- `lag_01__T_place_QUAD`: contribution `+0.025872`
- `lag_00__CT_shots_fired_sum`: contribution `+0.015270`
- `lag_00__T4__flash_duration`: contribution `+0.005974`
- `lag_04__T_flash_duration_sum`: contribution `+0.005767`

Top utility-only movements:
- `lag_00__T4__flash_duration`: contribution `+0.005974`
- `lag_04__T_flash_duration_sum`: contribution `+0.005767`
- `lag_04__T1__flash_duration`: contribution `+0.004708`
- `lag_04__T3__flash_duration`: contribution `+0.004196`
- `lag_13__CT_B_site_active_infernos`: contribution `+0.001890`

### tick `123205`, seconds `80.00`, LSTM delta `+0.1752`

Top all feature movements:
- `lag_00__T_place_PIT`: contribution `+0.018885`
- `lag_00__T_bomb_zone_count`: contribution `+0.016333`
- `lag_14__CT_place_LIBRARY`: contribution `+0.011227`
- `lag_14__T3__duck_amount`: contribution `+0.005353`
- `lag_05__T_duck_amount_mean`: contribution `+0.004958`

Top utility-only movements:
- `lag_08__T5__smoke`: contribution `+0.002284`

### tick `122085`, seconds `62.50`, LSTM delta `-0.1416`

Top all feature movements:
- `lag_07__T_place_QUAD`: contribution `-0.023091`
- `lag_03__T_place_QUAD`: contribution `-0.017532`
- `lag_00__CT_shots_fired_sum`: contribution `-0.008908`
- `lag_01__T_place_BALCONY`: contribution `+0.005076`
- `lag_00__kill_diff_last_3s`: contribution `-0.004694`

Top utility-only movements:
- `lag_06__T_flash_duration_sum`: contribution `-0.004302`
- `lag_06__T1__flash_duration`: contribution `-0.003784`
- `lag_06__T3__flash_duration`: contribution `-0.003384`
- `lag_06__T4__flash_duration`: contribution `-0.003087`
- `lag_00__T4__flash_duration`: contribution `-0.002619`
