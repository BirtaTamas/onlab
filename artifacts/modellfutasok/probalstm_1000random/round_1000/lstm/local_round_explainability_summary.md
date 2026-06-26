# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-legacy-bo3-o9hWAn-ugamRsSw8ngEfHF/astralis-vs-legacy-m3-ancient.csv`
- round_num: `6`

## Largest probability jumps

- tick `35510`, seconds `27.50`, LSTM `0.1795`, delta `-0.2481`
- tick `36278`, seconds `39.50`, LSTM `0.0299`, delta `-0.0850`
- tick `35542`, seconds `28.00`, LSTM `0.1213`, delta `-0.0582`
- tick `36182`, seconds `38.00`, LSTM `0.1494`, delta `-0.0500`
- tick `34518`, seconds `12.00`, LSTM `0.4719`, delta `-0.0445`
- tick `36246`, seconds `39.00`, LSTM `0.1149`, delta `-0.0405`
- tick `34742`, seconds `15.50`, LSTM `0.4978`, delta `+0.0392`
- tick `34998`, seconds `19.50`, LSTM `0.4408`, delta `-0.0337`
- tick `36022`, seconds `35.50`, LSTM `0.1641`, delta `+0.0322`
- tick `34710`, seconds `15.00`, LSTM `0.4586`, delta `+0.0300`

## Top 15 local ridge features

- `lag_00__CT_place_UNKNOWN`: coefficient `0.002717`, |coef| `0.002717`
- `lag_14__T3__flash_duration`: coefficient `0.002489`, |coef| `0.002489`
- `lag_15__T2__flash_duration`: coefficient `0.002350`, |coef| `0.002350`
- `lag_02__CT_utility_damage_last_5s`: coefficient `0.001689`, |coef| `0.001689`
- `lag_02__CT_place_SIDEENTRANCE`: coefficient `0.001661`, |coef| `0.001661`
- `lag_12__CT_utility_damage_last_5s`: coefficient `-0.001504`, |coef| `0.001504`
- `lag_00__T_kills_last_3s`: coefficient `-0.001454`, |coef| `0.001454`
- `lag_15__T_flash_duration_sum`: coefficient `0.001447`, |coef| `0.001447`
- `lag_00__CT3__alive`: coefficient `0.001303`, |coef| `0.001303`
- `lag_02__utility_damage_diff_last_5s`: coefficient `0.001294`, |coef| `0.001294`
- `lag_14__T_flash_duration_sum`: coefficient `0.001276`, |coef| `0.001276`
- `lag_11__T4__duck_amount`: coefficient `0.001266`, |coef| `0.001266`
- `lag_12__utility_damage_diff_last_5s`: coefficient `-0.001250`, |coef| `0.001250`
- `lag_00__CT_burning_players`: coefficient `0.001219`, |coef| `0.001219`
- `lag_00__CT3__armor`: coefficient `0.001193`, |coef| `0.001193`

## Top 10 utility ridge features

- `lag_14__T3__flash_duration`: coefficient `0.002489` (raises CT win probability)
- `lag_15__T2__flash_duration`: coefficient `0.002350` (raises CT win probability)
- `lag_02__CT_utility_damage_last_5s`: coefficient `0.001689` (raises CT win probability)
- `lag_12__CT_utility_damage_last_5s`: coefficient `-0.001504` (lowers CT win probability)
- `lag_15__T_flash_duration_sum`: coefficient `0.001447` (raises CT win probability)
- `lag_02__utility_damage_diff_last_5s`: coefficient `0.001294` (raises CT win probability)
- `lag_14__T_flash_duration_sum`: coefficient `0.001276` (raises CT win probability)
- `lag_12__utility_damage_diff_last_5s`: coefficient `-0.001250` (lowers CT win probability)
- `lag_00__CT3__smoke`: coefficient `0.001189` (raises CT win probability)
- `lag_15__T3__flash_duration`: coefficient `0.001137` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_UNKNOWN`: coefficient `0.002717` (raises CT win probability)
- `lag_02__CT_place_SIDEENTRANCE`: coefficient `0.001661` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001454` (lowers CT win probability)
- `lag_00__CT3__alive`: coefficient `0.001303` (raises CT win probability)
- `lag_11__T4__duck_amount`: coefficient `0.001266` (raises CT win probability)
- `lag_00__CT_burning_players`: coefficient `0.001219` (raises CT win probability)
- `lag_00__CT3__armor`: coefficient `0.001193` (raises CT win probability)
- `lag_15__CT_burning_players`: coefficient `-0.001174` (lowers CT win probability)
- `lag_00__T2__duck_amount`: coefficient `-0.001126` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001110` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `35510`, seconds `27.50`, LSTM delta `-0.2481`

Top all feature movements:
- `lag_14__T3__flash_duration`: contribution `-0.017282`
- `lag_15__T2__flash_duration`: contribution `-0.015999`
- `lag_02__CT_utility_damage_last_5s`: contribution `-0.008739`
- `lag_12__CT_utility_damage_last_5s`: contribution `-0.007778`
- `lag_02__CT_place_SIDEENTRANCE`: contribution `-0.006686`

Top utility-only movements:
- `lag_14__T3__flash_duration`: contribution `-0.017282`
- `lag_15__T2__flash_duration`: contribution `-0.015999`
- `lag_02__CT_utility_damage_last_5s`: contribution `-0.008739`
- `lag_12__CT_utility_damage_last_5s`: contribution `-0.007778`
- `lag_02__utility_damage_diff_last_5s`: contribution `-0.005490`

### tick `36278`, seconds `39.50`, LSTM delta `-0.0850`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.004607`
- `lag_00__T_shots_fired_sum`: contribution `-0.004051`
- `lag_10__CT2__flash_duration`: contribution `-0.003545`
- `lag_00__CT_burning_players`: contribution `-0.003129`
- `lag_00__CT2__flash_duration`: contribution `-0.003099`

Top utility-only movements:
- `lag_10__CT2__flash_duration`: contribution `-0.003545`
- `lag_00__CT2__flash_duration`: contribution `-0.003099`

### tick `35542`, seconds `28.00`, LSTM delta `-0.0582`

Top all feature movements:
- `lag_15__T3__flash_duration`: contribution `-0.007891`
- `lag_00__T_shots_fired_sum`: contribution `+0.005672`
- `lag_15__T_flash_duration_sum`: contribution `-0.004185`
- `lag_03__CT_utility_damage_last_5s`: contribution `-0.003690`
- `lag_01__T_shots_fired_sum`: contribution `-0.003425`

Top utility-only movements:
- `lag_15__T3__flash_duration`: contribution `-0.007891`
- `lag_15__T_flash_duration_sum`: contribution `-0.004185`
- `lag_03__CT_utility_damage_last_5s`: contribution `-0.003690`
- `lag_13__CT_utility_damage_last_5s`: contribution `-0.003309`
- `lag_13__utility_damage_diff_last_5s`: contribution `-0.002333`

### tick `36182`, seconds `38.00`, LSTM delta `-0.0500`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.004629`
- `lag_04__CT_place_HOUSE`: contribution `-0.002180`
- `lag_07__CT2__flash_duration`: contribution `-0.002178`
- `lag_04__CT1__is_walking`: contribution `+0.001691`
- `lag_05__T_place_HOUSE`: contribution `-0.001599`

Top utility-only movements:
- `lag_07__CT2__flash_duration`: contribution `-0.002178`

### tick `34518`, seconds `12.00`, LSTM delta `-0.0445`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.014657`
- `lag_00__CT_place_MAINHALL`: contribution `-0.004478`
- `lag_03__CT_utility_damage_last_5s`: contribution `+0.004004`
- `lag_14__T_place_TUNNEL`: contribution `-0.003146`
- `lag_00__CT3__shots_fired`: contribution `+0.002532`

Top utility-only movements:
- `lag_03__CT_utility_damage_last_5s`: contribution `+0.004004`
- `lag_03__utility_damage_diff_last_5s`: contribution `+0.002422`
- `lag_03__T3__flash_duration`: contribution `+0.000862`
- `lag_05__CT3__flash_duration`: contribution `-0.000862`
