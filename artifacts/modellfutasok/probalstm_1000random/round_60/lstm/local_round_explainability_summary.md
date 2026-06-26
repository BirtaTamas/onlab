# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-natus-vincere-vs-tyloo-bo3-u9zlDGjnIy0eSohnO5P-Xx/natus-vincere-vs-tyloo-m3-ancient.csv`
- round_num: `18`

## Largest probability jumps

- tick `131148`, seconds `12.50`, LSTM `0.8063`, delta `+0.1959`
- tick `131596`, seconds `19.50`, LSTM `0.5560`, delta `-0.1886`
- tick `131660`, seconds `20.50`, LSTM `0.3913`, delta `-0.1768`
- tick `131020`, seconds `10.50`, LSTM `0.5321`, delta `+0.1263`
- tick `133612`, seconds `51.00`, LSTM `0.3635`, delta `+0.0902`
- tick `134572`, seconds `66.00`, LSTM `0.4163`, delta `-0.0889`
- tick `136172`, seconds `91.00`, LSTM `0.2235`, delta `+0.0841`
- tick `134348`, seconds `62.50`, LSTM `0.4660`, delta `+0.0809`
- tick `137836`, seconds `117.00`, LSTM `0.0418`, delta `-0.0782`
- tick `135596`, seconds `82.00`, LSTM `0.3132`, delta `+0.0739`

## Top 15 local ridge features

- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.005102`, |coef| `0.005102`
- `lag_00__CT_place_SIDEHALL`: coefficient `-0.003065`, |coef| `0.003065`
- `lag_01__CT_place_HOUSE`: coefficient `-0.002699`, |coef| `0.002699`
- `lag_00__CT_place_HOUSE`: coefficient `-0.002537`, |coef| `0.002537`
- `lag_00__CT4__duck_amount`: coefficient `0.002226`, |coef| `0.002226`
- `lag_14__CT_mollies_last_5s`: coefficient `-0.002122`, |coef| `0.002122`
- `lag_08__T_he_last_5s`: coefficient `0.001947`, |coef| `0.001947`
- `lag_14__T_place_MAINHALL`: coefficient `-0.001923`, |coef| `0.001923`
- `lag_00__T1__is_walking`: coefficient `0.001863`, |coef| `0.001863`
- `lag_15__T_place_MAINHALL`: coefficient `-0.001843`, |coef| `0.001843`
- `lag_04__CT3__is_walking`: coefficient `0.001801`, |coef| `0.001801`
- `lag_00__T_place_MAINHALL`: coefficient `0.001743`, |coef| `0.001743`
- `lag_10__T_he_last_5s`: coefficient `0.001710`, |coef| `0.001710`
- `lag_01__CT_place_SIDEHALL`: coefficient `-0.001702`, |coef| `0.001702`
- `lag_01__CT_place_TOPOFMID`: coefficient `0.001676`, |coef| `0.001676`

## Top 10 utility ridge features

- `lag_14__CT_mollies_last_5s`: coefficient `-0.002122` (lowers CT win probability)
- `lag_08__T_he_last_5s`: coefficient `0.001947` (raises CT win probability)
- `lag_10__T_he_last_5s`: coefficient `0.001710` (raises CT win probability)
- `lag_00__CT_mollies_last_5s`: coefficient `-0.001612` (lowers CT win probability)
- `lag_00__T_he_last_5s`: coefficient `0.001264` (raises CT win probability)
- `lag_00__T_A_site_active_infernos`: coefficient `-0.001249` (lowers CT win probability)
- `lag_13__CT_he_last_5s`: coefficient `-0.001176` (lowers CT win probability)
- `lag_03__T_A_site_active_infernos`: coefficient `-0.001136` (lowers CT win probability)
- `lag_12__CT_utility_damage_last_5s`: coefficient `0.001105` (raises CT win probability)
- `lag_14__T2__flash_duration`: coefficient `0.001101` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.005102` (raises CT win probability)
- `lag_00__CT_place_SIDEHALL`: coefficient `-0.003065` (lowers CT win probability)
- `lag_01__CT_place_HOUSE`: coefficient `-0.002699` (lowers CT win probability)
- `lag_00__CT_place_HOUSE`: coefficient `-0.002537` (lowers CT win probability)
- `lag_00__CT4__duck_amount`: coefficient `0.002226` (raises CT win probability)
- `lag_14__T_place_MAINHALL`: coefficient `-0.001923` (lowers CT win probability)
- `lag_00__T1__is_walking`: coefficient `0.001863` (raises CT win probability)
- `lag_15__T_place_MAINHALL`: coefficient `-0.001843` (lowers CT win probability)
- `lag_04__CT3__is_walking`: coefficient `0.001801` (raises CT win probability)
- `lag_00__T_place_MAINHALL`: coefficient `0.001743` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `131148`, seconds `12.50`, LSTM delta `+0.1959`

Top all feature movements:
- `lag_14__CT_mollies_last_5s`: contribution `+0.070357`
- `lag_13__CT_he_last_5s`: contribution `+0.021571`
- `lag_04__T_he_last_5s`: contribution `+0.009570`
- `lag_12__CT_flashes_last_5s`: contribution `+0.007737`
- `lag_15__CT_place_HOUSE`: contribution `-0.007367`

Top utility-only movements:
- `lag_14__CT_mollies_last_5s`: contribution `+0.070357`
- `lag_13__CT_he_last_5s`: contribution `+0.021571`
- `lag_04__T_he_last_5s`: contribution `+0.009570`
- `lag_12__CT_flashes_last_5s`: contribution `+0.007737`
- `lag_06__T2__flash_duration`: contribution `+0.007245`

### tick `131596`, seconds `19.50`, LSTM delta `-0.1886`

Top all feature movements:
- `lag_08__T_he_last_5s`: contribution `-0.025415`
- `lag_14__T2__flash_duration`: contribution `-0.006679`
- `lag_10__CT2__flash_duration`: contribution `-0.006175`
- `lag_00__CT5__duck_amount`: contribution `-0.005979`
- `lag_12__CT_utility_damage_last_5s`: contribution `-0.005474`

Top utility-only movements:
- `lag_08__T_he_last_5s`: contribution `-0.025415`
- `lag_14__T2__flash_duration`: contribution `-0.006679`
- `lag_10__CT2__flash_duration`: contribution `-0.006175`
- `lag_12__CT_utility_damage_last_5s`: contribution `-0.005474`
- `lag_10__CT_flash_duration_sum`: contribution `-0.003620`

### tick `131660`, seconds `20.50`, LSTM delta `-0.1768`

Top all feature movements:
- `lag_10__T_he_last_5s`: contribution `-0.022321`
- `lag_00__T_kills_last_3s`: contribution `-0.005006`
- `lag_12__CT2__flash_duration`: contribution `-0.004648`
- `lag_15__CT_utility_damage_last_5s`: contribution `-0.004530`
- `lag_11__T4__is_walking`: contribution `-0.003416`

Top utility-only movements:
- `lag_10__T_he_last_5s`: contribution `-0.022321`
- `lag_12__CT2__flash_duration`: contribution `-0.004648`
- `lag_15__CT_utility_damage_last_5s`: contribution `-0.004530`
- `lag_14__CT_utility_damage_last_5s`: contribution `-0.003288`
- `lag_12__CT_flash_duration_sum`: contribution `-0.002948`

### tick `131020`, seconds `10.50`, LSTM delta `+0.1263`

Top all feature movements:
- `lag_10__CT_mollies_last_5s`: contribution `+0.016656`
- `lag_00__T_he_last_5s`: contribution `+0.016500`
- `lag_13__CT_place_HOUSE`: contribution `+0.010725`
- `lag_10__T_place_TUNNEL`: contribution `+0.007410`
- `lag_09__CT_he_last_5s`: contribution `+0.005106`

Top utility-only movements:
- `lag_10__CT_mollies_last_5s`: contribution `+0.016656`
- `lag_00__T_he_last_5s`: contribution `+0.016500`
- `lag_09__CT_he_last_5s`: contribution `+0.005106`
- `lag_05__CT_utility_damage_last_5s`: contribution `+0.003208`
- `lag_04__CT_utility_damage_last_5s`: contribution `+0.003148`

### tick `133612`, seconds `51.00`, LSTM delta `+0.0902`

Top all feature movements:
- `lag_01__CT_place_HOUSE`: contribution `+0.019069`
- `lag_00__CT_place_SIDEHALL`: contribution `+0.013109`
- `lag_01__CT_place_TOPOFMID`: contribution `+0.012160`
- `lag_11__T_place_TUNNEL`: contribution `+0.007396`
- `lag_04__CT3__is_walking`: contribution `+0.004299`

Top utility-only movements:
- No utility movement among the top local contributors.
