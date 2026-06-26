# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-gamerlegion-vs-tyloo-bo3-0g9mXt3FIxC8XzjXNUjRL7/gamerlegion-vs-tyloo-m1-ancient-p3.csv`
- round_num: `10`

## Largest probability jumps

- tick `87031`, seconds `110.00`, LSTM `0.8187`, delta `+0.2860`
- tick `87415`, seconds `116.00`, LSTM `0.8848`, delta `+0.2476`
- tick `86839`, seconds `107.00`, LSTM `0.4655`, delta `+0.2033`
- tick `87383`, seconds `115.50`, LSTM `0.6373`, delta `-0.1689`
- tick `83991`, seconds `62.50`, LSTM `0.4223`, delta `-0.1616`
- tick `87351`, seconds `115.00`, LSTM `0.8061`, delta `-0.1446`
- tick `86615`, seconds `103.50`, LSTM `0.3198`, delta `-0.1376`
- tick `86999`, seconds `109.50`, LSTM `0.5328`, delta `+0.1293`
- tick `86135`, seconds `96.00`, LSTM `0.3186`, delta `-0.1076`
- tick `87703`, seconds `120.50`, LSTM `0.9499`, delta `+0.1073`

## Top 15 local ridge features

- `lag_00__T_flashes_last_5s`: coefficient `-0.003272`, |coef| `0.003272`
- `lag_00__kill_diff_last_3s`: coefficient `0.003187`, |coef| `0.003187`
- `lag_06__CT_place_TSIDEUPPER`: coefficient `-0.002971`, |coef| `0.002971`
- `lag_07__CT_place_TSIDEUPPER`: coefficient `-0.002924`, |coef| `0.002924`
- `lag_00__damage_diff_last_5s`: coefficient `0.002796`, |coef| `0.002796`
- `lag_01__T_flash_duration_sum`: coefficient `0.002764`, |coef| `0.002764`
- `lag_04__CT_place_MAINHALL`: coefficient `0.002742`, |coef| `0.002742`
- `lag_11__T1__flash_duration`: coefficient `-0.002739`, |coef| `0.002739`
- `lag_01__T5__flash_duration`: coefficient `0.002708`, |coef| `0.002708`
- `lag_00__T_shots_fired_sum`: coefficient `-0.002464`, |coef| `0.002464`
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.002414`, |coef| `0.002414`
- `lag_00__T1__flash_duration`: coefficient `0.002126`, |coef| `0.002126`
- `lag_13__T5__flash_duration`: coefficient `0.002098`, |coef| `0.002098`
- `lag_01__T_flashed_players`: coefficient `0.002081`, |coef| `0.002081`
- `lag_12__T5__flash_duration`: coefficient `-0.002079`, |coef| `0.002079`

## Top 10 utility ridge features

- `lag_00__T_flashes_last_5s`: coefficient `-0.003272` (lowers CT win probability)
- `lag_01__T_flash_duration_sum`: coefficient `0.002764` (raises CT win probability)
- `lag_11__T1__flash_duration`: coefficient `-0.002739` (lowers CT win probability)
- `lag_01__T5__flash_duration`: coefficient `0.002708` (raises CT win probability)
- `lag_00__T1__flash_duration`: coefficient `0.002126` (raises CT win probability)
- `lag_13__T5__flash_duration`: coefficient `0.002098` (raises CT win probability)
- `lag_12__T5__flash_duration`: coefficient `-0.002079` (lowers CT win probability)
- `lag_01__T1__flash_duration`: coefficient `0.002073` (raises CT win probability)
- `lag_06__T_flashes_last_5s`: coefficient `0.001899` (raises CT win probability)
- `lag_13__T_flash_duration_sum`: coefficient `0.001888` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003187` (raises CT win probability)
- `lag_06__CT_place_TSIDEUPPER`: coefficient `-0.002971` (lowers CT win probability)
- `lag_07__CT_place_TSIDEUPPER`: coefficient `-0.002924` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002796` (raises CT win probability)
- `lag_04__CT_place_MAINHALL`: coefficient `0.002742` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.002464` (lowers CT win probability)
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.002414` (raises CT win probability)
- `lag_01__T_flashed_players`: coefficient `0.002081` (raises CT win probability)
- `lag_07__CT2__is_walking`: coefficient `-0.002044` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002035` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `87031`, seconds `110.00`, LSTM delta `+0.2860`

Top all feature movements:
- `lag_01__T_flash_duration_sum`: contribution `+0.022897`
- `lag_04__CT_place_MAINHALL`: contribution `+0.022694`
- `lag_01__T5__flash_duration`: contribution `+0.021058`
- `lag_01__T1__flash_duration`: contribution `+0.015252`
- `lag_10__CT_place_TSIDEUPPER`: contribution `+0.014796`

Top utility-only movements:
- `lag_01__T_flash_duration_sum`: contribution `+0.022897`
- `lag_01__T5__flash_duration`: contribution `+0.021058`
- `lag_01__T1__flash_duration`: contribution `+0.015252`
- `lag_11__T1__flash_duration`: contribution `+0.013964`
- `lag_01__T4__flash_duration`: contribution `+0.007981`

### tick `87415`, seconds `116.00`, LSTM delta `+0.2476`

Top all feature movements:
- `lag_13__T5__flash_duration`: contribution `+0.016319`
- `lag_12__T5__flash_duration`: contribution `+0.016169`
- `lag_13__T_flash_duration_sum`: contribution `+0.015641`
- `lag_13__T1__flash_duration`: contribution `+0.011639`
- `lag_00__kill_diff_last_3s`: contribution `+0.007671`

Top utility-only movements:
- `lag_13__T5__flash_duration`: contribution `+0.016319`
- `lag_12__T5__flash_duration`: contribution `+0.016169`
- `lag_13__T_flash_duration_sum`: contribution `+0.015641`
- `lag_13__T1__flash_duration`: contribution `+0.011639`
- `lag_07__CT2__flash_duration`: contribution `+0.005814`

### tick `86839`, seconds `107.00`, LSTM delta `+0.2033`

Top all feature movements:
- `lag_07__CT_place_TSIDEUPPER`: contribution `+0.021981`
- `lag_08__T3__is_scoped`: contribution `+0.009995`
- `lag_14__CT_place_TSIDEUPPER`: contribution `+0.008723`
- `lag_00__kill_diff_last_3s`: contribution `+0.007671`
- `lag_14__T_flashed_players`: contribution `+0.006315`

Top utility-only movements:
- `lag_05__T1__flash_duration`: contribution `+0.004062`
- `lag_14__T1__flash_duration`: contribution `+0.003608`
- `lag_06__T_B_site_active_infernos`: contribution `+0.003415`
- `lag_14__T_flash_duration_sum`: contribution `+0.002903`

### tick `87383`, seconds `115.50`, LSTM delta `-0.1689`

Top all feature movements:
- `lag_12__T5__flash_duration`: contribution `-0.016169`
- `lag_15__CT_place_TSIDEUPPER`: contribution `+0.013299`
- `lag_00__T_shots_fired_sum`: contribution `-0.011086`
- `lag_01__damage_diff_last_5s`: contribution `-0.009786`
- `lag_12__T_flash_duration_sum`: contribution `-0.009655`

Top utility-only movements:
- `lag_12__T5__flash_duration`: contribution `-0.016169`
- `lag_12__T_flash_duration_sum`: contribution `-0.009655`
- `lag_06__CT2__flash_duration`: contribution `-0.007763`
- `lag_11__T_flash_duration_sum`: contribution `+0.003896`
- `lag_12__T1__flash_duration`: contribution `-0.003119`

### tick `83991`, seconds `62.50`, LSTM delta `-0.1616`

Top all feature movements:
- `lag_11__CT4__flash_duration`: contribution `-0.011102`
- `lag_00__kill_diff_last_3s`: contribution `-0.007671`
- `lag_00__T_kills_last_3s`: contribution `-0.006448`
- `lag_00__damage_diff_last_5s`: contribution `-0.006308`
- `lag_14__CT5__duck_amount`: contribution `-0.006072`

Top utility-only movements:
- `lag_11__CT4__flash_duration`: contribution `-0.011102`
- `lag_02__T_A_site_active_infernos`: contribution `-0.004094`
- `lag_02__T_B_site_active_infernos`: contribution `-0.003155`
