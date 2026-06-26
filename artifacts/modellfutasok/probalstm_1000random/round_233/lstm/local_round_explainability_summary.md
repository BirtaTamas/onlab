# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-virtuspro-bo3-qivzNI2LmnWi0RrHw-7sxj/falcons-vs-virtus-pro-m2-ancient.csv`
- round_num: `13`

## Largest probability jumps

- tick `107522`, seconds `49.00`, LSTM `0.8497`, delta `+0.2041`
- tick `107266`, seconds `45.00`, LSTM `0.6415`, delta `-0.1407`
- tick `107202`, seconds `44.00`, LSTM `0.6574`, delta `+0.1341`
- tick `107234`, seconds `44.50`, LSTM `0.7823`, delta `+0.1249`
- tick `107170`, seconds `43.50`, LSTM `0.5233`, delta `+0.1044`
- tick `107554`, seconds `49.50`, LSTM `0.9230`, delta `+0.0732`
- tick `107138`, seconds `43.00`, LSTM `0.4189`, delta `+0.0618`
- tick `107106`, seconds `42.50`, LSTM `0.3571`, delta `+0.0592`
- tick `105570`, seconds `18.50`, LSTM `0.5022`, delta `+0.0568`
- tick `107042`, seconds `41.50`, LSTM `0.3323`, delta `-0.0525`

## Top 15 local ridge features

- `lag_05__T_place_RAMP`: coefficient `-0.001932`, |coef| `0.001932`
- `lag_13__CT_flashed_players`: coefficient `0.001853`, |coef| `0.001853`
- `lag_13__T_place_RAMP`: coefficient `0.001823`, |coef| `0.001823`
- `lag_00__CT_kills_last_3s`: coefficient `0.001785`, |coef| `0.001785`
- `lag_00__kill_diff_last_3s`: coefficient `0.001750`, |coef| `0.001750`
- `lag_00__T_place_TSIDELOWER`: coefficient `-0.001581`, |coef| `0.001581`
- `lag_08__T_flashed_players`: coefficient `0.001435`, |coef| `0.001435`
- `lag_13__T_place_TSIDELOWER`: coefficient `-0.001370`, |coef| `0.001370`
- `lag_00__damage_diff_last_5s`: coefficient `0.001326`, |coef| `0.001326`
- `lag_05__T_place_TSIDELOWER`: coefficient `0.001292`, |coef| `0.001292`
- `lag_02__CT_flashed_players`: coefficient `0.001269`, |coef| `0.001269`
- `lag_00__CT_damage_last_5s`: coefficient `0.001242`, |coef| `0.001242`
- `lag_07__T_flashed_players`: coefficient `0.001230`, |coef| `0.001230`
- `lag_13__T2__flash_duration`: coefficient `0.001214`, |coef| `0.001214`
- `lag_12__T_place_TSIDELOWER`: coefficient `-0.001203`, |coef| `0.001203`

## Top 10 utility ridge features

- `lag_13__T2__flash_duration`: coefficient `0.001214` (raises CT win probability)
- `lag_01__T2__flash_duration`: coefficient `0.001178` (raises CT win probability)
- `lag_12__T4__flash_duration`: coefficient `0.001082` (raises CT win probability)
- `lag_09__T2__flash_duration`: coefficient `-0.001034` (lowers CT win probability)
- `lag_13__T_flash_duration_sum`: coefficient `0.000894` (raises CT win probability)
- `lag_02__CT3__flash_duration`: coefficient `0.000884` (raises CT win probability)
- `lag_13__CT3__flash_duration`: coefficient `0.000874` (raises CT win probability)
- `lag_02__CT_flash_duration_sum`: coefficient `0.000814` (raises CT win probability)
- `lag_13__CT_flash_duration_sum`: coefficient `0.000812` (raises CT win probability)
- `lag_13__T4__flash_duration`: coefficient `0.000745` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_05__T_place_RAMP`: coefficient `-0.001932` (lowers CT win probability)
- `lag_13__CT_flashed_players`: coefficient `0.001853` (raises CT win probability)
- `lag_13__T_place_RAMP`: coefficient `0.001823` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001785` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001750` (raises CT win probability)
- `lag_00__T_place_TSIDELOWER`: coefficient `-0.001581` (lowers CT win probability)
- `lag_08__T_flashed_players`: coefficient `0.001435` (raises CT win probability)
- `lag_13__T_place_TSIDELOWER`: coefficient `-0.001370` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001326` (raises CT win probability)
- `lag_05__T_place_TSIDELOWER`: coefficient `0.001292` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `107522`, seconds `49.00`, LSTM delta `+0.2041`

Top all feature movements:
- `lag_13__CT_flashed_players`: contribution `+0.016229`
- `lag_09__T2__flash_duration`: contribution `+0.006883`
- `lag_13__T_flashed_players`: contribution `+0.006827`
- `lag_13__T_place_RAMP`: contribution `+0.006446`
- `lag_13__T2__flash_duration`: contribution `+0.005393`

Top utility-only movements:
- `lag_09__T2__flash_duration`: contribution `+0.006883`
- `lag_13__T2__flash_duration`: contribution `+0.005393`
- `lag_12__T4__flash_duration`: contribution `+0.005221`
- `lag_13__T_flash_duration_sum`: contribution `+0.002549`
- `lag_13__CT3__flash_duration`: contribution `+0.002508`

### tick `107266`, seconds `45.00`, LSTM delta `-0.1407`

Top all feature movements:
- `lag_10__T_flashed_players`: contribution `-0.008516`
- `lag_01__T2__flash_duration`: contribution `-0.007841`
- `lag_05__CT_flashed_players`: contribution `-0.007135`
- `lag_07__T_flashed_players`: contribution `-0.007121`
- `lag_05__T_place_RAMP`: contribution `-0.006833`

Top utility-only movements:
- `lag_01__T2__flash_duration`: contribution `-0.007841`
- `lag_04__T4__flash_duration`: contribution `-0.002420`

### tick `107202`, seconds `44.00`, LSTM delta `+0.1341`

Top all feature movements:
- `lag_08__T_flashed_players`: contribution `+0.011079`
- `lag_03__CT_flashed_players`: contribution `+0.008934`
- `lag_00__CT_kills_last_3s`: contribution `+0.005152`
- `lag_05__T_place_TSIDELOWER`: contribution `-0.004843`
- `lag_05__T_flashed_players`: contribution `+0.004509`

Top utility-only movements:
- `lag_03__T2__flash_duration`: contribution `+0.002898`

### tick `107234`, seconds `44.50`, LSTM delta `+0.1249`

Top all feature movements:
- `lag_04__CT_flashed_players`: contribution `+0.009057`
- `lag_05__T_place_RAMP`: contribution `+0.006833`
- `lag_13__T_place_RAMP`: contribution `+0.006446`
- `lag_04__T_flashed_players`: contribution `-0.005925`
- `lag_00__CT_kills_last_3s`: contribution `+0.005152`

Top utility-only movements:
- `lag_03__T4__flash_duration`: contribution `+0.002670`

### tick `107170`, seconds `43.50`, LSTM delta `+0.1044`

Top all feature movements:
- `lag_02__CT_flashed_players`: contribution `+0.011120`
- `lag_07__T_flashed_players`: contribution `+0.009494`
- `lag_05__T_place_RAMP`: contribution `+0.006833`
- `lag_04__T_flashed_players`: contribution `+0.005925`
- `lag_10__CT_place_HOUSE`: contribution `+0.003846`

Top utility-only movements:
- `lag_01__T2__flash_duration`: contribution `+0.002605`
- `lag_02__CT3__flash_duration`: contribution `+0.002536`
- `lag_02__T2__flash_duration`: contribution `+0.002306`
- `lag_02__CT_flash_duration_sum`: contribution `+0.002248`
