# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-vitality-bo5-RwgqrXEuhDJTxQHhSIn72X/mouz-vs-vitality-m2-nuke.csv`
- round_num: `5`

## Largest probability jumps

- tick `26292`, seconds `33.50`, LSTM `0.8096`, delta `+0.1711`
- tick `28500`, seconds `68.00`, LSTM `0.8694`, delta `-0.0873`
- tick `26356`, seconds `34.50`, LSTM `0.9102`, delta `+0.0834`
- tick `25524`, seconds `21.50`, LSTM `0.6196`, delta `-0.0448`
- tick `28628`, seconds `70.00`, LSTM `0.9105`, delta `+0.0362`
- tick `25204`, seconds `16.50`, LSTM `0.6499`, delta `-0.0333`
- tick `26260`, seconds `33.00`, LSTM `0.6384`, delta `-0.0276`
- tick `28692`, seconds `71.00`, LSTM `0.9142`, delta `+0.0271`
- tick `24180`, seconds `0.50`, LSTM `0.7447`, delta `-0.0261`
- tick `28532`, seconds `68.50`, LSTM `0.8944`, delta `+0.0250`

## Top 15 local ridge features

- `lag_00__CT_place_LOCKERROOM`: coefficient `-0.001988`, |coef| `0.001988`
- `lag_10__T_place_ADMIN`: coefficient `0.001702`, |coef| `0.001702`
- `lag_03__T_place_CONTROL`: coefficient `0.001467`, |coef| `0.001467`
- `lag_15__T_place_TROPHY`: coefficient `0.001316`, |coef| `0.001316`
- `lag_03__T_place_TROPHY`: coefficient `-0.001269`, |coef| `0.001269`
- `lag_10__T_place_HELL`: coefficient `-0.001191`, |coef| `0.001191`
- `lag_01__T_place_CONTROL`: coefficient `0.001054`, |coef| `0.001054`
- `lag_00__T_place_VENDING`: coefficient `-0.000967`, |coef| `0.000967`
- `lag_14__T_place_VENDING`: coefficient `0.000935`, |coef| `0.000935`
- `lag_06__T_place_VENDING`: coefficient `-0.000923`, |coef| `0.000923`
- `lag_00__kill_diff_last_3s`: coefficient `0.000892`, |coef| `0.000892`
- `lag_05__T3__duck_amount`: coefficient `-0.000891`, |coef| `0.000891`
- `lag_07__T_place_TROPHY`: coefficient `0.000857`, |coef| `0.000857`
- `lag_00__damage_diff_last_5s`: coefficient `0.000854`, |coef| `0.000854`
- `lag_00__T2__duck_amount`: coefficient `-0.000811`, |coef| `0.000811`

## Top 10 utility ridge features

- `lag_05__T5__flash_duration`: coefficient `0.000502` (raises CT win probability)
- `lag_00__T_smokes_last_5s`: coefficient `-0.000465` (lowers CT win probability)
- `lag_13__CT3__smoke`: coefficient `-0.000464` (lowers CT win probability)
- `lag_08__T_smokes_last_5s`: coefficient `-0.000462` (lowers CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `0.000447` (raises CT win probability)
- `lag_00__T5__flash_duration`: coefficient `-0.000421` (lowers CT win probability)
- `lag_07__CT_B_site_active_smokes`: coefficient `0.000421` (raises CT win probability)
- `lag_07__CT_A_site_active_smokes`: coefficient `0.000409` (raises CT win probability)
- `lag_04__T1__flash`: coefficient `-0.000378` (lowers CT win probability)
- `lag_08__CT1__flash`: coefficient `-0.000371` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_LOCKERROOM`: coefficient `-0.001988` (lowers CT win probability)
- `lag_10__T_place_ADMIN`: coefficient `0.001702` (raises CT win probability)
- `lag_03__T_place_CONTROL`: coefficient `0.001467` (raises CT win probability)
- `lag_15__T_place_TROPHY`: coefficient `0.001316` (raises CT win probability)
- `lag_03__T_place_TROPHY`: coefficient `-0.001269` (lowers CT win probability)
- `lag_10__T_place_HELL`: coefficient `-0.001191` (lowers CT win probability)
- `lag_01__T_place_CONTROL`: coefficient `0.001054` (raises CT win probability)
- `lag_00__T_place_VENDING`: coefficient `-0.000967` (lowers CT win probability)
- `lag_14__T_place_VENDING`: coefficient `0.000935` (raises CT win probability)
- `lag_06__T_place_VENDING`: coefficient `-0.000923` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `26292`, seconds `33.50`, LSTM delta `+0.1711`

Top all feature movements:
- `lag_00__CT_place_LOCKERROOM`: contribution `+0.024744`
- `lag_03__T_place_CONTROL`: contribution `+0.010426`
- `lag_15__T_place_TROPHY`: contribution `+0.008347`
- `lag_03__T_place_TROPHY`: contribution `+0.008049`
- `lag_01__T_place_CONTROL`: contribution `+0.007491`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `28500`, seconds `68.00`, LSTM delta `-0.0873`

Top all feature movements:
- `lag_10__T_place_ADMIN`: contribution `-0.033090`
- `lag_10__T_place_HELL`: contribution `-0.025391`
- `lag_00__kill_diff_last_3s`: contribution `-0.002148`
- `lag_00__T_shots_fired_sum`: contribution `-0.002026`
- `lag_00__damage_diff_last_5s`: contribution `-0.001926`

Top utility-only movements:
- `lag_12__T_A_site_active_infernos`: contribution `-0.001015`
- `lag_12__T_B_site_active_infernos`: contribution `-0.000915`

### tick `26356`, seconds `34.50`, LSTM delta `+0.0834`

Top all feature movements:
- `lag_03__T_place_CONTROL`: contribution `+0.010426`
- `lag_03__T_place_TROPHY`: contribution `+0.008049`
- `lag_02__CT_place_LOCKERROOM`: contribution `+0.006339`
- `lag_08__T_place_TROPHY`: contribution `+0.004495`
- `lag_05__T_place_TROPHY`: contribution `+0.003987`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `25524`, seconds `21.50`, LSTM delta `-0.0448`

Top all feature movements:
- `lag_00__CT_place_LOCKERROOM`: contribution `-0.024744`
- `lag_07__CT_place_LOCKERROOM`: contribution `-0.006716`
- `lag_00__T_place_TROPHY`: contribution `-0.003188`
- `lag_00__CT_place_HELL`: contribution `-0.002249`
- `lag_05__T3__duck_amount`: contribution `-0.001922`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `28628`, seconds `70.00`, LSTM delta `+0.0362`

Top all feature movements:
- `lag_00__T_place_ADMIN`: contribution `+0.011661`
- `lag_14__T_place_HELL`: contribution `+0.009938`
- `lag_14__T_place_ADMIN`: contribution `+0.005001`
- `lag_00__T_duck_amount_mean`: contribution `+0.003132`
- `lag_00__T5__duck_amount`: contribution `+0.001658`

Top utility-only movements:
- No utility movement among the top local contributors.
