# Local Round Explainability

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-tyloo-vs-rare-atom-bo3-8GB1HWZtKOlh9_707n2A62/tyloo-vs-rare-atom-m2-inferno.csv`
- round_num: `15`

## Largest probability jumps

- tick `129981`, seconds `77.50`, LSTM `0.2535`, delta `-0.2277`
- tick `128989`, seconds `62.00`, LSTM `0.3212`, delta `-0.2157`
- tick `128861`, seconds `60.00`, LSTM `0.4894`, delta `+0.1560`
- tick `130173`, seconds `80.50`, LSTM `0.0153`, delta `-0.1020`
- tick `129373`, seconds `68.00`, LSTM `0.3741`, delta `+0.0621`
- tick `130109`, seconds `79.50`, LSTM `0.1173`, delta `-0.0611`
- tick `128829`, seconds `59.50`, LSTM `0.3335`, delta `+0.0607`
- tick `125085`, seconds `1.00`, LSTM `0.2093`, delta `-0.0597`
- tick `125341`, seconds `5.00`, LSTM `0.2688`, delta `+0.0507`
- tick `127325`, seconds `36.00`, LSTM `0.2921`, delta `+0.0440`

## Top 15 local ridge features

- `lag_00__CT5__flash_duration`: coefficient `-0.003080`, |coef| `0.003080`
- `lag_00__kill_diff_last_3s`: coefficient `0.002814`, |coef| `0.002814`
- `lag_00__T_kills_last_3s`: coefficient `-0.002551`, |coef| `0.002551`
- `lag_03__T_place_MIDDLE`: coefficient `0.002082`, |coef| `0.002082`
- `lag_00__T_damage_last_5s`: coefficient `-0.002000`, |coef| `0.002000`
- `lag_00__T_place_QUAD`: coefficient `-0.001984`, |coef| `0.001984`
- `lag_00__CT_place_APARTMENTS`: coefficient `0.001958`, |coef| `0.001958`
- `lag_12__CT4__duck_amount`: coefficient `0.001955`, |coef| `0.001955`
- `lag_00__damage_diff_last_5s`: coefficient `0.001883`, |coef| `0.001883`
- `lag_05__CT4__duck_amount`: coefficient `0.001831`, |coef| `0.001831`
- `lag_07__T2__duck_amount`: coefficient `0.001702`, |coef| `0.001702`
- `lag_03__T2__duck_amount`: coefficient `-0.001674`, |coef| `0.001674`
- `lag_03__T_place_TOPOFMID`: coefficient `-0.001664`, |coef| `0.001664`
- `lag_06__T2__duck_amount`: coefficient `0.001611`, |coef| `0.001611`
- `lag_01__T_place_TOPOFMID`: coefficient `-0.001597`, |coef| `0.001597`

## Top 10 utility ridge features

- `lag_00__CT5__flash_duration`: coefficient `-0.003080` (lowers CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `-0.001416` (lowers CT win probability)
- `lag_02__T4__smoke`: coefficient `0.001162` (raises CT win probability)
- `lag_01__T3__smoke`: coefficient `-0.001111` (lowers CT win probability)
- `lag_04__CT5__flash_duration`: coefficient `-0.001056` (lowers CT win probability)
- `lag_01__CT5__flash_duration`: coefficient `-0.001022` (lowers CT win probability)
- `lag_13__T3__flash`: coefficient `0.000978` (raises CT win probability)
- `lag_03__T2__smoke`: coefficient `-0.000960` (lowers CT win probability)
- `lag_05__CT5__flash_duration`: coefficient `-0.000803` (lowers CT win probability)
- `lag_02__T2__smoke`: coefficient `-0.000803` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002814` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002551` (lowers CT win probability)
- `lag_03__T_place_MIDDLE`: coefficient `0.002082` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002000` (lowers CT win probability)
- `lag_00__T_place_QUAD`: coefficient `-0.001984` (lowers CT win probability)
- `lag_00__CT_place_APARTMENTS`: coefficient `0.001958` (raises CT win probability)
- `lag_12__CT4__duck_amount`: coefficient `0.001955` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001883` (raises CT win probability)
- `lag_05__CT4__duck_amount`: coefficient `0.001831` (raises CT win probability)
- `lag_07__T2__duck_amount`: coefficient `0.001702` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `129981`, seconds `77.50`, LSTM delta `-0.2277`

Top all feature movements:
- `lag_00__CT5__flash_duration`: contribution `-0.019182`
- `lag_00__T_kills_last_3s`: contribution `-0.008081`
- `lag_00__CT_place_APARTMENTS`: contribution `-0.007521`
- `lag_12__CT4__duck_amount`: contribution `-0.007179`
- `lag_00__kill_diff_last_3s`: contribution `-0.006773`

Top utility-only movements:
- `lag_00__CT5__flash_duration`: contribution `-0.019182`
- `lag_00__CT_flash_duration_sum`: contribution `-0.004066`

### tick `128989`, seconds `62.00`, LSTM delta `-0.2157`

Top all feature movements:
- `lag_06__CT_place_LIBRARY`: contribution `-0.008316`
- `lag_00__T_kills_last_3s`: contribution `-0.008081`
- `lag_00__kill_diff_last_3s`: contribution `-0.006773`
- `lag_07__T2__duck_amount`: contribution `-0.006505`
- `lag_03__T_place_UNDERPASS`: contribution `-0.005730`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `128861`, seconds `60.00`, LSTM delta `+0.1560`

Top all feature movements:
- `lag_03__CT_place_LIBRARY`: contribution `+0.009576`
- `lag_00__kill_diff_last_3s`: contribution `+0.006773`
- `lag_05__CT4__duck_amount`: contribution `+0.006724`
- `lag_06__T2__duck_amount`: contribution `+0.006159`
- `lag_14__CT3__duck_amount`: contribution `+0.005212`

Top utility-only movements:
- `lag_01__T3__smoke`: contribution `+0.002415`

### tick `130173`, seconds `80.50`, LSTM delta `-0.1020`

Top all feature movements:
- `lag_00__T_place_QUAD`: contribution `-0.047798`
- `lag_12__CT4__duck_amount`: contribution `+0.007179`
- `lag_05__CT4__duck_amount`: contribution `-0.006724`
- `lag_05__CT5__flash_duration`: contribution `-0.005002`
- `lag_00__T_damage_last_5s`: contribution `-0.004077`

Top utility-only movements:
- `lag_05__CT5__flash_duration`: contribution `-0.005002`

### tick `129373`, seconds `68.00`, LSTM delta `+0.0621`

Top all feature movements:
- `lag_06__CT_place_LIBRARY`: contribution `+0.008316`
- `lag_03__T_place_UNDERPASS`: contribution `+0.005730`
- `lag_14__CT3__duck_amount`: contribution `+0.004401`
- `lag_00__CT_place_RUINS`: contribution `+0.003497`
- `lag_12__CT3__duck_amount`: contribution `+0.003386`

Top utility-only movements:
- No utility movement among the top local contributors.
