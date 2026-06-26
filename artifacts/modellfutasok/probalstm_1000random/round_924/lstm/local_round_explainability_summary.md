# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-fnatic-vs-legacy-bo3-XoJZ8zL16kSaGnHRZrLL4s/legacy-vs-fnatic-m1-ancient.csv`
- round_num: `4`

## Largest probability jumps

- tick `33282`, seconds `62.00`, LSTM `0.5746`, delta `-0.2839`
- tick `32450`, seconds `49.00`, LSTM `0.6446`, delta `-0.2410`
- tick `33186`, seconds `60.50`, LSTM `0.8182`, delta `+0.2101`
- tick `31522`, seconds `34.50`, LSTM `0.5310`, delta `-0.2066`
- tick `31490`, seconds `34.00`, LSTM `0.7376`, delta `+0.1665`
- tick `32098`, seconds `43.50`, LSTM `0.6948`, delta `+0.1571`
- tick `31042`, seconds `27.00`, LSTM `0.5965`, delta `-0.1357`
- tick `32162`, seconds `44.50`, LSTM `0.8247`, delta `+0.1135`
- tick `31170`, seconds `29.00`, LSTM `0.6591`, delta `+0.0943`
- tick `31298`, seconds `31.00`, LSTM `0.5913`, delta `-0.0715`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005045`, |coef| `0.005045`
- `lag_07__CT_place_MAINHALL`: coefficient `0.004649`, |coef| `0.004649`
- `lag_00__T_duck_amount_mean`: coefficient `-0.004074`, |coef| `0.004074`
- `lag_00__T_kills_last_3s`: coefficient `-0.003739`, |coef| `0.003739`
- `lag_12__T_place_SIDEHALL`: coefficient `0.003649`, |coef| `0.003649`
- `lag_09__T_place_SIDEHALL`: coefficient `-0.003430`, |coef| `0.003430`
- `lag_00__damage_diff_last_5s`: coefficient `0.003154`, |coef| `0.003154`
- `lag_00__CT_place_MAINHALL`: coefficient `0.003047`, |coef| `0.003047`
- `lag_03__CT_place_TUNNEL`: coefficient `0.002719`, |coef| `0.002719`
- `lag_00__CT_kills_last_3s`: coefficient `0.002643`, |coef| `0.002643`
- `lag_10__CT_place_MAINHALL`: coefficient `-0.002528`, |coef| `0.002528`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002412`, |coef| `0.002412`
- `lag_15__T_shots_fired_sum`: coefficient `0.002340`, |coef| `0.002340`
- `lag_06__CT_place_MAINHALL`: coefficient `0.002199`, |coef| `0.002199`
- `lag_00__T_damage_last_5s`: coefficient `-0.002173`, |coef| `0.002173`

## Top 10 utility ridge features

- `lag_09__T4__flash_duration`: coefficient `-0.001381` (lowers CT win probability)
- `lag_04__CT_A_site_active_infernos`: coefficient `0.001226` (raises CT win probability)
- `lag_00__CT1__smoke`: coefficient `0.001133` (raises CT win probability)
- `lag_01__CT2__flash_duration`: coefficient `0.001096` (raises CT win probability)
- `lag_12__T4__flash_duration`: coefficient `-0.000991` (lowers CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `0.000935` (raises CT win probability)
- `lag_01__T4__flash_duration`: coefficient `0.000919` (raises CT win probability)
- `lag_07__T2__flash_duration`: coefficient `-0.000886` (lowers CT win probability)
- `lag_00__CT3__molly`: coefficient `0.000865` (raises CT win probability)
- `lag_07__CT3__smoke`: coefficient `0.000724` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005045` (raises CT win probability)
- `lag_07__CT_place_MAINHALL`: coefficient `0.004649` (raises CT win probability)
- `lag_00__T_duck_amount_mean`: coefficient `-0.004074` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003739` (lowers CT win probability)
- `lag_12__T_place_SIDEHALL`: coefficient `0.003649` (raises CT win probability)
- `lag_09__T_place_SIDEHALL`: coefficient `-0.003430` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003154` (raises CT win probability)
- `lag_00__CT_place_MAINHALL`: coefficient `0.003047` (raises CT win probability)
- `lag_03__CT_place_TUNNEL`: coefficient `0.002719` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002643` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `33282`, seconds `62.00`, LSTM delta `-0.2839`

Top all feature movements:
- `lag_00__CT_place_MAINHALL`: contribution `-0.025217`
- `lag_00__T_duck_amount_mean`: contribution `-0.023694`
- `lag_12__T_place_SIDEHALL`: contribution `-0.023646`
- `lag_10__CT_place_MAINHALL`: contribution `-0.020922`
- `lag_00__kill_diff_last_3s`: contribution `-0.012142`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `32450`, seconds `49.00`, LSTM delta `-0.2410`

Top all feature movements:
- `lag_03__CT_place_TUNNEL`: contribution `-0.043676`
- `lag_09__CT_place_TUNNEL`: contribution `-0.025947`
- `lag_00__kill_diff_last_3s`: contribution `-0.012142`
- `lag_00__T_duck_amount_mean`: contribution `-0.011847`
- `lag_00__T_kills_last_3s`: contribution `-0.011846`

Top utility-only movements:
- `lag_09__T4__flash_duration`: contribution `-0.010714`
- `lag_09__CT2__flash_duration`: contribution `-0.004732`
- `lag_04__CT_A_site_active_infernos`: contribution `-0.004326`
- `lag_12__T4__flash_duration`: contribution `-0.002728`
- `lag_09__CT_flash_duration_sum`: contribution `-0.002455`

### tick `33186`, seconds `60.50`, LSTM delta `+0.2101`

Top all feature movements:
- `lag_07__CT_place_MAINHALL`: contribution `+0.038478`
- `lag_09__T_place_SIDEHALL`: contribution `+0.022229`
- `lag_00__kill_diff_last_3s`: contribution `+0.012142`
- `lag_13__T2__duck_amount`: contribution `+0.008097`
- `lag_00__CT_kills_last_3s`: contribution `+0.007631`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `31522`, seconds `34.50`, LSTM delta `-0.2066`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.013403`
- `lag_00__kill_diff_last_3s`: contribution `-0.012142`
- `lag_00__T_kills_last_3s`: contribution `-0.011846`
- `lag_07__T_place_MAINHALL`: contribution `-0.011432`
- `lag_15__T_shots_fired_sum`: contribution `-0.010527`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `31490`, seconds `34.00`, LSTM delta `+0.1665`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.012142`
- `lag_15__T_shots_fired_sum`: contribution `+0.010527`
- `lag_00__CT_shots_fired_sum`: contribution `+0.010052`
- `lag_00__CT_kills_last_3s`: contribution `+0.007631`
- `lag_15__CT1__duck_amount`: contribution `+0.006568`

Top utility-only movements:
- No utility movement among the top local contributors.
