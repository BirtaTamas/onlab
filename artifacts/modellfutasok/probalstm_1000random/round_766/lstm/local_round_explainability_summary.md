# Local Round Explainability

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-g2-vs-liquid-bo3-w6HylYj4nF7GNnrWujmZUZ/g2-vs-liquid-m2-inferno.csv`
- round_num: `15`

## Largest probability jumps

- tick `126707`, seconds `58.00`, LSTM `0.6375`, delta `+0.3761`
- tick `128211`, seconds `81.50`, LSTM `0.4880`, delta `-0.3591`
- tick `126675`, seconds `57.50`, LSTM `0.2614`, delta `-0.2790`
- tick `127891`, seconds `76.50`, LSTM `0.7491`, delta `+0.2197`
- tick `128147`, seconds `80.50`, LSTM `0.8272`, delta `+0.2193`
- tick `127923`, seconds `77.00`, LSTM `0.5387`, delta `-0.2104`
- tick `126579`, seconds `56.00`, LSTM `0.4897`, delta `-0.0969`
- tick `127027`, seconds `63.00`, LSTM `0.4379`, delta `-0.0922`
- tick `126739`, seconds `58.50`, LSTM `0.5523`, delta `-0.0852`
- tick `126611`, seconds `56.50`, LSTM `0.5670`, delta `+0.0773`

## Top 15 local ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.003744`, |coef| `0.003744`
- `lag_00__kill_diff_last_3s`: coefficient `0.003163`, |coef| `0.003163`
- `lag_02__CT_shots_fired_sum`: coefficient `-0.002891`, |coef| `0.002891`
- `lag_00__damage_diff_last_5s`: coefficient `0.002882`, |coef| `0.002882`
- `lag_00__T_kills_last_3s`: coefficient `-0.002686`, |coef| `0.002686`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002498`, |coef| `0.002498`
- `lag_01__CT_shots_fired_sum`: coefficient `0.002386`, |coef| `0.002386`
- `lag_08__T_place_QUAD`: coefficient `0.002162`, |coef| `0.002162`
- `lag_14__T_place_QUAD`: coefficient `-0.002135`, |coef| `0.002135`
- `lag_06__T_place_QUAD`: coefficient `-0.002123`, |coef| `0.002123`
- `lag_00__T_place_QUAD`: coefficient `0.001933`, |coef| `0.001933`
- `lag_00__T_duck_amount_mean`: coefficient `-0.001818`, |coef| `0.001818`
- `lag_01__CT4__shots_fired`: coefficient `0.001802`, |coef| `0.001802`
- `lag_09__CT2__flash_duration`: coefficient `0.001799`, |coef| `0.001799`
- `lag_02__CT1__shots_fired`: coefficient `-0.001758`, |coef| `0.001758`

## Top 10 utility ridge features

- `lag_09__CT2__flash_duration`: coefficient `0.001799` (raises CT win probability)
- `lag_11__T5__flash_duration`: coefficient `-0.001087` (lowers CT win probability)
- `lag_00__T5__flash_duration`: coefficient `-0.000990` (lowers CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `0.000989` (raises CT win probability)
- `lag_11__T_flash_duration_sum`: coefficient `-0.000979` (lowers CT win probability)
- `lag_03__CT_A_site_active_infernos`: coefficient `0.000956` (raises CT win probability)
- `lag_10__CT2__flash_duration`: coefficient `-0.000938` (lowers CT win probability)
- `lag_02__T5__flash_duration`: coefficient `0.000915` (raises CT win probability)
- `lag_14__CT_utility_damage_last_5s`: coefficient `-0.000851` (lowers CT win probability)
- `lag_04__CT_A_site_active_infernos`: coefficient `-0.000848` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.003744` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003163` (raises CT win probability)
- `lag_02__CT_shots_fired_sum`: coefficient `-0.002891` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002882` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002686` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002498` (raises CT win probability)
- `lag_01__CT_shots_fired_sum`: coefficient `0.002386` (raises CT win probability)
- `lag_08__T_place_QUAD`: coefficient `0.002162` (raises CT win probability)
- `lag_14__T_place_QUAD`: coefficient `-0.002135` (lowers CT win probability)
- `lag_06__T_place_QUAD`: coefficient `-0.002123` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `126707`, seconds `58.00`, LSTM delta `+0.3761`

Top all feature movements:
- `lag_02__CT_shots_fired_sum`: contribution `+0.026110`
- `lag_00__T_shots_fired_sum`: contribution `+0.025265`
- `lag_00__CT_shots_fired_sum`: contribution `+0.024293`
- `lag_03__T_shots_fired_sum`: contribution `+0.009492`
- `lag_02__CT1__shots_fired`: contribution `+0.008359`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `128211`, seconds `81.50`, LSTM delta `-0.3591`

Top all feature movements:
- `lag_01__CT_shots_fired_sum`: contribution `-0.018232`
- `lag_02__CT_shots_fired_sum`: contribution `-0.016068`
- `lag_00__T_shots_fired_sum`: contribution `-0.014036`
- `lag_00__damage_diff_last_5s`: contribution `-0.011835`
- `lag_01__CT4__shots_fired`: contribution `-0.010679`

Top utility-only movements:
- `lag_09__CT2__flash_duration`: contribution `-0.009598`
- `lag_11__T5__flash_duration`: contribution `-0.005734`
- `lag_02__T5__flash_duration`: contribution `-0.004826`
- `lag_11__T_flash_duration_sum`: contribution `-0.004632`
- `lag_10__CT2__flash_duration`: contribution `-0.003720`

### tick `126675`, seconds `57.50`, LSTM delta `-0.2790`

Top all feature movements:
- `lag_01__CT_shots_fired_sum`: contribution `-0.021547`
- `lag_02__CT_shots_fired_sum`: contribution `-0.018076`
- `lag_00__T_shots_fired_sum`: contribution `-0.014036`
- `lag_00__T_kills_last_3s`: contribution `-0.008508`
- `lag_00__kill_diff_last_3s`: contribution `-0.007612`

Top utility-only movements:
- `lag_14__CT_utility_damage_last_5s`: contribution `-0.003278`
- `lag_04__CT_utility_damage_last_5s`: contribution `-0.002946`

### tick `127891`, seconds `76.50`, LSTM delta `+0.2197`

Top all feature movements:
- `lag_08__T_place_QUAD`: contribution `+0.052082`
- `lag_06__T_place_QUAD`: contribution `+0.051142`
- `lag_00__kill_diff_last_3s`: contribution `+0.007612`
- `lag_15__CT_place_ARCH`: contribution `+0.007323`
- `lag_00__T_shots_fired_sum`: contribution `-0.005614`

Top utility-only movements:
- `lag_00__CT2__flash_duration`: contribution `+0.003922`
- `lag_01__T_flash_duration_sum`: contribution `+0.002184`
- `lag_00__T_flash_duration_sum`: contribution `+0.001801`
- `lag_01__T5__flash_duration`: contribution `+0.001713`

### tick `128147`, seconds `80.50`, LSTM delta `+0.2193`

Top all feature movements:
- `lag_14__T_place_QUAD`: contribution `+0.051415`
- `lag_00__CT_shots_fired_sum`: contribution `+0.013882`
- `lag_00__kill_diff_last_3s`: contribution `+0.007612`
- `lag_00__T5__flash_duration`: contribution `+0.005224`
- `lag_01__CT_shots_fired_sum`: contribution `+0.004972`

Top utility-only movements:
- `lag_00__T5__flash_duration`: contribution `+0.005224`
- `lag_09__T5__flash_duration`: contribution `+0.003727`
- `lag_07__CT2__flash_duration`: contribution `+0.003149`
- `lag_09__CT2__flash_duration`: contribution `+0.002464`
- `lag_09__T_flash_duration_sum`: contribution `+0.002381`
