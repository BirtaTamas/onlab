# Local Round Explainability

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-g2-vs-liquid-bo3-w6HylYj4nF7GNnrWujmZUZ/g2-vs-liquid-m2-inferno.csv`
- round_num: `4`

## Largest probability jumps

- tick `34094`, seconds `37.50`, LSTM `0.3217`, delta `-0.1307`
- tick `33102`, seconds `22.00`, LSTM `0.4240`, delta `-0.1245`
- tick `33294`, seconds `25.00`, LSTM `0.5973`, delta `+0.1065`
- tick `33134`, seconds `22.50`, LSTM `0.5118`, delta `+0.0878`
- tick `33326`, seconds `25.50`, LSTM `0.6722`, delta `+0.0749`
- tick `32590`, seconds `14.00`, LSTM `0.4123`, delta `+0.0740`
- tick `32942`, seconds `19.50`, LSTM `0.5508`, delta `+0.0628`
- tick `34126`, seconds `38.00`, LSTM `0.2641`, delta `-0.0576`
- tick `34350`, seconds `41.50`, LSTM `0.2013`, delta `-0.0496`
- tick `32302`, seconds `9.50`, LSTM `0.2954`, delta `-0.0468`

## Top 15 local ridge features

- `lag_00__T1__flash_duration`: coefficient `0.001700`, |coef| `0.001700`
- `lag_09__CT_place_RUINS`: coefficient `-0.001483`, |coef| `0.001483`
- `lag_04__CT_place_QUAD`: coefficient `0.001295`, |coef| `0.001295`
- `lag_11__CT_place_QUAD`: coefficient `0.001276`, |coef| `0.001276`
- `lag_05__CT_place_QUAD`: coefficient `0.001126`, |coef| `0.001126`
- `lag_01__T4__flash_duration`: coefficient `0.001055`, |coef| `0.001055`
- `lag_03__CT_place_ARCH`: coefficient `-0.001052`, |coef| `0.001052`
- `lag_02__CT_place_ARCH`: coefficient `-0.001049`, |coef| `0.001049`
- `lag_09__CT_place_QUAD`: coefficient `0.001035`, |coef| `0.001035`
- `lag_00__CT_place_RUINS`: coefficient `0.000998`, |coef| `0.000998`
- `lag_00__CT_shots_fired_sum`: coefficient `0.000980`, |coef| `0.000980`
- `lag_11__T_shots_fired_sum`: coefficient `-0.000972`, |coef| `0.000972`
- `lag_12__T1__duck_amount`: coefficient `-0.000969`, |coef| `0.000969`
- `lag_14__T_B_site_active_infernos`: coefficient `0.000942`, |coef| `0.000942`
- `lag_00__T_bomb_zone_count`: coefficient `0.000935`, |coef| `0.000935`

## Top 10 utility ridge features

- `lag_00__T1__flash_duration`: coefficient `0.001700` (raises CT win probability)
- `lag_01__T4__flash_duration`: coefficient `0.001055` (raises CT win probability)
- `lag_14__T_B_site_active_infernos`: coefficient `0.000942` (raises CT win probability)
- `lag_08__T1__flash_duration`: coefficient `0.000935` (raises CT win probability)
- `lag_14__T_active_infernos`: coefficient `0.000912` (raises CT win probability)
- `lag_03__CT4__flash_duration`: coefficient `-0.000901` (lowers CT win probability)
- `lag_12__T_utility_damage_last_5s`: coefficient `0.000868` (raises CT win probability)
- `lag_11__T4__flash_duration`: coefficient `0.000845` (raises CT win probability)
- `lag_10__T3__flash_duration`: coefficient `-0.000798` (lowers CT win probability)
- `lag_01__CT_B_site_active_infernos`: coefficient `0.000789` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_09__CT_place_RUINS`: coefficient `-0.001483` (lowers CT win probability)
- `lag_04__CT_place_QUAD`: coefficient `0.001295` (raises CT win probability)
- `lag_11__CT_place_QUAD`: coefficient `0.001276` (raises CT win probability)
- `lag_05__CT_place_QUAD`: coefficient `0.001126` (raises CT win probability)
- `lag_03__CT_place_ARCH`: coefficient `-0.001052` (lowers CT win probability)
- `lag_02__CT_place_ARCH`: coefficient `-0.001049` (lowers CT win probability)
- `lag_09__CT_place_QUAD`: coefficient `0.001035` (raises CT win probability)
- `lag_00__CT_place_RUINS`: coefficient `0.000998` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.000980` (raises CT win probability)
- `lag_11__T_shots_fired_sum`: coefficient `-0.000972` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `34094`, seconds `37.50`, LSTM delta `-0.1307`

Top all feature movements:
- `lag_00__T1__flash_duration`: contribution `-0.012074`
- `lag_04__CT_place_QUAD`: contribution `-0.010203`
- `lag_01__T4__flash_duration`: contribution `-0.006826`
- `lag_00__T_bomb_zone_count`: contribution `-0.005444`
- `lag_09__CT_place_RUINS`: contribution `-0.005181`

Top utility-only movements:
- `lag_00__T1__flash_duration`: contribution `-0.012074`
- `lag_01__T4__flash_duration`: contribution `-0.006826`
- `lag_13__T1__flash_duration`: contribution `-0.003828`
- `lag_15__T_utility_damage_last_5s`: contribution `-0.002090`
- `lag_00__T_flash_duration_sum`: contribution `-0.001689`

### tick `33102`, seconds `22.00`, LSTM delta `-0.1245`

Top all feature movements:
- `lag_10__T_flash_duration_sum`: contribution `-0.006799`
- `lag_03__CT4__flash_duration`: contribution `-0.006709`
- `lag_10__T3__flash_duration`: contribution `-0.004608`
- `lag_03__T4__flash_duration`: contribution `-0.004023`
- `lag_11__CT_place_ARCH`: contribution `-0.003807`

Top utility-only movements:
- `lag_10__T_flash_duration_sum`: contribution `-0.006799`
- `lag_03__CT4__flash_duration`: contribution `-0.006709`
- `lag_10__T3__flash_duration`: contribution `-0.004608`
- `lag_03__T4__flash_duration`: contribution `-0.004023`
- `lag_10__T2__flash_duration`: contribution `-0.003759`

### tick `33294`, seconds `25.00`, LSTM delta `+0.1065`

Top all feature movements:
- `lag_00__CT_place_QUAD`: contribution `-0.007048`
- `lag_11__T_shots_fired_sum`: contribution `+0.006562`
- `lag_03__T2__flash_duration`: contribution `+0.005212`
- `lag_09__CT4__flash_duration`: contribution `+0.005141`
- `lag_02__CT_place_ARCH`: contribution `+0.004281`

Top utility-only movements:
- `lag_03__T2__flash_duration`: contribution `+0.005212`
- `lag_09__CT4__flash_duration`: contribution `+0.005141`
- `lag_06__CT4__flash_duration`: contribution `+0.004272`
- `lag_09__T3__flash_duration`: contribution `+0.002982`
- `lag_01__CT_B_site_active_infernos`: contribution `+0.002712`

### tick `33134`, seconds `22.50`, LSTM delta `+0.0878`

Top all feature movements:
- `lag_11__T4__flash_duration`: contribution `+0.005925`
- `lag_06__T_shots_fired_sum`: contribution `+0.005668`
- `lag_11__T_flash_duration_sum`: contribution `+0.005521`
- `lag_00__CT_shots_fired_sum`: contribution `+0.004767`
- `lag_11__T_flashed_players`: contribution `+0.004289`

Top utility-only movements:
- `lag_11__T4__flash_duration`: contribution `+0.005925`
- `lag_11__T_flash_duration_sum`: contribution `+0.005521`
- `lag_04__CT4__flash_duration`: contribution `+0.003600`
- `lag_11__T2__flash_duration`: contribution `+0.003487`
- `lag_01__CT4__flash_duration`: contribution `+0.002915`

### tick `33326`, seconds `25.50`, LSTM delta `+0.0749`

Top all feature movements:
- `lag_06__T_shots_fired_sum`: contribution `+0.006298`
- `lag_07__CT4__flash_duration`: contribution `+0.004448`
- `lag_03__CT_place_ARCH`: contribution `+0.004292`
- `lag_00__CT_shots_fired_sum`: contribution `+0.004086`
- `lag_10__T3__flash_duration`: contribution `+0.004008`

Top utility-only movements:
- `lag_07__CT4__flash_duration`: contribution `+0.004448`
- `lag_10__T3__flash_duration`: contribution `+0.004008`
- `lag_04__T2__flash_duration`: contribution `+0.003416`
- `lag_14__T_B_site_active_infernos`: contribution `+0.002663`
- `lag_10__T_flash_duration_sum`: contribution `+0.002593`
