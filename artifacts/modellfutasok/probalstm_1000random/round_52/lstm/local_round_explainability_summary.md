# Local Round Explainability

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-jijiehao-vs-lynn-vision-bo3-vHZRr1xxhgwfg-A38MzOQQ/jijiehao-vs-lynn-vision-m2-dust2.csv`
- round_num: `11`

## Largest probability jumps

- tick `72252`, seconds `9.50`, LSTM `0.6272`, delta `+0.2578`
- tick `76764`, seconds `80.00`, LSTM `0.8795`, delta `+0.2361`
- tick `72316`, seconds `10.50`, LSTM `0.7567`, delta `+0.1810`
- tick `73116`, seconds `23.00`, LSTM `0.6464`, delta `+0.0520`
- tick `72284`, seconds `10.00`, LSTM `0.5757`, delta `-0.0515`
- tick `72476`, seconds `13.00`, LSTM `0.8316`, delta `+0.0475`
- tick `72668`, seconds `16.00`, LSTM `0.7099`, delta `-0.0467`
- tick `72636`, seconds `15.50`, LSTM `0.7566`, delta `-0.0461`
- tick `76796`, seconds `80.50`, LSTM `0.9237`, delta `+0.0442`
- tick `72124`, seconds `7.50`, LSTM `0.4323`, delta `-0.0431`

## Top 15 local ridge features

- `lag_00__T5__is_scoped`: coefficient `0.002488`, |coef| `0.002488`
- `lag_00__CT_place_BDOORS`: coefficient `-0.002044`, |coef| `0.002044`
- `lag_00__T_place_LONGA`: coefficient `-0.001939`, |coef| `0.001939`
- `lag_12__CT_place_EXTENDEDA`: coefficient `0.001937`, |coef| `0.001937`
- `lag_00__CT_kills_last_3s`: coefficient `0.001933`, |coef| `0.001933`
- `lag_12__CT_place_SHORTSTAIRS`: coefficient `-0.001923`, |coef| `0.001923`
- `lag_00__kill_diff_last_3s`: coefficient `0.001781`, |coef| `0.001781`
- `lag_14__T_place_LONGA`: coefficient `0.001754`, |coef| `0.001754`
- `lag_00__T2__is_walking`: coefficient `-0.001680`, |coef| `0.001680`
- `lag_06__CT_flashed_players`: coefficient `0.001510`, |coef| `0.001510`
- `lag_04__T2__flash_duration`: coefficient `0.001484`, |coef| `0.001484`
- `lag_08__T5__is_scoped`: coefficient `-0.001467`, |coef| `0.001467`
- `lag_03__CT_place_BDOORS`: coefficient `0.001431`, |coef| `0.001431`
- `lag_11__CT_place_UNDERA`: coefficient `-0.001372`, |coef| `0.001372`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001326`, |coef| `0.001326`

## Top 10 utility ridge features

- `lag_04__T2__flash_duration`: coefficient `0.001484` (raises CT win probability)
- `lag_04__T4__flash_duration`: coefficient `0.001166` (raises CT win probability)
- `lag_04__T_flash_duration_sum`: coefficient `0.001124` (raises CT win probability)
- `lag_00__CT_flashes_last_5s`: coefficient `-0.001043` (lowers CT win probability)
- `lag_00__T4__molly`: coefficient `-0.001030` (lowers CT win probability)
- `lag_06__CT_flash_duration_sum`: coefficient `0.000986` (raises CT win probability)
- `lag_00__T_molly_inv`: coefficient `-0.000944` (lowers CT win probability)
- `lag_06__CT3__flash_duration`: coefficient `0.000927` (raises CT win probability)
- `lag_00__molly_inv_diff`: coefficient `0.000924` (raises CT win probability)
- `lag_00__T2__molly`: coefficient `-0.000884` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T5__is_scoped`: coefficient `0.002488` (raises CT win probability)
- `lag_00__CT_place_BDOORS`: coefficient `-0.002044` (lowers CT win probability)
- `lag_00__T_place_LONGA`: coefficient `-0.001939` (lowers CT win probability)
- `lag_12__CT_place_EXTENDEDA`: coefficient `0.001937` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001933` (raises CT win probability)
- `lag_12__CT_place_SHORTSTAIRS`: coefficient `-0.001923` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001781` (raises CT win probability)
- `lag_14__T_place_LONGA`: coefficient `0.001754` (raises CT win probability)
- `lag_00__T2__is_walking`: coefficient `-0.001680` (lowers CT win probability)
- `lag_06__CT_flashed_players`: coefficient `0.001510` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `72252`, seconds `9.50`, LSTM delta `+0.2578`

Top all feature movements:
- `lag_00__T5__is_scoped`: contribution `+0.011866`
- `lag_04__T2__flash_duration`: contribution `+0.010079`
- `lag_06__CT_flashed_players`: contribution `+0.009922`
- `lag_00__CT_place_BDOORS`: contribution `+0.009834`
- `lag_04__T4__flash_duration`: contribution `+0.006997`

Top utility-only movements:
- `lag_04__T2__flash_duration`: contribution `+0.010079`
- `lag_04__T4__flash_duration`: contribution `+0.006997`
- `lag_04__T_flash_duration_sum`: contribution `+0.006510`
- `lag_06__CT3__flash_duration`: contribution `+0.004788`
- `lag_06__CT_flash_duration_sum`: contribution `+0.003668`

### tick `76764`, seconds `80.00`, LSTM delta `+0.2361`

Top all feature movements:
- `lag_00__T5__is_scoped`: contribution `+0.011866`
- `lag_12__CT_place_EXTENDEDA`: contribution `+0.010875`
- `lag_12__CT_place_SHORTSTAIRS`: contribution `+0.010722`
- `lag_00__T_place_LONGA`: contribution `+0.008260`
- `lag_14__T_place_LONGA`: contribution `+0.007471`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `72316`, seconds `10.50`, LSTM delta `+0.1810`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.005580`
- `lag_06__T2__flash_duration`: contribution `+0.005308`
- `lag_00__CT_shots_fired_sum`: contribution `+0.004606`
- `lag_06__T4__flash_duration`: contribution `+0.004382`
- `lag_00__kill_diff_last_3s`: contribution `+0.004288`

Top utility-only movements:
- `lag_06__T2__flash_duration`: contribution `+0.005308`
- `lag_06__T4__flash_duration`: contribution `+0.004382`
- `lag_06__T_flash_duration_sum`: contribution `+0.003899`
- `lag_02__T3__flash_duration`: contribution `+0.003515`
- `lag_08__CT3__flash_duration`: contribution `+0.003257`

### tick `73116`, seconds `23.00`, LSTM delta `+0.0520`

Top all feature movements:
- `lag_03__CT_place_BDOORS`: contribution `+0.006884`
- `lag_00__T2__is_walking`: contribution `+0.003859`
- `lag_10__CT4__flash_duration`: contribution `+0.003446`
- `lag_08__T4__is_walking`: contribution `+0.002103`
- `lag_01__CT_active_infernos`: contribution `-0.001948`

Top utility-only movements:
- `lag_10__CT4__flash_duration`: contribution `+0.003446`
- `lag_01__CT_active_infernos`: contribution `-0.001948`
- `lag_14__CT2__flash_duration`: contribution `+0.001646`
- `lag_15__CT_active_infernos`: contribution `+0.001044`

### tick `72284`, seconds `10.00`, LSTM delta `-0.0515`

Top all feature movements:
- `lag_00__T5__is_scoped`: contribution `-0.011866`
- `lag_00__CT_shots_fired_sum`: contribution `+0.004606`
- `lag_00__kill_diff_last_3s`: contribution `-0.004288`
- `lag_15__T_place_OUTSIDELONG`: contribution `+0.003498`
- `lag_01__CT_shots_fired_sum`: contribution `+0.003386`

Top utility-only movements:
- `lag_05__T4__flash_duration`: contribution `-0.002066`
- `lag_05__CT2__flash_duration`: contribution `-0.001945`
- `lag_00__T3__molly`: contribution `-0.001856`
- `lag_01__T3__flash_duration`: contribution `-0.001688`
- `lag_01__T_flash_duration_sum`: contribution `-0.001596`
