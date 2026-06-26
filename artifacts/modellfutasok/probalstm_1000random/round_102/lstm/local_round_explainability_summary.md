# Local Round Explainability

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-tyloo-vs-rare-atom-bo3-8GB1HWZtKOlh9_707n2A62/tyloo-vs-rare-atom-m2-inferno.csv`
- round_num: `9`

## Largest probability jumps

- tick `73859`, seconds `55.00`, LSTM `0.8087`, delta `+0.1769`
- tick `73891`, seconds `55.50`, LSTM `0.9430`, delta `+0.1343`
- tick `71619`, seconds `20.00`, LSTM `0.5308`, delta `-0.0816`
- tick `73635`, seconds `51.50`, LSTM `0.5989`, delta `+0.0805`
- tick `73603`, seconds `51.00`, LSTM `0.5184`, delta `+0.0724`
- tick `70531`, seconds `3.00`, LSTM `0.6306`, delta `-0.0698`
- tick `73827`, seconds `54.50`, LSTM `0.6318`, delta `+0.0613`
- tick `73539`, seconds `50.00`, LSTM `0.4456`, delta `-0.0511`
- tick `71843`, seconds `23.50`, LSTM `0.3870`, delta `-0.0469`
- tick `71651`, seconds `20.50`, LSTM `0.4841`, delta `-0.0467`

## Top 15 local ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.001461`, |coef| `0.001461`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001306`, |coef| `0.001306`
- `lag_02__CT4__flash_duration`: coefficient `-0.001210`, |coef| `0.001210`
- `lag_08__T_flash_duration_sum`: coefficient `0.001125`, |coef| `0.001125`
- `lag_08__T5__flash_duration`: coefficient `0.001116`, |coef| `0.001116`
- `lag_08__T3__flash_duration`: coefficient `0.001063`, |coef| `0.001063`
- `lag_08__CT_shots_fired_sum`: coefficient `0.001022`, |coef| `0.001022`
- `lag_05__CT_place_RUINS`: coefficient `-0.000996`, |coef| `0.000996`
- `lag_00__CT_kills_last_3s`: coefficient `0.000990`, |coef| `0.000990`
- `lag_06__T4__flash_duration`: coefficient `-0.000969`, |coef| `0.000969`
- `lag_08__T_shots_fired_sum`: coefficient `-0.000957`, |coef| `0.000957`
- `lag_10__CT_place_BALCONY`: coefficient `-0.000948`, |coef| `0.000948`
- `lag_01__CT_shots_fired_sum`: coefficient `0.000944`, |coef| `0.000944`
- `lag_07__T4__flash_duration`: coefficient `-0.000920`, |coef| `0.000920`
- `lag_13__CT_flashes_last_5s`: coefficient `-0.000914`, |coef| `0.000914`

## Top 10 utility ridge features

- `lag_02__CT4__flash_duration`: coefficient `-0.001210` (lowers CT win probability)
- `lag_08__T_flash_duration_sum`: coefficient `0.001125` (raises CT win probability)
- `lag_08__T5__flash_duration`: coefficient `0.001116` (raises CT win probability)
- `lag_08__T3__flash_duration`: coefficient `0.001063` (raises CT win probability)
- `lag_06__T4__flash_duration`: coefficient `-0.000969` (lowers CT win probability)
- `lag_07__T4__flash_duration`: coefficient `-0.000920` (lowers CT win probability)
- `lag_13__CT_flashes_last_5s`: coefficient `-0.000914` (lowers CT win probability)
- `lag_09__T_flash_duration_sum`: coefficient `0.000903` (raises CT win probability)
- `lag_00__T_mollies_last_5s`: coefficient `-0.000869` (lowers CT win probability)
- `lag_09__T3__flash_duration`: coefficient `0.000853` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.001461` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001306` (lowers CT win probability)
- `lag_08__CT_shots_fired_sum`: coefficient `0.001022` (raises CT win probability)
- `lag_05__CT_place_RUINS`: coefficient `-0.000996` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000990` (raises CT win probability)
- `lag_08__T_shots_fired_sum`: coefficient `-0.000957` (lowers CT win probability)
- `lag_10__CT_place_BALCONY`: coefficient `-0.000948` (lowers CT win probability)
- `lag_01__CT_shots_fired_sum`: coefficient `0.000944` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000898` (raises CT win probability)
- `lag_10__T1__shots_fired`: coefficient `0.000817` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `73859`, seconds `55.00`, LSTM delta `+0.1769`

Top all feature movements:
- `lag_08__T_flash_duration_sum`: contribution `+0.011610`
- `lag_08__T5__flash_duration`: contribution `+0.008156`
- `lag_07__T4__flash_duration`: contribution `+0.007295`
- `lag_08__T3__flash_duration`: contribution `+0.007293`
- `lag_02__CT4__flash_duration`: contribution `+0.007009`

Top utility-only movements:
- `lag_08__T_flash_duration_sum`: contribution `+0.011610`
- `lag_08__T5__flash_duration`: contribution `+0.008156`
- `lag_07__T4__flash_duration`: contribution `+0.007295`
- `lag_08__T3__flash_duration`: contribution `+0.007293`
- `lag_02__CT4__flash_duration`: contribution `+0.007009`

### tick `73891`, seconds `55.50`, LSTM delta `+0.1343`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.011166`
- `lag_09__T_flash_duration_sum`: contribution `+0.009323`
- `lag_08__CT_shots_fired_sum`: contribution `+0.007104`
- `lag_09__T5__flash_duration`: contribution `+0.006210`
- `lag_09__T3__flash_duration`: contribution `+0.005853`

Top utility-only movements:
- `lag_09__T_flash_duration_sum`: contribution `+0.009323`
- `lag_09__T5__flash_duration`: contribution `+0.006210`
- `lag_09__T3__flash_duration`: contribution `+0.005853`
- `lag_08__T_flash_duration_sum`: contribution `-0.003626`
- `lag_13__CT4__flash_duration`: contribution `+0.002829`

### tick `71619`, seconds `20.00`, LSTM delta `-0.0816`

Top all feature movements:
- `lag_10__CT_place_BALCONY`: contribution `-0.006082`
- `lag_00__T_shots_fired_sum`: contribution `-0.005877`
- `lag_15__CT_place_QUAD`: contribution `-0.005402`
- `lag_08__CT_place_BALCONY`: contribution `-0.003442`
- `lag_02__T_flashed_players`: contribution `-0.002721`

Top utility-only movements:
- `lag_07__CT_utility_damage_last_5s`: contribution `-0.002562`
- `lag_02__T4__flash_duration`: contribution `-0.002224`
- `lag_07__utility_damage_diff_last_5s`: contribution `-0.001655`
- `lag_00__T_B_site_active_infernos`: contribution `-0.001620`
- `lag_14__CT_A_site_active_infernos`: contribution `-0.001203`

### tick `73635`, seconds `51.50`, LSTM delta `+0.0805`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.010151`
- `lag_08__T_flashed_players`: contribution `+0.004027`
- `lag_01__CT_shots_fired_sum`: contribution `+0.003278`
- `lag_01__T3__flash_duration`: contribution `+0.003125`
- `lag_01__T5__flash_duration`: contribution `+0.002878`

Top utility-only movements:
- `lag_01__T3__flash_duration`: contribution `+0.003125`
- `lag_01__T5__flash_duration`: contribution `+0.002878`
- `lag_01__T_flash_duration_sum`: contribution `+0.002705`
- `lag_08__T_flash_duration_sum`: contribution `+0.002062`
- `lag_08__T3__flash_duration`: contribution `+0.001412`

### tick `73603`, seconds `51.00`, LSTM delta `+0.0724`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.008816`
- `lag_00__CT_shots_fired_sum`: contribution `+0.005075`
- `lag_00__T3__flash_duration`: contribution `+0.004404`
- `lag_07__T_flashed_players`: contribution `+0.003427`
- `lag_00__CT_kills_last_3s`: contribution `+0.002857`

Top utility-only movements:
- `lag_00__T3__flash_duration`: contribution `+0.004404`
- `lag_04__CT4__flash_duration`: contribution `+0.002187`
- `lag_07__T4__flash_duration`: contribution `-0.001431`
- `lag_00__T2__flash_duration`: contribution `-0.001377`
- `lag_00__T_flash_duration_sum`: contribution `+0.001248`
