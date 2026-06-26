# Local Round Explainability

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-jijiehao-vs-lynn-vision-bo3-vHZRr1xxhgwfg-A38MzOQQ/jijiehao-vs-lynn-vision-m2-dust2.csv`
- round_num: `5`

## Largest probability jumps

- tick `27114`, seconds `25.50`, LSTM `0.1600`, delta `-0.3211`
- tick `26922`, seconds `22.50`, LSTM `0.4201`, delta `+0.2031`
- tick `27434`, seconds `30.50`, LSTM `0.0255`, delta `-0.0899`
- tick `25514`, seconds `0.50`, LSTM `0.1076`, delta `-0.0656`
- tick `26986`, seconds `23.50`, LSTM `0.5430`, delta `+0.0646`
- tick `26954`, seconds `23.00`, LSTM `0.4784`, delta `+0.0584`
- tick `26826`, seconds `21.00`, LSTM `0.2472`, delta `+0.0397`
- tick `26058`, seconds `9.00`, LSTM `0.1786`, delta `+0.0367`
- tick `27018`, seconds `24.00`, LSTM `0.5079`, delta `-0.0352`
- tick `27146`, seconds `26.00`, LSTM `0.1287`, delta `-0.0313`

## Top 15 local ridge features

- `lag_15__CT1__duck_amount`: coefficient `0.001872`, |coef| `0.001872`
- `lag_00__kill_diff_last_3s`: coefficient `0.001786`, |coef| `0.001786`
- `lag_00__CT_place_LOWERTUNNEL`: coefficient `0.001762`, |coef| `0.001762`
- `lag_03__CT1__flash_duration`: coefficient `-0.001675`, |coef| `0.001675`
- `lag_00__T_flashed_players`: coefficient `-0.001667`, |coef| `0.001667`
- `lag_09__CT1__flash_duration`: coefficient `0.001662`, |coef| `0.001662`
- `lag_03__T_shots_fired_sum`: coefficient `0.001609`, |coef| `0.001609`
- `lag_05__CT_place_UPPERTUNNEL`: coefficient `-0.001564`, |coef| `0.001564`
- `lag_06__T_flashed_players`: coefficient `0.001537`, |coef| `0.001537`
- `lag_08__T_flashed_players`: coefficient `-0.001510`, |coef| `0.001510`
- `lag_12__CT3__flash_duration`: coefficient `0.001487`, |coef| `0.001487`
- `lag_03__T_place_TUNNELSTAIRS`: coefficient `0.001472`, |coef| `0.001472`
- `lag_09__T2__duck_amount`: coefficient `-0.001458`, |coef| `0.001458`
- `lag_03__T2__shots_fired`: coefficient `0.001421`, |coef| `0.001421`
- `lag_00__CT_kills_last_3s`: coefficient `0.001402`, |coef| `0.001402`

## Top 10 utility ridge features

- `lag_03__CT1__flash_duration`: coefficient `-0.001675` (lowers CT win probability)
- `lag_09__CT1__flash_duration`: coefficient `0.001662` (raises CT win probability)
- `lag_12__CT3__flash_duration`: coefficient `0.001487` (raises CT win probability)
- `lag_06__CT3__flash_duration`: coefficient `-0.001091` (lowers CT win probability)
- `lag_00__CT_flashes_last_5s`: coefficient `-0.000999` (lowers CT win probability)
- `lag_12__CT_flash_duration_sum`: coefficient `0.000762` (raises CT win probability)
- `lag_03__CT_flash_duration_sum`: coefficient `-0.000747` (lowers CT win probability)
- `lag_06__T3__utility_total`: coefficient `0.000705` (raises CT win probability)
- `lag_09__CT_flash_duration_sum`: coefficient `0.000652` (raises CT win probability)
- `lag_07__CT3__flash_duration`: coefficient `-0.000642` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_15__CT1__duck_amount`: coefficient `0.001872` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001786` (raises CT win probability)
- `lag_00__CT_place_LOWERTUNNEL`: coefficient `0.001762` (raises CT win probability)
- `lag_00__T_flashed_players`: coefficient `-0.001667` (lowers CT win probability)
- `lag_03__T_shots_fired_sum`: coefficient `0.001609` (raises CT win probability)
- `lag_05__CT_place_UPPERTUNNEL`: coefficient `-0.001564` (lowers CT win probability)
- `lag_06__T_flashed_players`: coefficient `0.001537` (raises CT win probability)
- `lag_08__T_flashed_players`: coefficient `-0.001510` (lowers CT win probability)
- `lag_03__T_place_TUNNELSTAIRS`: coefficient `0.001472` (raises CT win probability)
- `lag_09__T2__duck_amount`: coefficient `-0.001458` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `27114`, seconds `25.50`, LSTM delta `-0.3211`

Top all feature movements:
- `lag_03__T_shots_fired_sum`: contribution `-0.014479`
- `lag_09__CT1__flash_duration`: contribution `-0.013677`
- `lag_00__CT_place_LOWERTUNNEL`: contribution `-0.012954`
- `lag_05__CT_place_UPPERTUNNEL`: contribution `-0.011995`
- `lag_06__T_flashed_players`: contribution `-0.011863`

Top utility-only movements:
- `lag_09__CT1__flash_duration`: contribution `-0.013677`
- `lag_12__CT3__flash_duration`: contribution `-0.009194`

### tick `26922`, seconds `22.50`, LSTM delta `+0.2031`

Top all feature movements:
- `lag_03__CT1__flash_duration`: contribution `+0.013779`
- `lag_00__T_flashed_players`: contribution `+0.012868`
- `lag_02__T_flashed_players`: contribution `+0.009831`
- `lag_00__T_place_TUNNELSTAIRS`: contribution `+0.008531`
- `lag_15__CT1__duck_amount`: contribution `+0.007141`

Top utility-only movements:
- `lag_03__CT1__flash_duration`: contribution `+0.013779`
- `lag_06__CT3__flash_duration`: contribution `+0.006745`
- `lag_03__CT_flash_duration_sum`: contribution `+0.002763`

### tick `27434`, seconds `30.50`, LSTM delta `-0.0899`

Top all feature movements:
- `lag_02__T_place_TUNNELSTAIRS`: contribution `-0.008095`
- `lag_00__CT_place_UPPERTUNNEL`: contribution `-0.007767`
- `lag_03__T_shots_fired_sum`: contribution `+0.004826`
- `lag_13__T_shots_fired_sum`: contribution `-0.004471`
- `lag_15__CT_place_UPPERTUNNEL`: contribution `-0.004368`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `25514`, seconds `0.50`, LSTM delta `-0.0656`

Top all feature movements:
- `lag_00__CT_flashes_last_5s`: contribution `-0.010984`
- `lag_01__CT_place_CTSPAWN`: contribution `-0.002565`
- `lag_01__T_place_TSPAWN`: contribution `-0.002173`
- `lag_00__T_velocity_mean`: contribution `-0.001911`
- `lag_00__CT_velocity_mean`: contribution `-0.001429`

Top utility-only movements:
- `lag_00__CT_flashes_last_5s`: contribution `-0.010984`
- `lag_01__molly_inv_diff`: contribution `-0.001031`
- `lag_01__utility_inv_diff`: contribution `-0.000759`
- `lag_01__T_molly_inv`: contribution `-0.000664`
- `lag_01__T_smoke_inv`: contribution `-0.000656`

### tick `26986`, seconds `23.50`, LSTM delta `+0.0646`

Top all feature movements:
- `lag_02__T_flashed_players`: contribution `-0.009831`
- `lag_02__T_place_TUNNELSTAIRS`: contribution `+0.008095`
- `lag_04__T_flashed_players`: contribution `+0.007508`
- `lag_04__T5__duck_amount`: contribution `+0.004500`
- `lag_08__CT3__flash_duration`: contribution `+0.003117`

Top utility-only movements:
- `lag_08__CT3__flash_duration`: contribution `+0.003117`
- `lag_05__CT1__flash_duration`: contribution `+0.001739`
