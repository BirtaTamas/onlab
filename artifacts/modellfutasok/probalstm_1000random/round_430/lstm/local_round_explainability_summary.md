# Local Round Explainability

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-lynn-vision-vs-tyloo-bo3-tXRL8tbpb2Kb27VbUgWfK1/lynn-vision-vs-tyloo-m2-ancient.csv`
- round_num: `1`

## Largest probability jumps

- tick `9905`, seconds `63.50`, LSTM `0.4818`, delta `+0.3427`
- tick `9361`, seconds `55.00`, LSTM `0.4932`, delta `+0.3217`
- tick `8209`, seconds `37.00`, LSTM `0.2123`, delta `-0.2411`
- tick `9073`, seconds `50.50`, LSTM `0.2922`, delta `-0.2383`
- tick `8273`, seconds `38.00`, LSTM `0.3762`, delta `+0.2010`
- tick `8785`, seconds `46.00`, LSTM `0.7112`, delta `+0.1524`
- tick `9841`, seconds `62.50`, LSTM `0.1767`, delta `-0.1489`
- tick `8817`, seconds `46.50`, LSTM `0.5736`, delta `-0.1376`
- tick `10001`, seconds `65.00`, LSTM `0.5351`, delta `+0.1291`
- tick `9521`, seconds `57.50`, LSTM `0.3486`, delta `-0.1279`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005343`, |coef| `0.005343`
- `lag_00__damage_diff_last_5s`: coefficient `0.005245`, |coef| `0.005245`
- `lag_05__T_bomb_zone_count`: coefficient `-0.003797`, |coef| `0.003797`
- `lag_00__CT_kills_last_3s`: coefficient `0.003773`, |coef| `0.003773`
- `lag_00__CT_defusing_count`: coefficient `0.003752`, |coef| `0.003752`
- `lag_00__CT_duck_amount_mean`: coefficient `0.003331`, |coef| `0.003331`
- `lag_00__CT_damage_last_5s`: coefficient `0.003003`, |coef| `0.003003`
- `lag_00__T_kills_last_3s`: coefficient `-0.002893`, |coef| `0.002893`
- `lag_11__T_duck_amount_mean`: coefficient `-0.002772`, |coef| `0.002772`
- `lag_01__CT_defusing_count`: coefficient `0.002751`, |coef| `0.002751`
- `lag_07__CT_place_ALLEY`: coefficient `-0.002684`, |coef| `0.002684`
- `lag_02__T_kills_last_3s`: coefficient `0.002609`, |coef| `0.002609`
- `lag_00__T_place_ALLEY`: coefficient `-0.002334`, |coef| `0.002334`
- `lag_02__CT_defusing_count`: coefficient `0.002305`, |coef| `0.002305`
- `lag_00__T_damage_last_5s`: coefficient `-0.002272`, |coef| `0.002272`

## Top 10 utility ridge features

- `lag_11__CT2__flash_duration`: coefficient `-0.001816` (lowers CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `0.001634` (raises CT win probability)
- `lag_13__T_B_site_active_smokes`: coefficient `-0.001193` (lowers CT win probability)
- `lag_05__CT_A_site_active_infernos`: coefficient `-0.001156` (lowers CT win probability)
- `lag_05__CT_B_site_active_infernos`: coefficient `-0.001125` (lowers CT win probability)
- `lag_13__T_A_site_active_smokes`: coefficient `-0.001121` (lowers CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `0.001075` (raises CT win probability)
- `lag_14__CT3__flash_duration`: coefficient `-0.001046` (lowers CT win probability)
- `lag_14__CT2__flash_duration`: coefficient `-0.001007` (lowers CT win probability)
- `lag_15__CT2__flash_duration`: coefficient `-0.000974` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005343` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.005245` (raises CT win probability)
- `lag_05__T_bomb_zone_count`: coefficient `-0.003797` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003773` (raises CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.003752` (raises CT win probability)
- `lag_00__CT_duck_amount_mean`: coefficient `0.003331` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.003003` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002893` (lowers CT win probability)
- `lag_11__T_duck_amount_mean`: coefficient `-0.002772` (lowers CT win probability)
- `lag_01__CT_defusing_count`: coefficient `0.002751` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `9905`, seconds `63.50`, LSTM delta `+0.3427`

Top all feature movements:
- `lag_00__CT_duck_amount_mean`: contribution `+0.019946`
- `lag_11__T_duck_amount_mean`: contribution `+0.016120`
- `lag_00__kill_diff_last_3s`: contribution `+0.012860`
- `lag_00__CT_kills_last_3s`: contribution `+0.010892`
- `lag_00__damage_diff_last_5s`: contribution `+0.010413`

Top utility-only movements:
- `lag_11__CT2__flash_duration`: contribution `+0.005803`

### tick `9361`, seconds `55.00`, LSTM delta `+0.3217`

Top all feature movements:
- `lag_05__T_bomb_zone_count`: contribution `+0.022105`
- `lag_00__kill_diff_last_3s`: contribution `+0.012860`
- `lag_00__CT_kills_last_3s`: contribution `+0.010892`
- `lag_14__T_bomb_zone_count`: contribution `+0.010384`
- `lag_00__T_place_ALLEY`: contribution `+0.009888`

Top utility-only movements:
- `lag_00__CT2__flash_duration`: contribution `+0.005222`

### tick `8209`, seconds `37.00`, LSTM delta `-0.2411`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.012860`
- `lag_00__damage_diff_last_5s`: contribution `-0.011833`
- `lag_03__T_place_CTSPAWN`: contribution `-0.010210`
- `lag_00__T_place_ALLEY`: contribution `-0.009888`
- `lag_00__T_kills_last_3s`: contribution `-0.009165`

Top utility-only movements:
- `lag_05__CT_A_site_active_infernos`: contribution `-0.004078`
- `lag_05__CT_B_site_active_infernos`: contribution `-0.003866`

### tick `9073`, seconds `50.50`, LSTM delta `-0.2383`

Top all feature movements:
- `lag_05__T_bomb_zone_count`: contribution `-0.022105`
- `lag_00__damage_diff_last_5s`: contribution `-0.019407`
- `lag_00__kill_diff_last_3s`: contribution `-0.012860`
- `lag_00__T_kills_last_3s`: contribution `-0.009165`
- `lag_02__T_kills_last_3s`: contribution `-0.008267`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `8273`, seconds `38.00`, LSTM delta `+0.2010`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.012860`
- `lag_00__damage_diff_last_5s`: contribution `+0.011833`
- `lag_00__CT_kills_last_3s`: contribution `+0.010892`
- `lag_03__T_place_CTSPAWN`: contribution `+0.010210`
- `lag_02__T_place_HOUSE`: contribution `+0.009055`

Top utility-only movements:
- `lag_07__CT_A_site_active_infernos`: contribution `+0.002751`
- `lag_07__CT_B_site_active_infernos`: contribution `+0.002607`
