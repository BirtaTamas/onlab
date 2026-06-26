# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-passion-ua-vs-spirit-bo3-WimU0hRkNcqhh3KAjCozBx/passion-ua-vs-spirit-m2-mirage.csv`
- round_num: `1`

## Largest probability jumps

- tick `9678`, seconds `35.00`, LSTM `0.1162`, delta `-0.1774`
- tick `9006`, seconds `24.50`, LSTM `0.6509`, delta `+0.1468`
- tick `10158`, seconds `42.50`, LSTM `0.0237`, delta `-0.1331`
- tick `9390`, seconds `30.50`, LSTM `0.3936`, delta `-0.0949`
- tick `9902`, seconds `38.50`, LSTM `0.1011`, delta `+0.0812`
- tick `9358`, seconds `30.00`, LSTM `0.4885`, delta `-0.0718`
- tick `9422`, seconds `31.00`, LSTM `0.3257`, delta `-0.0679`
- tick `9198`, seconds `27.50`, LSTM `0.5942`, delta `-0.0517`
- tick `9582`, seconds `33.50`, LSTM `0.2633`, delta `-0.0514`
- tick `9998`, seconds `40.00`, LSTM `0.1379`, delta `+0.0488`

## Top 15 local ridge features

- `lag_15__CT_place_BACKALLEY`: coefficient `-0.002297`, |coef| `0.002297`
- `lag_12__T_place_LADDER`: coefficient `0.001645`, |coef| `0.001645`
- `lag_07__CT_place_SIDEALLEY`: coefficient `0.001595`, |coef| `0.001595`
- `lag_00__CT_place_SIDEALLEY`: coefficient `-0.001538`, |coef| `0.001538`
- `lag_04__CT_place_SIDEALLEY`: coefficient `-0.001269`, |coef| `0.001269`
- `lag_15__CT_place_SIDEALLEY`: coefficient `-0.001250`, |coef| `0.001250`
- `lag_00__T_kills_last_3s`: coefficient `-0.001148`, |coef| `0.001148`
- `lag_00__kill_diff_last_3s`: coefficient `0.000992`, |coef| `0.000992`
- `lag_08__CT_place_BACKALLEY`: coefficient `0.000974`, |coef| `0.000974`
- `lag_06__T_place_SCAFFOLDING`: coefficient `0.000929`, |coef| `0.000929`
- `lag_02__CT_place_SIDEALLEY`: coefficient `-0.000920`, |coef| `0.000920`
- `lag_06__CT_place_BACKALLEY`: coefficient `-0.000850`, |coef| `0.000850`
- `lag_01__T_place_SCAFFOLDING`: coefficient `-0.000829`, |coef| `0.000829`
- `lag_00__CT5__duck_amount`: coefficient `-0.000819`, |coef| `0.000819`
- `lag_03__CT_place_BACKALLEY`: coefficient `-0.000818`, |coef| `0.000818`

## Top 10 utility ridge features

- `lag_04__CT_flash_alpha_mean`: coefficient `0.000770` (raises CT win probability)
- `lag_12__CT_flash_alpha_mean`: coefficient `0.000674` (raises CT win probability)
- `lag_11__CT_flash_alpha_mean`: coefficient `0.000650` (raises CT win probability)
- `lag_05__CT_flash_alpha_mean`: coefficient `0.000624` (raises CT win probability)
- `lag_13__CT_flash_alpha_mean`: coefficient `0.000607` (raises CT win probability)
- `lag_15__CT_flash_alpha_mean`: coefficient `0.000561` (raises CT win probability)
- `lag_01__CT_flash_alpha_mean`: coefficient `0.000509` (raises CT win probability)
- `lag_07__CT_flash_alpha_mean`: coefficient `0.000453` (raises CT win probability)
- `lag_09__CT_flash_alpha_mean`: coefficient `0.000448` (raises CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.000444` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_15__CT_place_BACKALLEY`: coefficient `-0.002297` (lowers CT win probability)
- `lag_12__T_place_LADDER`: coefficient `0.001645` (raises CT win probability)
- `lag_07__CT_place_SIDEALLEY`: coefficient `0.001595` (raises CT win probability)
- `lag_00__CT_place_SIDEALLEY`: coefficient `-0.001538` (lowers CT win probability)
- `lag_04__CT_place_SIDEALLEY`: coefficient `-0.001269` (lowers CT win probability)
- `lag_15__CT_place_SIDEALLEY`: coefficient `-0.001250` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001148` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000992` (raises CT win probability)
- `lag_08__CT_place_BACKALLEY`: coefficient `0.000974` (raises CT win probability)
- `lag_06__T_place_SCAFFOLDING`: coefficient `0.000929` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `9678`, seconds `35.00`, LSTM delta `-0.1774`

Top all feature movements:
- `lag_12__T_place_LADDER`: contribution `-0.037184`
- `lag_15__CT_place_BACKALLEY`: contribution `-0.034439`
- `lag_00__CT_place_SIDEALLEY`: contribution `-0.028067`
- `lag_08__CT_place_BACKALLEY`: contribution `-0.014603`
- `lag_00__T_kills_last_3s`: contribution `-0.003638`

Top utility-only movements:
- `lag_15__CT5__flash_duration`: contribution `-0.001678`

### tick `9006`, seconds `24.50`, LSTM delta `+0.1468`

Top all feature movements:
- `lag_01__T_place_SCAFFOLDING`: contribution `+0.056497`
- `lag_03__T_place_SCAFFOLDING`: contribution `+0.021670`
- `lag_07__T_place_LADDER`: contribution `+0.016575`
- `lag_04__T_place_SCAFFOLDING`: contribution `+0.014849`
- `lag_00__CT_place_STAIRS`: contribution `+0.004308`

Top utility-only movements:
- `lag_05__CT_flash_alpha_mean`: contribution `+0.001915`
- `lag_05__CT2__flash_duration`: contribution `+0.001786`
- `lag_05__CT5__flash_duration`: contribution `+0.001717`
- `lag_05__CT_flash_duration_sum`: contribution `+0.001489`

### tick `10158`, seconds `42.50`, LSTM delta `-0.1331`

Top all feature movements:
- `lag_07__CT_place_SIDEALLEY`: contribution `-0.029097`
- `lag_15__CT_place_SIDEALLEY`: contribution `-0.022809`
- `lag_02__T_place_SCAFFOLDING`: contribution `-0.015537`
- `lag_00__T_kills_last_3s`: contribution `-0.003638`
- `lag_11__CT_place_JUNGLE`: contribution `-0.003510`

Top utility-only movements:
- `lag_11__CT_flash_alpha_mean`: contribution `-0.001665`

### tick `9390`, seconds `30.50`, LSTM delta `-0.0949`

Top all feature movements:
- `lag_13__T_place_SCAFFOLDING`: contribution `-0.034800`
- `lag_03__T_place_LADDER`: contribution `-0.013511`
- `lag_06__CT_place_BACKALLEY`: contribution `-0.012743`
- `lag_12__CT_place_STAIRS`: contribution `-0.004394`
- `lag_15__CT_place_JUNGLE`: contribution `+0.002809`

Top utility-only movements:
- `lag_08__CT2__flash_duration`: contribution `-0.001042`

### tick `9902`, seconds `38.50`, LSTM delta `+0.0812`

Top all feature movements:
- `lag_15__CT_place_BACKALLEY`: contribution `+0.034439`
- `lag_07__CT_place_SIDEALLEY`: contribution `+0.029097`
- `lag_03__CT_place_JUNGLE`: contribution `+0.003832`
- `lag_00__T_bomb_zone_count`: contribution `+0.003076`
- `lag_00__kill_diff_last_3s`: contribution `+0.002387`

Top utility-only movements:
- `lag_03__CT_flash_alpha_mean`: contribution `-0.001061`
