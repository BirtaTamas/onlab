# Local Round Explainability

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-the-mongolz-vs-natus-vincere-bo3-C0GZxMhpGHBr28LeyjgICZ/the-mongolz-vs-natus-vincere-m1-mirage.csv`
- round_num: `2`

## Largest probability jumps

- tick `6670`, seconds `10.00`, LSTM `0.1193`, delta `-0.0562`
- tick `6574`, seconds `8.50`, LSTM `0.2047`, delta `-0.0543`
- tick `6638`, seconds `9.50`, LSTM `0.1755`, delta `-0.0454`
- tick `6446`, seconds `6.50`, LSTM `0.2977`, delta `-0.0375`
- tick `6926`, seconds `14.00`, LSTM `0.0329`, delta `-0.0374`
- tick `6286`, seconds `4.00`, LSTM `0.3648`, delta `-0.0314`
- tick `6894`, seconds `13.50`, LSTM `0.0703`, delta `-0.0284`
- tick `6062`, seconds `0.50`, LSTM `0.3656`, delta `-0.0211`
- tick `6702`, seconds `10.50`, LSTM `0.1006`, delta `-0.0188`
- tick `6606`, seconds `9.00`, LSTM `0.2209`, delta `+0.0162`

## Top 15 local ridge features

- `lag_00__CT_place_CTSPAWN`: coefficient `0.000764`, |coef| `0.000764`
- `lag_00__CT_place_SCAFFOLDING`: coefficient `-0.000678`, |coef| `0.000678`
- `lag_15__T_place_SIDEALLEY`: coefficient `-0.000664`, |coef| `0.000664`
- `lag_14__CT_flashes_last_5s`: coefficient `-0.000640`, |coef| `0.000640`
- `lag_00__T_place_TSPAWN`: coefficient `0.000625`, |coef| `0.000625`
- `lag_14__T_place_SIDEALLEY`: coefficient `-0.000601`, |coef| `0.000601`
- `lag_05__CT_place_PALACEINTERIOR`: coefficient `-0.000573`, |coef| `0.000573`
- `lag_10__CT_flashes_last_5s`: coefficient `-0.000571`, |coef| `0.000571`
- `lag_04__CT_place_PALACEINTERIOR`: coefficient `-0.000551`, |coef| `0.000551`
- `lag_02__CT_place_PALACEINTERIOR`: coefficient `-0.000544`, |coef| `0.000544`
- `lag_03__CT_place_PALACEINTERIOR`: coefficient `-0.000498`, |coef| `0.000498`
- `lag_06__CT_place_PALACEINTERIOR`: coefficient `-0.000484`, |coef| `0.000484`
- `lag_11__CT_place_PALACEINTERIOR`: coefficient `-0.000477`, |coef| `0.000477`
- `lag_02__CT_place_TRUCK`: coefficient `-0.000462`, |coef| `0.000462`
- `lag_07__T_place_SIDEALLEY`: coefficient `-0.000457`, |coef| `0.000457`

## Top 10 utility ridge features

- `lag_14__CT_flashes_last_5s`: coefficient `-0.000640` (lowers CT win probability)
- `lag_10__CT_flashes_last_5s`: coefficient `-0.000571` (lowers CT win probability)
- `lag_08__CT_flash_alpha_mean`: coefficient `0.000440` (raises CT win probability)
- `lag_01__CT_flash_alpha_mean`: coefficient `0.000424` (raises CT win probability)
- `lag_13__CT_flash_alpha_mean`: coefficient `0.000402` (raises CT win probability)
- `lag_02__CT_flash_alpha_mean`: coefficient `0.000396` (raises CT win probability)
- `lag_09__CT_flash_alpha_mean`: coefficient `0.000387` (raises CT win probability)
- `lag_15__CT_flashes_last_5s`: coefficient `-0.000378` (lowers CT win probability)
- `lag_12__CT_flash_alpha_mean`: coefficient `0.000378` (raises CT win probability)
- `lag_00__CT5__smoke`: coefficient `0.000376` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_CTSPAWN`: coefficient `0.000764` (raises CT win probability)
- `lag_00__CT_place_SCAFFOLDING`: coefficient `-0.000678` (lowers CT win probability)
- `lag_15__T_place_SIDEALLEY`: coefficient `-0.000664` (lowers CT win probability)
- `lag_00__T_place_TSPAWN`: coefficient `0.000625` (raises CT win probability)
- `lag_14__T_place_SIDEALLEY`: coefficient `-0.000601` (lowers CT win probability)
- `lag_05__CT_place_PALACEINTERIOR`: coefficient `-0.000573` (lowers CT win probability)
- `lag_04__CT_place_PALACEINTERIOR`: coefficient `-0.000551` (lowers CT win probability)
- `lag_02__CT_place_PALACEINTERIOR`: coefficient `-0.000544` (lowers CT win probability)
- `lag_03__CT_place_PALACEINTERIOR`: coefficient `-0.000498` (lowers CT win probability)
- `lag_06__CT_place_PALACEINTERIOR`: coefficient `-0.000484` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `6670`, seconds `10.00`, LSTM delta `-0.0562`

Top all feature movements:
- `lag_03__CT_place_SCAFFOLDING`: contribution `-0.008675`
- `lag_05__CT_place_UNDERPASS`: contribution `-0.002296`
- `lag_02__CT_place_PALACEINTERIOR`: contribution `-0.002217`
- `lag_15__T_place_SIDEALLEY`: contribution `-0.002118`
- `lag_14__T_place_SIDEALLEY`: contribution `-0.001915`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `6574`, seconds `8.50`, LSTM delta `-0.0543`

Top all feature movements:
- `lag_00__CT_place_SCAFFOLDING`: contribution `-0.014139`
- `lag_14__CT_flashes_last_5s`: contribution `-0.007037`
- `lag_08__T_place_HOUSE`: contribution `-0.001970`
- `lag_02__CT_place_UNDERPASS`: contribution `-0.001699`
- `lag_01__T_place_HOUSE`: contribution `-0.001371`

Top utility-only movements:
- `lag_14__CT_flashes_last_5s`: contribution `-0.007037`
- `lag_04__T1__flash_duration`: contribution `-0.001018`

### tick `6638`, seconds `9.50`, LSTM delta `-0.0454`

Top all feature movements:
- `lag_01__CT_place_SCAFFOLDING`: contribution `-0.007562`
- `lag_04__CT_place_UNDERPASS`: contribution `-0.002023`
- `lag_08__T_place_HOUSE`: contribution `-0.001970`
- `lag_14__T_place_SIDEALLEY`: contribution `-0.001915`
- `lag_02__CT_place_SCAFFOLDING`: contribution `+0.001724`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `6446`, seconds `6.50`, LSTM delta `-0.0375`

Top all feature movements:
- `lag_10__CT_flashes_last_5s`: contribution `-0.006279`
- `lag_00__CT_place_CTSPAWN`: contribution `-0.001591`
- `lag_07__T_place_SIDEALLEY`: contribution `-0.001458`
- `lag_05__T_place_PALACEINTERIOR`: contribution `-0.000997`
- `lag_03__CT1__duck_amount`: contribution `-0.000941`

Top utility-only movements:
- `lag_10__CT_flashes_last_5s`: contribution `-0.006279`
- `lag_00__CT2__smoke`: contribution `-0.000529`
- `lag_13__CT_flash_alpha_mean`: contribution `-0.000524`

### tick `6926`, seconds `14.00`, LSTM delta `-0.0374`

Top all feature movements:
- `lag_11__CT_place_SCAFFOLDING`: contribution `-0.005113`
- `lag_02__CT_place_TRUCK`: contribution `-0.002979`
- `lag_05__CT_flashes_last_5s`: contribution `+0.002263`
- `lag_13__CT_place_UNDERPASS`: contribution `-0.002009`
- `lag_10__CT_place_PALACEINTERIOR`: contribution `-0.001814`

Top utility-only movements:
- `lag_05__CT_flashes_last_5s`: contribution `+0.002263`
- `lag_00__CT1__smoke`: contribution `-0.000714`
