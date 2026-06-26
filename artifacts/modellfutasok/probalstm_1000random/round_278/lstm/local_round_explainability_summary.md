# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-vitality-vs-virtuspro-bo3-8Z0L17IYJlstHvIADVy9G9/vitality-vs-virtus-pro-m3-mirage.csv`
- round_num: `2`

## Largest probability jumps

- tick `7815`, seconds `5.00`, LSTM `0.9085`, delta `-0.0218`
- tick `13863`, seconds `99.50`, LSTM `0.9644`, delta `+0.0218`
- tick `13639`, seconds `96.00`, LSTM `0.9521`, delta `-0.0164`
- tick `7975`, seconds `7.50`, LSTM `0.9306`, delta `+0.0135`
- tick `13735`, seconds `97.50`, LSTM `0.9606`, delta `+0.0126`
- tick `7911`, seconds `6.50`, LSTM `0.9117`, delta `+0.0121`
- tick `7527`, seconds `0.50`, LSTM `0.9386`, delta `+0.0120`
- tick `13895`, seconds `100.00`, LSTM `0.9757`, delta `+0.0112`
- tick `8007`, seconds `8.00`, LSTM `0.9403`, delta `+0.0097`
- tick `13799`, seconds `98.50`, LSTM `0.9454`, delta `-0.0094`

## Top 15 local ridge features

- `lag_00__T_place_HOUSE`: coefficient `-0.000449`, |coef| `0.000449`
- `lag_07__T_place_TRUCK`: coefficient `0.000379`, |coef| `0.000379`
- `lag_00__CT3__is_walking`: coefficient `-0.000310`, |coef| `0.000310`
- `lag_00__T_place_SIDEALLEY`: coefficient `0.000280`, |coef| `0.000280`
- `lag_00__T_place_TRUCK`: coefficient `-0.000254`, |coef| `0.000254`
- `lag_03__T_place_TRUCK`: coefficient `0.000230`, |coef| `0.000230`
- `lag_00__T_walking_count`: coefficient `-0.000222`, |coef| `0.000222`
- `lag_01__T_place_HOUSE`: coefficient `-0.000219`, |coef| `0.000219`
- `lag_01__CT5__duck_amount`: coefficient `-0.000218`, |coef| `0.000218`
- `lag_08__T_place_TRUCK`: coefficient `0.000212`, |coef| `0.000212`
- `lag_00__CT_walking_count`: coefficient `-0.000208`, |coef| `0.000208`
- `lag_00__CT_place_SNIPERSNEST`: coefficient `0.000194`, |coef| `0.000194`
- `lag_00__CT5__duck_amount`: coefficient `-0.000192`, |coef| `0.000192`
- `lag_02__CT_place_STAIRS`: coefficient `0.000172`, |coef| `0.000172`
- `lag_00__T4__is_walking`: coefficient `-0.000168`, |coef| `0.000168`

## Top 10 utility ridge features

- `lag_02__CT1__flash_duration`: coefficient `0.000145` (raises CT win probability)
- `lag_01__CT1__flash_duration`: coefficient `0.000142` (raises CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `-0.000133` (lowers CT win probability)
- `lag_15__T_flash_alpha_mean`: coefficient `-0.000123` (lowers CT win probability)
- `lag_11__CT_utility_damage_last_5s`: coefficient `0.000113` (raises CT win probability)
- `lag_04__CT_utility_damage_last_5s`: coefficient `-0.000106` (lowers CT win probability)
- `lag_10__T_flash_alpha_mean`: coefficient `0.000105` (raises CT win probability)
- `lag_13__T_flash_alpha_mean`: coefficient `-0.000104` (lowers CT win probability)
- `lag_14__CT_utility_damage_last_5s`: coefficient `0.000104` (raises CT win probability)
- `lag_06__CT_utility_damage_last_5s`: coefficient `-0.000103` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_HOUSE`: coefficient `-0.000449` (lowers CT win probability)
- `lag_07__T_place_TRUCK`: coefficient `0.000379` (raises CT win probability)
- `lag_00__CT3__is_walking`: coefficient `-0.000310` (lowers CT win probability)
- `lag_00__T_place_SIDEALLEY`: coefficient `0.000280` (raises CT win probability)
- `lag_00__T_place_TRUCK`: coefficient `-0.000254` (lowers CT win probability)
- `lag_03__T_place_TRUCK`: coefficient `0.000230` (raises CT win probability)
- `lag_00__T_walking_count`: coefficient `-0.000222` (lowers CT win probability)
- `lag_01__T_place_HOUSE`: coefficient `-0.000219` (lowers CT win probability)
- `lag_01__CT5__duck_amount`: coefficient `-0.000218` (lowers CT win probability)
- `lag_08__T_place_TRUCK`: coefficient `0.000212` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `7815`, seconds `5.00`, LSTM delta `-0.0218`

Top all feature movements:
- `lag_00__T_place_HOUSE`: contribution `-0.003945`
- `lag_01__T_place_HOUSE`: contribution `-0.000963`
- `lag_00__T_place_SIDEALLEY`: contribution `-0.000891`
- `lag_04__T_place_SIDEALLEY`: contribution `-0.000743`
- `lag_10__CT_place_CTSPAWN`: contribution `-0.000673`

Top utility-only movements:
- `lag_10__T_flash_alpha_mean`: contribution `-0.000485`
- `lag_10__smoke_inv_diff`: contribution `-0.000235`

### tick `13863`, seconds `99.50`, LSTM delta `+0.0218`

Top all feature movements:
- `lag_07__T_place_TRUCK`: contribution `+0.006574`
- `lag_14__CT_place_STAIRS`: contribution `+0.001304`
- `lag_09__CT_place_STAIRS`: contribution `+0.001259`
- `lag_06__CT_place_JUNGLE`: contribution `+0.000736`
- `lag_03__T_bomb_zone_count`: contribution `+0.000618`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `13639`, seconds `96.00`, LSTM delta `-0.0164`

Top all feature movements:
- `lag_00__T_place_TRUCK`: contribution `-0.004417`
- `lag_02__CT_place_STAIRS`: contribution `-0.001339`
- `lag_00__CT_place_CATWALK`: contribution `-0.000547`
- `lag_06__CT_place_SNIPERSNEST`: contribution `-0.000466`
- `lag_00__T_place_APARTMENTS`: contribution `-0.000392`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `7975`, seconds `7.50`, LSTM delta `+0.0135`

Top all feature movements:
- `lag_00__CT_place_SNIPERSNEST`: contribution `+0.001037`
- `lag_01__CT1__flash_duration`: contribution `+0.000832`
- `lag_00__CT3__is_walking`: contribution `-0.000740`
- `lag_09__T_place_SIDEALLEY`: contribution `+0.000600`
- `lag_02__CT_place_SNIPERSNEST`: contribution `+0.000573`

Top utility-only movements:
- `lag_01__CT1__flash_duration`: contribution `+0.000832`
- `lag_15__T_flash_alpha_mean`: contribution `+0.000568`
- `lag_15__smoke_inv_diff`: contribution `+0.000276`

### tick `13735`, seconds `97.50`, LSTM delta `+0.0126`

Top all feature movements:
- `lag_03__T_place_TRUCK`: contribution `+0.003990`
- `lag_01__CT5__duck_amount`: contribution `+0.000821`
- `lag_04__CT_place_SHOP`: contribution `+0.000509`
- `lag_00__CT_kills_last_3s`: contribution `+0.000447`
- `lag_00__T_place_APARTMENTS`: contribution `-0.000392`

Top utility-only movements:
- No utility movement among the top local contributors.
