# Local Round Explainability

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-the-mongolz-vs-natus-vincere-bo3-C0GZxMhpGHBr28LeyjgICZ/the-mongolz-vs-natus-vincere-m1-mirage.csv`
- round_num: `5`

## Largest probability jumps

- tick `32523`, seconds `21.00`, LSTM `0.5148`, delta `-0.2446`
- tick `33195`, seconds `31.50`, LSTM `0.1051`, delta `-0.2194`
- tick `32587`, seconds `22.00`, LSTM `0.3521`, delta `-0.1464`
- tick `31979`, seconds `12.50`, LSTM `0.6410`, delta `+0.1319`
- tick `32683`, seconds `23.50`, LSTM `0.4479`, delta `+0.1036`
- tick `32363`, seconds `18.50`, LSTM `0.7650`, delta `-0.0976`
- tick `32811`, seconds `25.50`, LSTM `0.4116`, delta `-0.0955`
- tick `32971`, seconds `28.00`, LSTM `0.3148`, delta `-0.0944`
- tick `32075`, seconds `14.00`, LSTM `0.8602`, delta `+0.0900`
- tick `32043`, seconds `13.50`, LSTM `0.7702`, delta `+0.0761`

## Top 15 local ridge features

- `lag_06__T_place_SCAFFOLDING`: coefficient `0.002012`, |coef| `0.002012`
- `lag_01__CT_place_TRUCK`: coefficient `0.001964`, |coef| `0.001964`
- `lag_04__T_shots_fired_sum`: coefficient `0.001800`, |coef| `0.001800`
- `lag_09__T_shots_fired_sum`: coefficient `-0.001452`, |coef| `0.001452`
- `lag_00__CT1__flash_duration`: coefficient `-0.001411`, |coef| `0.001411`
- `lag_15__CT_place_SNIPERSNEST`: coefficient `0.001392`, |coef| `0.001392`
- `lag_07__T_place_SCAFFOLDING`: coefficient `-0.001346`, |coef| `0.001346`
- `lag_00__kill_diff_last_3s`: coefficient `0.001338`, |coef| `0.001338`
- `lag_00__damage_diff_last_5s`: coefficient `0.001324`, |coef| `0.001324`
- `lag_02__CT_place_TRUCK`: coefficient `0.001312`, |coef| `0.001312`
- `lag_00__T_place_SCAFFOLDING`: coefficient `-0.001280`, |coef| `0.001280`
- `lag_08__T_place_SCAFFOLDING`: coefficient `-0.001258`, |coef| `0.001258`
- `lag_00__CT_place_TRUCK`: coefficient `0.001195`, |coef| `0.001195`
- `lag_07__T_shots_fired_sum`: coefficient `-0.001172`, |coef| `0.001172`
- `lag_13__T_shots_fired_sum`: coefficient `0.001132`, |coef| `0.001132`

## Top 10 utility ridge features

- `lag_00__CT1__flash_duration`: coefficient `-0.001411` (lowers CT win probability)
- `lag_05__CT5__flash_duration`: coefficient `0.000998` (raises CT win probability)
- `lag_07__CT1__flash_duration`: coefficient `-0.000988` (lowers CT win probability)
- `lag_04__CT5__flash_duration`: coefficient `0.000942` (raises CT win probability)
- `lag_03__CT5__flash_duration`: coefficient `0.000887` (raises CT win probability)
- `lag_03__T_B_site_active_infernos`: coefficient `0.000862` (raises CT win probability)
- `lag_05__CT1__flash_duration`: coefficient `-0.000802` (lowers CT win probability)
- `lag_06__CT5__flash_duration`: coefficient `0.000769` (raises CT win probability)
- `lag_12__CT1__flash_duration`: coefficient `-0.000743` (lowers CT win probability)
- `lag_12__CT5__flash_duration`: coefficient `-0.000741` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_06__T_place_SCAFFOLDING`: coefficient `0.002012` (raises CT win probability)
- `lag_01__CT_place_TRUCK`: coefficient `0.001964` (raises CT win probability)
- `lag_04__T_shots_fired_sum`: coefficient `0.001800` (raises CT win probability)
- `lag_09__T_shots_fired_sum`: coefficient `-0.001452` (lowers CT win probability)
- `lag_15__CT_place_SNIPERSNEST`: coefficient `0.001392` (raises CT win probability)
- `lag_07__T_place_SCAFFOLDING`: coefficient `-0.001346` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001338` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001324` (raises CT win probability)
- `lag_02__CT_place_TRUCK`: coefficient `0.001312` (raises CT win probability)
- `lag_00__T_place_SCAFFOLDING`: coefficient `-0.001280` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `32523`, seconds `21.00`, LSTM delta `-0.2446`

Top all feature movements:
- `lag_04__T_shots_fired_sum`: contribution `-0.025634`
- `lag_01__CT_place_TRUCK`: contribution `-0.012667`
- `lag_15__CT_place_SNIPERSNEST`: contribution `-0.007455`
- `lag_04__T2__shots_fired`: contribution `-0.006612`
- `lag_02__CT_place_SNIPERSNEST`: contribution `-0.005280`

Top utility-only movements:
- `lag_04__CT_B_site_active_infernos`: contribution `-0.002512`
- `lag_03__T_B_site_active_infernos`: contribution `-0.002439`

### tick `33195`, seconds `31.50`, LSTM delta `-0.2194`

Top all feature movements:
- `lag_06__T_place_SCAFFOLDING`: contribution `-0.068513`
- `lag_07__T_place_SCAFFOLDING`: contribution `-0.045826`
- `lag_05__CT5__flash_duration`: contribution `-0.007642`
- `lag_11__CT_place_SIDEALLEY`: contribution `-0.006065`
- `lag_00__CT_place_TSPAWN`: contribution `-0.005927`

Top utility-only movements:
- `lag_05__CT5__flash_duration`: contribution `-0.007642`
- `lag_05__T_flash_duration_sum`: contribution `-0.001847`
- `lag_14__T_A_site_active_infernos`: contribution `-0.001789`

### tick `32587`, seconds `22.00`, LSTM delta `-0.1464`

Top all feature movements:
- `lag_00__CT1__flash_duration`: contribution `-0.009390`
- `lag_07__T_shots_fired_sum`: contribution `-0.008788`
- `lag_03__CT_place_TRUCK`: contribution `-0.007219`
- `lag_08__T_shots_fired_sum`: contribution `-0.005913`
- `lag_04__CT_place_SNIPERSNEST`: contribution `-0.005538`

Top utility-only movements:
- `lag_00__CT1__flash_duration`: contribution `-0.009390`
- `lag_00__CT5__flash_duration`: contribution `-0.004765`
- `lag_00__T5__flash_duration`: contribution `-0.002532`

### tick `31979`, seconds `12.50`, LSTM delta `+0.1319`

Top all feature movements:
- `lag_12__CT_place_SNIPERSNEST`: contribution `+0.003836`
- `lag_10__CT_place_SHOP`: contribution `+0.003654`
- `lag_12__T3__flash_duration`: contribution `+0.003546`
- `lag_15__CT_place_SHOP`: contribution `+0.003391`
- `lag_00__kill_diff_last_3s`: contribution `+0.003219`

Top utility-only movements:
- `lag_12__T3__flash_duration`: contribution `+0.003546`
- `lag_12__T1__flash_duration`: contribution `+0.003107`
- `lag_12__T_flash_duration_sum`: contribution `+0.001992`
- `lag_05__CT1__flash_duration`: contribution `+0.001883`
- `lag_01__CT_A_site_active_infernos`: contribution `+0.001746`

### tick `32683`, seconds `23.50`, LSTM delta `+0.1036`

Top all feature movements:
- `lag_09__T_shots_fired_sum`: contribution `+0.020685`
- `lag_03__CT5__flash_duration`: contribution `+0.006793`
- `lag_09__T2__shots_fired`: contribution `+0.005771`
- `lag_04__T_shots_fired_sum`: contribution `-0.004048`
- `lag_11__T_shots_fired_sum`: contribution `+0.003917`

Top utility-only movements:
- `lag_03__CT5__flash_duration`: contribution `+0.006793`
- `lag_03__T5__flash_duration`: contribution `+0.003166`
- `lag_14__T_A_site_active_infernos`: contribution `+0.001789`
