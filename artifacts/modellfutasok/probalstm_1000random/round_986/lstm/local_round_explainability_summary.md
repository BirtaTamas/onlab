# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-imperial-vs-legacy-bo3-GRvbnL5Q4zT_JzAd-0AXgo/imperial-vs-legacy-m3-mirage.csv`
- round_num: `2`

## Largest probability jumps

- tick `10870`, seconds `43.00`, LSTM `0.9529`, delta `+0.0366`
- tick `10678`, seconds `40.00`, LSTM `0.8755`, delta `-0.0355`
- tick `9078`, seconds `15.00`, LSTM `0.9305`, delta `+0.0355`
- tick `10838`, seconds `42.50`, LSTM `0.9163`, delta `+0.0320`
- tick `8790`, seconds `10.50`, LSTM `0.8914`, delta `-0.0292`
- tick `8918`, seconds `12.50`, LSTM `0.8807`, delta `-0.0253`
- tick `8502`, seconds `6.00`, LSTM `0.8886`, delta `-0.0242`
- tick `9046`, seconds `14.50`, LSTM `0.8950`, delta `+0.0217`
- tick `12086`, seconds `62.00`, LSTM `0.9701`, delta `+0.0209`
- tick `8630`, seconds `8.00`, LSTM `0.9064`, delta `+0.0190`

## Top 15 local ridge features

- `lag_01__CT_place_SCAFFOLDING`: coefficient `-0.000606`, |coef| `0.000606`
- `lag_00__CT_place_LADDER`: coefficient `-0.000513`, |coef| `0.000513`
- `lag_00__T_kills_last_3s`: coefficient `-0.000468`, |coef| `0.000468`
- `lag_12__T_place_HOUSE`: coefficient `-0.000428`, |coef| `0.000428`
- `lag_04__CT_place_SNIPERSNEST`: coefficient `-0.000409`, |coef| `0.000409`
- `lag_00__CT_place_SNIPERSNEST`: coefficient `0.000402`, |coef| `0.000402`
- `lag_00__CT3__duck_amount`: coefficient `0.000392`, |coef| `0.000392`
- `lag_00__CT_place_SCAFFOLDING`: coefficient `-0.000370`, |coef| `0.000370`
- `lag_00__T_place_SIDEALLEY`: coefficient `0.000368`, |coef| `0.000368`
- `lag_04__T2__duck_amount`: coefficient `-0.000366`, |coef| `0.000366`
- `lag_00__CT4__is_walking`: coefficient `-0.000359`, |coef| `0.000359`
- `lag_00__T_duck_amount_mean`: coefficient `-0.000359`, |coef| `0.000359`
- `lag_01__CT_place_STAIRS`: coefficient `-0.000356`, |coef| `0.000356`
- `lag_14__CT_place_SIDEALLEY`: coefficient `0.000352`, |coef| `0.000352`
- `lag_09__CT_place_LADDER`: coefficient `0.000343`, |coef| `0.000343`

## Top 10 utility ridge features

- `lag_03__CT_B_site_active_smokes`: coefficient `-0.000142` (lowers CT win probability)
- `lag_06__CT_A_site_active_infernos`: coefficient `-0.000134` (lowers CT win probability)
- `lag_08__CT1__smoke`: coefficient `-0.000126` (lowers CT win probability)
- `lag_12__CT_A_site_active_infernos`: coefficient `-0.000125` (lowers CT win probability)
- `lag_01__CT_active_infernos`: coefficient `0.000124` (raises CT win probability)
- `lag_09__CT1__smoke`: coefficient `-0.000118` (lowers CT win probability)
- `lag_05__CT_active_infernos`: coefficient `-0.000114` (lowers CT win probability)
- `lag_00__CT_flash_alpha_mean`: coefficient `-0.000113` (lowers CT win probability)
- `lag_01__CT_B_site_active_infernos`: coefficient `0.000113` (raises CT win probability)
- `lag_04__CT_B_site_active_smokes`: coefficient `-0.000107` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_SCAFFOLDING`: coefficient `-0.000606` (lowers CT win probability)
- `lag_00__CT_place_LADDER`: coefficient `-0.000513` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.000468` (lowers CT win probability)
- `lag_12__T_place_HOUSE`: coefficient `-0.000428` (lowers CT win probability)
- `lag_04__CT_place_SNIPERSNEST`: coefficient `-0.000409` (lowers CT win probability)
- `lag_00__CT_place_SNIPERSNEST`: coefficient `0.000402` (raises CT win probability)
- `lag_00__CT3__duck_amount`: coefficient `0.000392` (raises CT win probability)
- `lag_00__CT_place_SCAFFOLDING`: coefficient `-0.000370` (lowers CT win probability)
- `lag_00__T_place_SIDEALLEY`: coefficient `0.000368` (raises CT win probability)
- `lag_04__T2__duck_amount`: coefficient `-0.000366` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `10870`, seconds `43.00`, LSTM delta `+0.0366`

Top all feature movements:
- `lag_01__CT_place_SCAFFOLDING`: contribution `+0.012642`
- `lag_00__T_kills_last_3s`: contribution `+0.001481`
- `lag_10__CT_place_SNIPERSNEST`: contribution `+0.001477`
- `lag_00__CT3__duck_amount`: contribution `+0.001459`
- `lag_00__T1__duck_amount`: contribution `+0.001248`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `10678`, seconds `40.00`, LSTM delta `-0.0355`

Top all feature movements:
- `lag_04__CT_place_SNIPERSNEST`: contribution `-0.002189`
- `lag_00__T_kills_last_3s`: contribution `-0.001481`
- `lag_04__T2__duck_amount`: contribution `-0.001401`
- `lag_15__CT_place_JUNGLE`: contribution `-0.001363`
- `lag_08__T2__duck_amount`: contribution `-0.000974`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `9078`, seconds `15.00`, LSTM delta `+0.0355`

Top all feature movements:
- `lag_00__CT_place_LADDER`: contribution `+0.005331`
- `lag_09__CT_place_LADDER`: contribution `+0.003567`
- `lag_12__T_place_HOUSE`: contribution `+0.001883`
- `lag_05__CT_place_SNIPERSNEST`: contribution `+0.001573`
- `lag_09__CT_place_SNIPERSNEST`: contribution `-0.001526`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `10838`, seconds `42.50`, LSTM delta `+0.0320`

Top all feature movements:
- `lag_00__CT_place_SCAFFOLDING`: contribution `+0.007729`
- `lag_09__CT_place_SNIPERSNEST`: contribution `+0.001526`
- `lag_04__T4__duck_amount`: contribution `+0.001162`
- `lag_09__T2__duck_amount`: contribution `+0.000934`
- `lag_06__T2__is_walking`: contribution `+0.000777`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `8790`, seconds `10.50`, LSTM delta `-0.0292`

Top all feature movements:
- `lag_00__CT_place_LADDER`: contribution `-0.005331`
- `lag_12__T_place_HOUSE`: contribution `-0.003766`
- `lag_00__CT_place_SNIPERSNEST`: contribution `-0.002153`
- `lag_05__CT_place_SNIPERSNEST`: contribution `-0.001573`
- `lag_00__CT3__duck_amount`: contribution `-0.001459`

Top utility-only movements:
- No utility movement among the top local contributors.
