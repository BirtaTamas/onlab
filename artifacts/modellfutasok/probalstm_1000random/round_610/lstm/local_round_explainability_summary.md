# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-furia-vs-virtuspro-bo3-E_bOFuD3YUjLJCO2xRj0mq/furia-vs-virtus-pro-m1-mirage.csv`
- round_num: `11`

## Largest probability jumps

- tick `101781`, seconds `74.50`, LSTM `0.6309`, delta `+0.1156`
- tick `101973`, seconds `77.50`, LSTM `0.8893`, delta `+0.0821`
- tick `101941`, seconds `77.00`, LSTM `0.8072`, delta `+0.0637`
- tick `102037`, seconds `78.50`, LSTM `0.9576`, delta `+0.0631`
- tick `101909`, seconds `76.50`, LSTM `0.7435`, delta `+0.0397`
- tick `97461`, seconds `7.00`, LSTM `0.5921`, delta `-0.0352`
- tick `101813`, seconds `75.00`, LSTM `0.6656`, delta `+0.0347`
- tick `101845`, seconds `75.50`, LSTM `0.6969`, delta `+0.0313`
- tick `97429`, seconds `6.50`, LSTM `0.6274`, delta `+0.0301`
- tick `103061`, seconds `94.50`, LSTM `0.9720`, delta `+0.0258`

## Top 15 local ridge features

- `lag_08__T1__flash_duration`: coefficient `-0.001108`, |coef| `0.001108`
- `lag_06__CT_place_SHOP`: coefficient `0.000998`, |coef| `0.000998`
- `lag_12__CT_place_SHOP`: coefficient `0.000934`, |coef| `0.000934`
- `lag_10__CT1__flash_duration`: coefficient `0.000870`, |coef| `0.000870`
- `lag_13__T1__flash_duration`: coefficient `-0.000775`, |coef| `0.000775`
- `lag_00__damage_diff_last_5s`: coefficient `0.000767`, |coef| `0.000767`
- `lag_00__CT_kills_last_3s`: coefficient `0.000752`, |coef| `0.000752`
- `lag_15__CT1__flash_duration`: coefficient `0.000734`, |coef| `0.000734`
- `lag_11__CT_place_SHOP`: coefficient `0.000718`, |coef| `0.000718`
- `lag_09__CT3__duck_amount`: coefficient `-0.000693`, |coef| `0.000693`
- `lag_10__T4__flash_duration`: coefficient `0.000665`, |coef| `0.000665`
- `lag_07__CT_place_SHOP`: coefficient `0.000664`, |coef| `0.000664`
- `lag_14__T1__flash_duration`: coefficient `-0.000652`, |coef| `0.000652`
- `lag_01__damage_diff_last_5s`: coefficient `0.000651`, |coef| `0.000651`
- `lag_00__CT_damage_last_5s`: coefficient `0.000650`, |coef| `0.000650`

## Top 10 utility ridge features

- `lag_08__T1__flash_duration`: coefficient `-0.001108` (lowers CT win probability)
- `lag_10__CT1__flash_duration`: coefficient `0.000870` (raises CT win probability)
- `lag_13__T1__flash_duration`: coefficient `-0.000775` (lowers CT win probability)
- `lag_15__CT1__flash_duration`: coefficient `0.000734` (raises CT win probability)
- `lag_10__T4__flash_duration`: coefficient `0.000665` (raises CT win probability)
- `lag_14__T1__flash_duration`: coefficient `-0.000652` (lowers CT win probability)
- `lag_00__T_he_last_5s`: coefficient `0.000647` (raises CT win probability)
- `lag_09__T1__flash_duration`: coefficient `-0.000623` (lowers CT win probability)
- `lag_12__T1__flash_duration`: coefficient `-0.000578` (lowers CT win probability)
- `lag_14__CT1__flash_duration`: coefficient `0.000512` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_06__CT_place_SHOP`: coefficient `0.000998` (raises CT win probability)
- `lag_12__CT_place_SHOP`: coefficient `0.000934` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000767` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000752` (raises CT win probability)
- `lag_11__CT_place_SHOP`: coefficient `0.000718` (raises CT win probability)
- `lag_09__CT3__duck_amount`: coefficient `-0.000693` (lowers CT win probability)
- `lag_07__CT_place_SHOP`: coefficient `0.000664` (raises CT win probability)
- `lag_01__damage_diff_last_5s`: coefficient `0.000651` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.000650` (raises CT win probability)
- `lag_09__CT1__is_scoped`: coefficient `-0.000642` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `101781`, seconds `74.50`, LSTM delta `+0.1156`

Top all feature movements:
- `lag_08__T1__flash_duration`: contribution `+0.008231`
- `lag_10__CT1__flash_duration`: contribution `+0.005190`
- `lag_06__CT_place_SHOP`: contribution `+0.005007`
- `lag_01__T3__is_scoped`: contribution `+0.003982`
- `lag_10__T4__flash_duration`: contribution `+0.003482`

Top utility-only movements:
- `lag_08__T1__flash_duration`: contribution `+0.008231`
- `lag_10__CT1__flash_duration`: contribution `+0.005190`
- `lag_10__T4__flash_duration`: contribution `+0.003482`
- `lag_08__T_flash_duration_sum`: contribution `+0.001559`
- `lag_15__CT_B_site_active_infernos`: contribution `+0.001480`

### tick `101973`, seconds `77.50`, LSTM delta `+0.0821`

Top all feature movements:
- `lag_14__T1__flash_duration`: contribution `+0.004841`
- `lag_12__CT_place_SHOP`: contribution `+0.004684`
- `lag_06__T_place_UNDERPASS`: contribution `+0.002454`
- `lag_03__CT1__flash_duration`: contribution `+0.001825`
- `lag_00__T3__is_scoped`: contribution `-0.001753`

Top utility-only movements:
- `lag_14__T1__flash_duration`: contribution `+0.004841`
- `lag_03__CT1__flash_duration`: contribution `+0.001825`
- `lag_04__T4__flash_duration`: contribution `+0.001439`

### tick `101941`, seconds `77.00`, LSTM delta `+0.0637`

Top all feature movements:
- `lag_13__T1__flash_duration`: contribution `+0.005758`
- `lag_15__CT1__flash_duration`: contribution `+0.004376`
- `lag_11__CT_place_SHOP`: contribution `+0.003602`
- `lag_09__CT1__is_scoped`: contribution `+0.002751`
- `lag_15__T4__flash_duration`: contribution `+0.002662`

Top utility-only movements:
- `lag_13__T1__flash_duration`: contribution `+0.005758`
- `lag_15__CT1__flash_duration`: contribution `+0.004376`
- `lag_15__T4__flash_duration`: contribution `+0.002662`
- `lag_02__CT1__flash_duration`: contribution `+0.002228`
- `lag_03__T4__flash_duration`: contribution `+0.001254`

### tick `102037`, seconds `78.50`, LSTM delta `+0.0631`

Top all feature movements:
- `lag_14__CT_place_SHOP`: contribution `+0.003136`
- `lag_09__CT3__duck_amount`: contribution `+0.002578`
- `lag_00__CT_kills_last_3s`: contribution `+0.002171`
- `lag_05__CT1__flash_duration`: contribution `+0.001902`
- `lag_06__T4__flash_duration`: contribution `+0.001795`

Top utility-only movements:
- `lag_05__CT1__flash_duration`: contribution `+0.001902`
- `lag_06__T4__flash_duration`: contribution `+0.001795`

### tick `101909`, seconds `76.50`, LSTM delta `+0.0397`

Top all feature movements:
- `lag_12__T1__flash_duration`: contribution `+0.004292`
- `lag_14__CT1__flash_duration`: contribution `+0.003054`
- `lag_10__CT_place_SHOP`: contribution `+0.002958`
- `lag_04__T_place_UNDERPASS`: contribution `+0.001936`
- `lag_14__T4__flash_duration`: contribution `+0.001746`

Top utility-only movements:
- `lag_12__T1__flash_duration`: contribution `+0.004292`
- `lag_14__CT1__flash_duration`: contribution `+0.003054`
- `lag_14__T4__flash_duration`: contribution `+0.001746`
- `lag_01__CT1__flash_duration`: contribution `+0.001587`
- `lag_02__T4__flash_duration`: contribution `+0.001007`
