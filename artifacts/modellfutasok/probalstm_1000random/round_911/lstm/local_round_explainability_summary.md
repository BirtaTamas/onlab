# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-faze-vs-aurora-bo3-OcxcOl9bFIHQQ2588nwUWG/faze-vs-aurora-m1-inferno.csv`
- round_num: `13`

## Largest probability jumps

- tick `102477`, seconds `29.00`, LSTM `0.1504`, delta `-0.2885`
- tick `102093`, seconds `23.00`, LSTM `0.5472`, delta `+0.2683`
- tick `101453`, seconds `13.00`, LSTM `0.3181`, delta `-0.1901`
- tick `103917`, seconds `51.50`, LSTM `0.0452`, delta `-0.1793`
- tick `103885`, seconds `51.00`, LSTM `0.2245`, delta `+0.1640`
- tick `101485`, seconds `13.50`, LSTM `0.2084`, delta `-0.1097`
- tick `101837`, seconds `19.00`, LSTM `0.3571`, delta `+0.0751`
- tick `102509`, seconds `29.50`, LSTM `0.0784`, delta `-0.0720`
- tick `102445`, seconds `28.50`, LSTM `0.4389`, delta `+0.0545`
- tick `104269`, seconds `57.00`, LSTM `0.0179`, delta `-0.0510`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004172`, |coef| `0.004172`
- `lag_00__T_kills_last_3s`: coefficient `-0.003115`, |coef| `0.003115`
- `lag_00__damage_diff_last_5s`: coefficient `0.003077`, |coef| `0.003077`
- `lag_01__T5__duck_amount`: coefficient `0.002761`, |coef| `0.002761`
- `lag_07__T_macro_B`: coefficient `-0.002679`, |coef| `0.002679`
- `lag_07__T_place_BOMBSITEB`: coefficient `-0.002679`, |coef| `0.002679`
- `lag_14__CT_place_QUAD`: coefficient `-0.002641`, |coef| `0.002641`
- `lag_13__CT_place_ARCH`: coefficient `0.002281`, |coef| `0.002281`
- `lag_05__T2__is_walking`: coefficient `-0.002221`, |coef| `0.002221`
- `lag_00__CT_kills_last_3s`: coefficient `0.002166`, |coef| `0.002166`
- `lag_10__CT4__duck_amount`: coefficient `0.002127`, |coef| `0.002127`
- `lag_15__T_place_ARCH`: coefficient `0.002121`, |coef| `0.002121`
- `lag_08__T4__duck_amount`: coefficient `0.002083`, |coef| `0.002083`
- `lag_10__CT_place_TOPOFMID`: coefficient `-0.002064`, |coef| `0.002064`
- `lag_05__CT_place_ARCH`: coefficient `0.002028`, |coef| `0.002028`

## Top 10 utility ridge features

- `lag_15__T_B_site_active_infernos`: coefficient `0.001874` (raises CT win probability)
- `lag_10__T_B_site_active_smokes`: coefficient `-0.001473` (lowers CT win probability)
- `lag_03__T_B_site_active_infernos`: coefficient `-0.001460` (lowers CT win probability)
- `lag_15__T_active_infernos`: coefficient `0.001397` (raises CT win probability)
- `lag_07__T3__flash`: coefficient `-0.001123` (lowers CT win probability)
- `lag_03__T_active_infernos`: coefficient `-0.001060` (lowers CT win probability)
- `lag_15__active_infernos_total`: coefficient `0.000982` (raises CT win probability)
- `lag_12__T5__flash`: coefficient `0.000979` (raises CT win probability)
- `lag_10__T_active_smokes`: coefficient `-0.000934` (lowers CT win probability)
- `lag_08__T5__smoke`: coefficient `-0.000910` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004172` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003115` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003077` (raises CT win probability)
- `lag_01__T5__duck_amount`: coefficient `0.002761` (raises CT win probability)
- `lag_07__T_macro_B`: coefficient `-0.002679` (lowers CT win probability)
- `lag_07__T_place_BOMBSITEB`: coefficient `-0.002679` (lowers CT win probability)
- `lag_14__CT_place_QUAD`: coefficient `-0.002641` (lowers CT win probability)
- `lag_13__CT_place_ARCH`: coefficient `0.002281` (raises CT win probability)
- `lag_05__T2__is_walking`: coefficient `-0.002221` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002166` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `102477`, seconds `29.00`, LSTM delta `-0.2885`

Top all feature movements:
- `lag_01__T5__duck_amount`: contribution `-0.010483`
- `lag_00__kill_diff_last_3s`: contribution `-0.010042`
- `lag_00__T_kills_last_3s`: contribution `-0.009867`
- `lag_13__CT_place_ARCH`: contribution `-0.009307`
- `lag_07__T_place_BOMBSITEB`: contribution `-0.008363`

Top utility-only movements:
- `lag_15__T_B_site_active_infernos`: contribution `-0.005297`

### tick `102093`, seconds `23.00`, LSTM delta `+0.2683`

Top all feature movements:
- `lag_14__CT_place_QUAD`: contribution `+0.020816`
- `lag_00__kill_diff_last_3s`: contribution `+0.010042`
- `lag_08__T4__duck_amount`: contribution `+0.007702`
- `lag_10__CT_place_TOPOFMID`: contribution `+0.007491`
- `lag_06__CT2__duck_amount`: contribution `+0.007394`

Top utility-only movements:
- `lag_03__T_B_site_active_infernos`: contribution `+0.004128`

### tick `101453`, seconds `13.00`, LSTM delta `-0.1901`

Top all feature movements:
- `lag_15__T_place_LOWERMID`: contribution `-0.011189`
- `lag_11__T_place_LOWERMID`: contribution `-0.010960`
- `lag_00__kill_diff_last_3s`: contribution `-0.010042`
- `lag_00__T_kills_last_3s`: contribution `-0.009867`
- `lag_13__CT_place_ARCH`: contribution `-0.009307`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `103917`, seconds `51.50`, LSTM delta `-0.1793`

Top all feature movements:
- `lag_10__T_place_ARCH`: contribution `-0.018245`
- `lag_00__kill_diff_last_3s`: contribution `-0.010042`
- `lag_00__T_kills_last_3s`: contribution `-0.009867`
- `lag_10__T_place_CTSPAWN`: contribution `-0.008976`
- `lag_01__T_place_CTSPAWN`: contribution `-0.006946`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `103885`, seconds `51.00`, LSTM delta `+0.1640`

Top all feature movements:
- `lag_15__T_place_ARCH`: contribution `+0.019733`
- `lag_09__T_place_ARCH`: contribution `+0.014340`
- `lag_00__kill_diff_last_3s`: contribution `+0.010042`
- `lag_00__T_place_CTSPAWN`: contribution `+0.006610`
- `lag_00__CT_kills_last_3s`: contribution `+0.006253`

Top utility-only movements:
- No utility movement among the top local contributors.
