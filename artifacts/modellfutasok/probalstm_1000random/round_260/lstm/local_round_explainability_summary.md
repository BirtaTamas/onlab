# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-furia-vs-virtuspro-bo3-E_bOFuD3YUjLJCO2xRj0mq/furia-vs-virtus-pro-m1-mirage.csv`
- round_num: `10`

## Largest probability jumps

- tick `93465`, seconds `94.00`, LSTM `0.1174`, delta `-0.3519`
- tick `93337`, seconds `92.00`, LSTM `0.5751`, delta `-0.1451`
- tick `93433`, seconds `93.50`, LSTM `0.4694`, delta `-0.0624`
- tick `93497`, seconds `94.50`, LSTM `0.0724`, delta `-0.0451`
- tick `93081`, seconds `88.00`, LSTM `0.7152`, delta `-0.0319`
- tick `93401`, seconds `93.00`, LSTM `0.5317`, delta `-0.0310`
- tick `91641`, seconds `65.50`, LSTM `0.7489`, delta `-0.0269`
- tick `90009`, seconds `40.00`, LSTM `0.7233`, delta `-0.0265`
- tick `88185`, seconds `11.50`, LSTM `0.7956`, delta `-0.0248`
- tick `92793`, seconds `83.50`, LSTM `0.7545`, delta `+0.0239`

## Top 15 local ridge features

- `lag_07__CT_place_STAIRS`: coefficient `-0.002793`, |coef| `0.002793`
- `lag_01__T_place_CONNECTOR`: coefficient `-0.002663`, |coef| `0.002663`
- `lag_02__T_place_CONNECTOR`: coefficient `-0.002502`, |coef| `0.002502`
- `lag_04__CT_place_STAIRS`: coefficient `0.002313`, |coef| `0.002313`
- `lag_11__CT5__flash_duration`: coefficient `0.001999`, |coef| `0.001999`
- `lag_14__CT_place_JUNGLE`: coefficient `-0.001838`, |coef| `0.001838`
- `lag_00__T_kills_last_3s`: coefficient `-0.001767`, |coef| `0.001767`
- `lag_13__T_place_UNDERPASS`: coefficient `0.001630`, |coef| `0.001630`
- `lag_08__T_place_PALACEINTERIOR`: coefficient `-0.001622`, |coef| `0.001622`
- `lag_00__T_damage_last_5s`: coefficient `-0.001561`, |coef| `0.001561`
- `lag_00__T3__is_scoped`: coefficient `0.001520`, |coef| `0.001520`
- `lag_04__CT4__is_scoped`: coefficient `-0.001500`, |coef| `0.001500`
- `lag_00__damage_diff_last_5s`: coefficient `0.001454`, |coef| `0.001454`
- `lag_04__T_kills_last_3s`: coefficient `-0.001434`, |coef| `0.001434`
- `lag_00__CT_money_sum`: coefficient `0.001370`, |coef| `0.001370`

## Top 10 utility ridge features

- `lag_11__CT5__flash_duration`: coefficient `0.001999` (raises CT win probability)
- `lag_11__CT_flash_duration_sum`: coefficient `0.000942` (raises CT win probability)
- `lag_07__CT5__flash_duration`: coefficient `0.000873` (raises CT win probability)
- `lag_04__CT4__flash`: coefficient `0.000766` (raises CT win probability)
- `lag_12__CT_A_site_active_smokes`: coefficient `-0.000717` (lowers CT win probability)
- `lag_12__CT_B_site_active_smokes`: coefficient `-0.000705` (lowers CT win probability)
- `lag_00__CT4__utility_total`: coefficient `0.000583` (raises CT win probability)
- `lag_00__CT4__flash`: coefficient `0.000583` (raises CT win probability)
- `lag_08__CT_A_site_active_smokes`: coefficient `-0.000572` (lowers CT win probability)
- `lag_13__CT4__smoke`: coefficient `0.000542` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_07__CT_place_STAIRS`: coefficient `-0.002793` (lowers CT win probability)
- `lag_01__T_place_CONNECTOR`: coefficient `-0.002663` (lowers CT win probability)
- `lag_02__T_place_CONNECTOR`: coefficient `-0.002502` (lowers CT win probability)
- `lag_04__CT_place_STAIRS`: coefficient `0.002313` (raises CT win probability)
- `lag_14__CT_place_JUNGLE`: coefficient `-0.001838` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001767` (lowers CT win probability)
- `lag_13__T_place_UNDERPASS`: coefficient `0.001630` (raises CT win probability)
- `lag_08__T_place_PALACEINTERIOR`: coefficient `-0.001622` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001561` (lowers CT win probability)
- `lag_00__T3__is_scoped`: coefficient `0.001520` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `93465`, seconds `94.00`, LSTM delta `-0.3519`

Top all feature movements:
- `lag_07__CT_place_STAIRS`: contribution `-0.021741`
- `lag_04__CT_place_STAIRS`: contribution `-0.018002`
- `lag_01__T_place_CONNECTOR`: contribution `-0.012895`
- `lag_11__CT5__flash_duration`: contribution `-0.012264`
- `lag_02__T_place_CONNECTOR`: contribution `-0.012115`

Top utility-only movements:
- `lag_11__CT5__flash_duration`: contribution `-0.012264`

### tick `93337`, seconds `92.00`, LSTM delta `-0.1451`

Top all feature movements:
- `lag_00__T3__is_scoped`: contribution `-0.009753`
- `lag_00__CT_place_STAIRS`: contribution `-0.008954`
- `lag_10__CT_place_JUNGLE`: contribution `+0.007960`
- `lag_00__T_kills_last_3s`: contribution `-0.005597`
- `lag_07__CT5__flash_duration`: contribution `-0.005356`

Top utility-only movements:
- `lag_07__CT5__flash_duration`: contribution `-0.005356`

### tick `93433`, seconds `93.50`, LSTM delta `-0.0624`

Top all feature movements:
- `lag_01__T_place_CONNECTOR`: contribution `-0.012895`
- `lag_00__T_place_CONNECTOR`: contribution `-0.006021`
- `lag_06__CT_place_STAIRS`: contribution `-0.005550`
- `lag_09__CT_place_JUNGLE`: contribution `-0.004332`
- `lag_12__T3__duck_amount`: contribution `+0.004204`

Top utility-only movements:
- `lag_10__CT5__flash_duration`: contribution `-0.003040`

### tick `93497`, seconds `94.50`, LSTM delta `-0.0451`

Top all feature movements:
- `lag_02__T_place_CONNECTOR`: contribution `-0.012115`
- `lag_08__CT_place_STAIRS`: contribution `-0.006722`
- `lag_03__T_place_CONNECTOR`: contribution `-0.006062`
- `lag_12__CT5__flash_duration`: contribution `-0.002857`
- `lag_05__CT4__duck_amount`: contribution `+0.002740`

Top utility-only movements:
- `lag_12__CT5__flash_duration`: contribution `-0.002857`

### tick `93081`, seconds `88.00`, LSTM delta `-0.0319`

Top all feature movements:
- `lag_00__CT5__duck_amount`: contribution `-0.004515`
- `lag_10__CT5__flash_duration`: contribution `+0.003040`
- `lag_01__T_place_UNDERPASS`: contribution `-0.001936`
- `lag_06__CT4__is_walking`: contribution `-0.001913`
- `lag_12__CT_place_SNIPERSNEST`: contribution `-0.001689`

Top utility-only movements:
- `lag_10__CT5__flash_duration`: contribution `+0.003040`
- `lag_12__CT_B_site_active_infernos`: contribution `-0.001471`
- `lag_10__CT_A_site_active_infernos`: contribution `-0.001223`
- `lag_12__CT_active_infernos`: contribution `-0.001051`
- `lag_10__CT_active_infernos`: contribution `-0.000856`
