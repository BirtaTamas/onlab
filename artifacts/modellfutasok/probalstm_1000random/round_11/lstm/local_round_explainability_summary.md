# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-vitality-vs-astralis-bo3-Zley6FZuKcttfrliAqsvWJ/astralis-vs-vitality-m1-inferno.csv`
- round_num: `9`

## Largest probability jumps

- tick `58734`, seconds `45.50`, LSTM `0.7399`, delta `+0.2771`
- tick `58446`, seconds `41.00`, LSTM `0.5336`, delta `-0.2561`
- tick `57006`, seconds `18.50`, LSTM `0.2687`, delta `-0.2046`
- tick `57902`, seconds `32.50`, LSTM `0.8602`, delta `+0.2039`
- tick `57678`, seconds `29.00`, LSTM `0.7200`, delta `+0.1795`
- tick `58862`, seconds `47.50`, LSTM `0.8725`, delta `+0.1208`
- tick `57870`, seconds `32.00`, LSTM `0.6563`, delta `-0.1137`
- tick `57390`, seconds `24.50`, LSTM `0.4456`, delta `+0.0927`
- tick `58798`, seconds `46.50`, LSTM `0.7498`, delta `+0.0428`
- tick `58670`, seconds `44.50`, LSTM `0.4878`, delta `-0.0366`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003250`, |coef| `0.003250`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.003212`, |coef| `0.003212`
- `lag_12__T_bomb_zone_count`: coefficient `-0.003082`, |coef| `0.003082`
- `lag_10__T_duck_amount_mean`: coefficient `-0.002957`, |coef| `0.002957`
- `lag_02__T_duck_amount_mean`: coefficient `-0.002945`, |coef| `0.002945`
- `lag_00__CT_velocity_mean`: coefficient `-0.002725`, |coef| `0.002725`
- `lag_13__T_bomb_zone_count`: coefficient `-0.002689`, |coef| `0.002689`
- `lag_12__T_duck_amount_mean`: coefficient `-0.002578`, |coef| `0.002578`
- `lag_02__CT_flashes_last_5s`: coefficient `0.002409`, |coef| `0.002409`
- `lag_03__T_bomb_zone_count`: coefficient `0.002406`, |coef| `0.002406`
- `lag_00__damage_diff_last_5s`: coefficient `0.002219`, |coef| `0.002219`
- `lag_00__CT_duck_amount_mean`: coefficient `0.002207`, |coef| `0.002207`
- `lag_00__CT_kills_last_3s`: coefficient `0.002167`, |coef| `0.002167`
- `lag_00__CT_defusing_count`: coefficient `0.002127`, |coef| `0.002127`
- `lag_07__CT_place_LIBRARY`: coefficient `0.001944`, |coef| `0.001944`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.003212` (lowers CT win probability)
- `lag_02__CT_flashes_last_5s`: coefficient `0.002409` (raises CT win probability)
- `lag_14__T_A_site_active_infernos`: coefficient `-0.001476` (lowers CT win probability)
- `lag_02__T1__flash_duration`: coefficient `-0.001431` (lowers CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `0.001415` (raises CT win probability)
- `lag_04__T_flash_alpha_mean`: coefficient `-0.001310` (lowers CT win probability)
- `lag_03__CT2__flash_duration`: coefficient `-0.001271` (lowers CT win probability)
- `lag_09__T_A_site_active_infernos`: coefficient `-0.001270` (lowers CT win probability)
- `lag_05__T_A_site_active_infernos`: coefficient `0.001256` (raises CT win probability)
- `lag_08__CT_flashes_last_5s`: coefficient `0.001170` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003250` (raises CT win probability)
- `lag_12__T_bomb_zone_count`: coefficient `-0.003082` (lowers CT win probability)
- `lag_10__T_duck_amount_mean`: coefficient `-0.002957` (lowers CT win probability)
- `lag_02__T_duck_amount_mean`: coefficient `-0.002945` (lowers CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.002725` (lowers CT win probability)
- `lag_13__T_bomb_zone_count`: coefficient `-0.002689` (lowers CT win probability)
- `lag_12__T_duck_amount_mean`: coefficient `-0.002578` (lowers CT win probability)
- `lag_03__T_bomb_zone_count`: coefficient `0.002406` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002219` (raises CT win probability)
- `lag_00__CT_duck_amount_mean`: coefficient `0.002207` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `58734`, seconds `45.50`, LSTM delta `+0.2771`

Top all feature movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.019487`
- `lag_12__T_bomb_zone_count`: contribution `+0.017942`
- `lag_10__T_duck_amount_mean`: contribution `+0.017199`
- `lag_12__T_duck_amount_mean`: contribution `+0.014991`
- `lag_00__CT_velocity_mean`: contribution `+0.009961`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.019487`
- `lag_14__T_A_site_active_infernos`: contribution `+0.004394`

### tick `58446`, seconds `41.00`, LSTM delta `-0.2561`

Top all feature movements:
- `lag_02__T_duck_amount_mean`: contribution `-0.017131`
- `lag_13__T_bomb_zone_count`: contribution `-0.015654`
- `lag_03__T_bomb_zone_count`: contribution `-0.014009`
- `lag_07__CT_place_LIBRARY`: contribution `-0.012462`
- `lag_01__T_duck_amount_mean`: contribution `-0.007991`

Top utility-only movements:
- `lag_09__T_A_site_active_infernos`: contribution `-0.003780`

### tick `57006`, seconds `18.50`, LSTM delta `-0.2046`

Top all feature movements:
- `lag_02__T_flashed_players`: contribution `-0.011431`
- `lag_02__T1__flash_duration`: contribution `-0.010854`
- `lag_04__CT_place_BALCONY`: contribution `-0.008348`
- `lag_03__CT2__flash_duration`: contribution `-0.008345`
- `lag_00__CT2__flash_duration`: contribution `-0.008020`

Top utility-only movements:
- `lag_02__T1__flash_duration`: contribution `-0.010854`
- `lag_03__CT2__flash_duration`: contribution `-0.008345`
- `lag_00__CT2__flash_duration`: contribution `-0.008020`
- `lag_02__T_flash_duration_sum`: contribution `-0.005844`
- `lag_07__CT_A_site_active_infernos`: contribution `-0.002861`

### tick `57902`, seconds `32.50`, LSTM delta `+0.2039`

Top all feature movements:
- `lag_07__T_place_ARCH`: contribution `+0.022957`
- `lag_05__CT_place_TRAMP`: contribution `+0.019281`
- `lag_08__CT_place_TRAMP`: contribution `+0.019094`
- `lag_00__kill_diff_last_3s`: contribution `+0.007823`
- `lag_02__T3__duck_amount`: contribution `+0.006682`

Top utility-only movements:
- `lag_07__T4__flash_duration`: contribution `+0.003317`
- `lag_08__T4__flash_duration`: contribution `+0.002238`

### tick `57678`, seconds `29.00`, LSTM delta `+0.1795`

Top all feature movements:
- `lag_00__T_place_ARCH`: contribution `+0.030181`
- `lag_01__CT_place_TRAMP`: contribution `+0.022152`
- `lag_12__T_place_ARCH`: contribution `+0.009887`
- `lag_11__T_place_ARCH`: contribution `+0.009664`
- `lag_00__damage_diff_last_5s`: contribution `+0.008961`

Top utility-only movements:
- `lag_09__T1__flash_duration`: contribution `+0.005836`
- `lag_01__T4__flash_duration`: contribution `+0.001827`
- `lag_00__T4__flash_duration`: contribution `+0.001637`
