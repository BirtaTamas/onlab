# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-furia-bo5-6eeTFVdtPEH4qPNc6w4Z3Y/the-mongolz-vs-furia-m5-dust2.csv`
- round_num: `13`

## Largest probability jumps

- tick `93352`, seconds `46.50`, LSTM `0.6854`, delta `+0.2551`
- tick `93192`, seconds `44.00`, LSTM `0.3564`, delta `-0.2302`
- tick `93000`, seconds `41.00`, LSTM `0.4024`, delta `+0.2296`
- tick `93160`, seconds `43.50`, LSTM `0.5866`, delta `+0.1677`
- tick `91208`, seconds `13.00`, LSTM `0.4137`, delta `-0.1339`
- tick `91592`, seconds `19.00`, LSTM `0.1479`, delta `-0.1164`
- tick `93480`, seconds `48.50`, LSTM `0.8827`, delta `+0.1138`
- tick `93224`, seconds `44.50`, LSTM `0.4459`, delta `+0.0894`
- tick `92680`, seconds `36.00`, LSTM `0.0949`, delta `+0.0855`
- tick `91560`, seconds `18.50`, LSTM `0.2643`, delta `-0.0768`

## Top 15 local ridge features

- `lag_00__CT_defusing_count`: coefficient `0.005703`, |coef| `0.005703`
- `lag_05__CT_place_HOLE`: coefficient `0.003724`, |coef| `0.003724`
- `lag_09__CT_place_HOLE`: coefficient `-0.003577`, |coef| `0.003577`
- `lag_00__kill_diff_last_3s`: coefficient `0.003253`, |coef| `0.003253`
- `lag_00__CT_velocity_mean`: coefficient `-0.003183`, |coef| `0.003183`
- `lag_00__CT_kills_last_3s`: coefficient `0.003163`, |coef| `0.003163`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.002950`, |coef| `0.002950`
- `lag_07__CT_place_UPPERTUNNEL`: coefficient `-0.002936`, |coef| `0.002936`
- `lag_02__CT_defusing_count`: coefficient `0.002742`, |coef| `0.002742`
- `lag_04__CT_place_HOLE`: coefficient `0.002448`, |coef| `0.002448`
- `lag_02__T_place_OUTSIDETUNNEL`: coefficient `-0.002420`, |coef| `0.002420`
- `lag_11__CT_place_HOLE`: coefficient `-0.002217`, |coef| `0.002217`
- `lag_08__CT_place_HOLE`: coefficient `-0.002013`, |coef| `0.002013`
- `lag_06__CT_defusing_count`: coefficient `0.001963`, |coef| `0.001963`
- `lag_05__CT_place_BDOORS`: coefficient `-0.001951`, |coef| `0.001951`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.002950` (lowers CT win probability)
- `lag_03__T_flash_alpha_mean`: coefficient `-0.001570` (lowers CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `-0.001402` (lowers CT win probability)
- `lag_04__T_flash_alpha_mean`: coefficient `-0.001359` (lowers CT win probability)
- `lag_05__T_flash_alpha_mean`: coefficient `-0.001321` (lowers CT win probability)
- `lag_15__T_B_site_active_smokes`: coefficient `-0.001072` (lowers CT win probability)
- `lag_02__T_flash_alpha_mean`: coefficient `-0.000904` (lowers CT win probability)
- `lag_07__T_flash_alpha_mean`: coefficient `-0.000834` (lowers CT win probability)
- `lag_06__T_flash_alpha_mean`: coefficient `-0.000749` (lowers CT win probability)
- `lag_15__T_active_smokes`: coefficient `-0.000720` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_defusing_count`: coefficient `0.005703` (raises CT win probability)
- `lag_05__CT_place_HOLE`: coefficient `0.003724` (raises CT win probability)
- `lag_09__CT_place_HOLE`: coefficient `-0.003577` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003253` (raises CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.003183` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003163` (raises CT win probability)
- `lag_07__CT_place_UPPERTUNNEL`: coefficient `-0.002936` (lowers CT win probability)
- `lag_02__CT_defusing_count`: coefficient `0.002742` (raises CT win probability)
- `lag_04__CT_place_HOLE`: coefficient `0.002448` (raises CT win probability)
- `lag_02__T_place_OUTSIDETUNNEL`: coefficient `-0.002420` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `93352`, seconds `46.50`, LSTM delta `+0.2551`

Top all feature movements:
- `lag_09__CT_place_HOLE`: contribution `+0.039935`
- `lag_07__CT_place_UPPERTUNNEL`: contribution `+0.022515`
- `lag_06__CT_defusing_count`: contribution `+0.019031`
- `lag_00__T_flash_alpha_mean`: contribution `+0.017901`
- `lag_05__CT_defusing_count`: contribution `+0.015833`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.017901`

### tick `93192`, seconds `44.00`, LSTM delta `-0.2302`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `-0.055284`
- `lag_04__CT_place_HOLE`: contribution `-0.027324`
- `lag_11__CT_place_HOLE`: contribution `-0.024752`
- `lag_00__CT_velocity_mean`: contribution `-0.017526`
- `lag_01__CT_defusing_count`: contribution `-0.009281`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `93000`, seconds `41.00`, LSTM delta `+0.2296`

Top all feature movements:
- `lag_05__CT_place_HOLE`: contribution `+0.041580`
- `lag_02__T_place_OUTSIDETUNNEL`: contribution `+0.012096`
- `lag_05__CT_place_BDOORS`: contribution `+0.009386`
- `lag_00__CT_kills_last_3s`: contribution `+0.009132`
- `lag_00__kill_diff_last_3s`: contribution `+0.007829`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `93160`, seconds `43.50`, LSTM delta `+0.1677`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `+0.055284`
- `lag_01__CT_place_UPPERTUNNEL`: contribution `+0.013957`
- `lag_03__CT_place_HOLE`: contribution `+0.012768`
- `lag_10__CT_place_HOLE`: contribution `+0.010000`
- `lag_10__CT_place_BDOORS`: contribution `+0.006837`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `91208`, seconds `13.00`, LSTM delta `-0.1339`

Top all feature movements:
- `lag_14__T_place_OUTSIDETUNNEL`: contribution `-0.009364`
- `lag_00__kill_diff_last_3s`: contribution `-0.007829`
- `lag_14__CT_place_BDOORS`: contribution `-0.007687`
- `lag_00__CT_flashed_players`: contribution `-0.005707`
- `lag_15__CT1__duck_amount`: contribution `-0.004971`

Top utility-only movements:
- `lag_01__CT2__flash_duration`: contribution `-0.001984`
