# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-liquid-bo3-pfm398EHUpu3zLY0TgcmxO/the-mongolz-vs-liquid-m1-dust2.csv`
- round_num: `5`

## Largest probability jumps

- tick `20777`, seconds `107.00`, LSTM `0.3750`, delta `+0.2636`
- tick `21257`, seconds `114.50`, LSTM `0.0605`, delta `-0.2321`
- tick `22793`, seconds `138.50`, LSTM `0.7760`, delta `+0.2300`
- tick `22185`, seconds `129.00`, LSTM `0.5273`, delta `+0.2280`
- tick `22441`, seconds `133.00`, LSTM `0.3450`, delta `-0.2090`
- tick `22697`, seconds `137.00`, LSTM `0.5697`, delta `+0.1969`
- tick `20265`, seconds `99.00`, LSTM `0.4909`, delta `-0.1593`
- tick `20457`, seconds `102.00`, LSTM `0.0179`, delta `-0.1411`
- tick `20361`, seconds `100.50`, LSTM `0.2164`, delta `-0.1228`
- tick `18185`, seconds `66.50`, LSTM `0.6940`, delta `+0.1123`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.006255`, |coef| `0.006255`
- `lag_00__damage_diff_last_5s`: coefficient `0.005108`, |coef| `0.005108`
- `lag_00__CT_defusing_count`: coefficient `0.005005`, |coef| `0.005005`
- `lag_12__CT_place_TUNNELSTAIRS`: coefficient `-0.004764`, |coef| `0.004764`
- `lag_00__CT_kills_last_3s`: coefficient `0.004121`, |coef| `0.004121`
- `lag_02__CT_place_UPPERTUNNEL`: coefficient `0.004010`, |coef| `0.004010`
- `lag_05__CT_place_LOWERTUNNEL`: coefficient `-0.003967`, |coef| `0.003967`
- `lag_00__T_kills_last_3s`: coefficient `-0.003710`, |coef| `0.003710`
- `lag_03__T_flash_alpha_mean`: coefficient `-0.003149`, |coef| `0.003149`
- `lag_00__T_damage_last_5s`: coefficient `-0.003026`, |coef| `0.003026`
- `lag_13__CT_place_UPPERTUNNEL`: coefficient `-0.002908`, |coef| `0.002908`
- `lag_10__CT_place_TUNNELSTAIRS`: coefficient `-0.002890`, |coef| `0.002890`
- `lag_08__CT_place_TUNNELSTAIRS`: coefficient `-0.002859`, |coef| `0.002859`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.002854`, |coef| `0.002854`
- `lag_00__T_place_BDOORS`: coefficient `-0.002779`, |coef| `0.002779`

## Top 10 utility ridge features

- `lag_03__T_flash_alpha_mean`: coefficient `-0.003149` (lowers CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.002854` (lowers CT win probability)
- `lag_02__T_flash_alpha_mean`: coefficient `-0.001315` (lowers CT win probability)
- `lag_01__CT5__flash`: coefficient `-0.000966` (lowers CT win probability)
- `lag_04__T_flash_alpha_mean`: coefficient `-0.000878` (lowers CT win probability)
- `lag_09__CT5__flash`: coefficient `0.000845` (raises CT win probability)
- `lag_12__CT5__flash`: coefficient `0.000768` (raises CT win probability)
- `lag_08__T_flash_alpha_mean`: coefficient `-0.000729` (lowers CT win probability)
- `lag_00__CT5__flash`: coefficient `0.000655` (raises CT win probability)
- `lag_01__CT1__flash`: coefficient `0.000624` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.006255` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.005108` (raises CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.005005` (raises CT win probability)
- `lag_12__CT_place_TUNNELSTAIRS`: coefficient `-0.004764` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.004121` (raises CT win probability)
- `lag_02__CT_place_UPPERTUNNEL`: coefficient `0.004010` (raises CT win probability)
- `lag_05__CT_place_LOWERTUNNEL`: coefficient `-0.003967` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003710` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.003026` (lowers CT win probability)
- `lag_13__CT_place_UPPERTUNNEL`: coefficient `-0.002908` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `20777`, seconds `107.00`, LSTM delta `+0.2636`

Top all feature movements:
- `lag_06__T_place_HOLE`: contribution `+0.063631`
- `lag_09__T_place_HOLE`: contribution `+0.027081`
- `lag_12__T_place_HOLE`: contribution `+0.023946`
- `lag_00__kill_diff_last_3s`: contribution `+0.015055`
- `lag_00__CT_kills_last_3s`: contribution `+0.011899`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `21257`, seconds `114.50`, LSTM delta `-0.2321`

Top all feature movements:
- `lag_05__CT_place_LOWERTUNNEL`: contribution `-0.058320`
- `lag_00__kill_diff_last_3s`: contribution `-0.015055`
- `lag_05__T_place_TUNNELSTAIRS`: contribution `-0.011843`
- `lag_00__T_kills_last_3s`: contribution `-0.011754`
- `lag_00__CT_place_LOWERTUNNEL`: contribution `-0.011645`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `22793`, seconds `138.50`, LSTM delta `+0.2300`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `+0.048517`
- `lag_13__CT_place_UPPERTUNNEL`: contribution `+0.022302`
- `lag_03__T_flash_alpha_mean`: contribution `+0.019105`
- `lag_05__CT_place_BDOORS`: contribution `+0.007755`
- `lag_00__CT_velocity_mean`: contribution `+0.007737`

Top utility-only movements:
- `lag_03__T_flash_alpha_mean`: contribution `+0.019105`

### tick `22185`, seconds `129.00`, LSTM delta `+0.2280`

Top all feature movements:
- `lag_12__CT_place_TUNNELSTAIRS`: contribution `+0.067102`
- `lag_00__kill_diff_last_3s`: contribution `+0.015055`
- `lag_00__CT_kills_last_3s`: contribution `+0.011899`
- `lag_00__damage_diff_last_5s`: contribution `+0.011524`
- `lag_12__CT_place_UPPERTUNNEL`: contribution `+0.009927`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `22441`, seconds `133.00`, LSTM delta `-0.2090`

Top all feature movements:
- `lag_02__CT_place_UPPERTUNNEL`: contribution `-0.030755`
- `lag_00__kill_diff_last_3s`: contribution `-0.015055`
- `lag_00__T_kills_last_3s`: contribution `-0.011754`
- `lag_00__damage_diff_last_5s`: contribution `-0.011524`
- `lag_09__CT_place_BDOORS`: contribution `-0.011440`

Top utility-only movements:
- No utility movement among the top local contributors.
