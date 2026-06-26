# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-3dmax-bo3-u02WLpVJ6Q22MzSL2B_-Tu/the-mongolz-vs-3dmax-m2-ancient.csv`
- round_num: `14`

## Largest probability jumps

- tick `96474`, seconds `59.00`, LSTM `0.7558`, delta `+0.3321`
- tick `93562`, seconds `13.50`, LSTM `0.3470`, delta `-0.2237`
- tick `96154`, seconds `54.00`, LSTM `0.4396`, delta `-0.2163`
- tick `94906`, seconds `34.50`, LSTM `0.3839`, delta `+0.2073`
- tick `96090`, seconds `53.00`, LSTM `0.5228`, delta `+0.1834`
- tick `96122`, seconds `53.50`, LSTM `0.6559`, delta `+0.1331`
- tick `93594`, seconds `14.00`, LSTM `0.2150`, delta `-0.1320`
- tick `94010`, seconds `20.50`, LSTM `0.2344`, delta `-0.1122`
- tick `93946`, seconds `19.50`, LSTM `0.2794`, delta `+0.0831`
- tick `95162`, seconds `38.50`, LSTM `0.3166`, delta `+0.0801`

## Top 15 local ridge features

- `lag_08__CT_place_TUNNEL`: coefficient `-0.005327`, |coef| `0.005327`
- `lag_00__CT_shots_fired_sum`: coefficient `0.004679`, |coef| `0.004679`
- `lag_05__CT_place_TUNNEL`: coefficient `0.003596`, |coef| `0.003596`
- `lag_00__kill_diff_last_3s`: coefficient `0.003557`, |coef| `0.003557`
- `lag_06__CT_place_TUNNEL`: coefficient `0.003260`, |coef| `0.003260`
- `lag_07__CT_place_TUNNEL`: coefficient `-0.003101`, |coef| `0.003101`
- `lag_00__CT_kills_last_3s`: coefficient `0.002887`, |coef| `0.002887`
- `lag_00__damage_diff_last_5s`: coefficient `0.002864`, |coef| `0.002864`
- `lag_05__CT_place_TSPAWN`: coefficient `-0.002856`, |coef| `0.002856`
- `lag_01__damage_diff_last_5s`: coefficient `0.002648`, |coef| `0.002648`
- `lag_00__T_place_ALLEY`: coefficient `-0.002492`, |coef| `0.002492`
- `lag_10__CT2__is_walking`: coefficient `-0.002433`, |coef| `0.002433`
- `lag_00__CT_damage_last_5s`: coefficient `0.002292`, |coef| `0.002292`
- `lag_09__CT2__duck_amount`: coefficient `-0.002261`, |coef| `0.002261`
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.002216`, |coef| `0.002216`

## Top 10 utility ridge features

- `lag_00__CT3__flash_duration`: coefficient `0.001095` (raises CT win probability)
- `lag_02__CT3__flash_duration`: coefficient `-0.001077` (lowers CT win probability)
- `lag_15__T3__smoke`: coefficient `-0.000943` (lowers CT win probability)
- `lag_10__CT4__smoke`: coefficient `-0.000856` (lowers CT win probability)
- `lag_12__T_A_site_active_smokes`: coefficient `-0.000834` (lowers CT win probability)
- `lag_02__CT1__smoke`: coefficient `-0.000796` (lowers CT win probability)
- `lag_00__T5__flash`: coefficient `-0.000791` (lowers CT win probability)
- `lag_14__CT3__flash_duration`: coefficient `0.000754` (raises CT win probability)
- `lag_13__T_B_site_active_smokes`: coefficient `0.000750` (raises CT win probability)
- `lag_14__T3__smoke`: coefficient `-0.000737` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_08__CT_place_TUNNEL`: coefficient `-0.005327` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.004679` (raises CT win probability)
- `lag_05__CT_place_TUNNEL`: coefficient `0.003596` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003557` (raises CT win probability)
- `lag_06__CT_place_TUNNEL`: coefficient `0.003260` (raises CT win probability)
- `lag_07__CT_place_TUNNEL`: coefficient `-0.003101` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002887` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002864` (raises CT win probability)
- `lag_05__CT_place_TSPAWN`: coefficient `-0.002856` (lowers CT win probability)
- `lag_01__damage_diff_last_5s`: coefficient `0.002648` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `96474`, seconds `59.00`, LSTM delta `+0.3321`

Top all feature movements:
- `lag_08__CT_place_TUNNEL`: contribution `+0.085560`
- `lag_10__CT_place_TSIDEUPPER`: contribution `+0.015662`
- `lag_08__CT_place_WATER`: contribution `+0.013295`
- `lag_00__CT_shots_fired_sum`: contribution `+0.009751`
- `lag_00__kill_diff_last_3s`: contribution `+0.008561`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `93562`, seconds `13.50`, LSTM delta `-0.2237`

Top all feature movements:
- `lag_15__T_place_WATER`: contribution `-0.023333`
- `lag_00__CT_shots_fired_sum`: contribution `-0.019502`
- `lag_05__T_place_RAMP`: contribution `-0.010167`
- `lag_15__T_place_RUINS`: contribution `-0.008634`
- `lag_00__kill_diff_last_3s`: contribution `-0.008561`

Top utility-only movements:
- `lag_00__CT3__flash_duration`: contribution `-0.006844`
- `lag_02__CT3__flash_duration`: contribution `-0.006733`

### tick `96154`, seconds `54.00`, LSTM delta `-0.2163`

Top all feature movements:
- `lag_07__CT_place_TUNNEL`: contribution `-0.049804`
- `lag_00__CT_place_TSIDEUPPER`: contribution `-0.016659`
- `lag_13__CT_place_TSPAWN`: contribution `-0.016316`
- `lag_07__CT_place_TSPAWN`: contribution `-0.009244`
- `lag_09__CT2__duck_amount`: contribution `-0.008615`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `94906`, seconds `34.50`, LSTM delta `+0.2073`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.016252`
- `lag_00__T_place_ALLEY`: contribution `+0.010557`
- `lag_00__kill_diff_last_3s`: contribution `+0.008561`
- `lag_00__CT_kills_last_3s`: contribution `+0.008335`
- `lag_04__CT1__duck_amount`: contribution `+0.006342`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `96090`, seconds `53.00`, LSTM delta `+0.1834`

Top all feature movements:
- `lag_05__CT_place_TUNNEL`: contribution `+0.057756`
- `lag_05__CT_place_TSPAWN`: contribution `+0.021381`
- `lag_11__CT_place_TSPAWN`: contribution `+0.016153`
- `lag_09__CT2__duck_amount`: contribution `+0.008615`
- `lag_00__damage_diff_last_5s`: contribution `+0.006333`

Top utility-only movements:
- No utility movement among the top local contributors.
