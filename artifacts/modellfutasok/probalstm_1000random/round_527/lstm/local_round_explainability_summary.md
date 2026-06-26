# Local Round Explainability

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-lynn-vision-vs-tyloo-bo3-tXRL8tbpb2Kb27VbUgWfK1/lynn-vision-vs-tyloo-m2-ancient.csv`
- round_num: `13`

## Largest probability jumps

- tick `131996`, seconds `36.50`, LSTM `0.1294`, delta `-0.3849`
- tick `136732`, seconds `110.50`, LSTM `0.4969`, delta `+0.2034`
- tick `132412`, seconds `43.00`, LSTM `0.4246`, delta `+0.1999`
- tick `132508`, seconds `44.50`, LSTM `0.6646`, delta `+0.1854`
- tick `133148`, seconds `54.50`, LSTM `0.5593`, delta `-0.1649`
- tick `132188`, seconds `39.50`, LSTM `0.1618`, delta `+0.1267`
- tick `136252`, seconds `103.00`, LSTM `0.1476`, delta `+0.1214`
- tick `135516`, seconds `91.50`, LSTM `0.0327`, delta `-0.0970`
- tick `136668`, seconds `109.50`, LSTM `0.3147`, delta `+0.0911`
- tick `134524`, seconds `76.00`, LSTM `0.2693`, delta `-0.0875`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.007062`, |coef| `0.007062`
- `lag_00__T_place_SIDEHALL`: coefficient `-0.005643`, |coef| `0.005643`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.005034`, |coef| `0.005034`
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.004897`, |coef| `0.004897`
- `lag_05__CT_place_TSIDELOWER`: coefficient `-0.004844`, |coef| `0.004844`
- `lag_00__CT_kills_last_3s`: coefficient `0.004617`, |coef| `0.004617`
- `lag_00__T_kills_last_3s`: coefficient `-0.004229`, |coef| `0.004229`
- `lag_00__T_velocity_mean`: coefficient `-0.003661`, |coef| `0.003661`
- `lag_00__T_place_SIDEENTRANCE`: coefficient `-0.003377`, |coef| `0.003377`
- `lag_00__damage_diff_last_5s`: coefficient `0.003170`, |coef| `0.003170`
- `lag_00__CT_place_TSIDELOWER`: coefficient `0.002961`, |coef| `0.002961`
- `lag_00__alive_diff`: coefficient `0.002902`, |coef| `0.002902`
- `lag_13__CT1__is_walking`: coefficient `-0.002471`, |coef| `0.002471`
- `lag_13__T4__alive`: coefficient `-0.002442`, |coef| `0.002442`
- `lag_05__CT1__is_walking`: coefficient `-0.002442`, |coef| `0.002442`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.005034` (lowers CT win probability)
- `lag_03__T_A_site_active_infernos`: coefficient `-0.001938` (lowers CT win probability)
- `lag_05__T4__flash_duration`: coefficient `0.001925` (raises CT win probability)
- `lag_02__T4__smoke`: coefficient `0.001473` (raises CT win probability)
- `lag_06__T2__molly`: coefficient `0.001403` (raises CT win probability)
- `lag_03__T_active_infernos`: coefficient `-0.001363` (lowers CT win probability)
- `lag_05__T_flash_alpha_mean`: coefficient `-0.001155` (lowers CT win probability)
- `lag_02__T_A_site_active_infernos`: coefficient `-0.001151` (lowers CT win probability)
- `lag_02__T_flash_alpha_mean`: coefficient `-0.001100` (lowers CT win probability)
- `lag_05__T_A_site_active_infernos`: coefficient `-0.001100` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.007062` (raises CT win probability)
- `lag_00__T_place_SIDEHALL`: coefficient `-0.005643` (lowers CT win probability)
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.004897` (raises CT win probability)
- `lag_05__CT_place_TSIDELOWER`: coefficient `-0.004844` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.004617` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.004229` (lowers CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.003661` (lowers CT win probability)
- `lag_00__T_place_SIDEENTRANCE`: coefficient `-0.003377` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003170` (raises CT win probability)
- `lag_00__CT_place_TSIDELOWER`: coefficient `0.002961` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `131996`, seconds `36.50`, LSTM delta `-0.3849`

Top all feature movements:
- `lag_05__CT_place_TSIDELOWER`: contribution `-0.065801`
- `lag_00__CT_place_TSIDELOWER`: contribution `-0.040228`
- `lag_00__kill_diff_last_3s`: contribution `-0.033996`
- `lag_00__T_kills_last_3s`: contribution `-0.026796`
- `lag_03__T3__duck_amount`: contribution `-0.006412`

Top utility-only movements:
- `lag_03__T_A_site_active_infernos`: contribution `-0.005767`

### tick `136732`, seconds `110.50`, LSTM delta `+0.2034`

Top all feature movements:
- `lag_00__T_place_SIDEHALL`: contribution `+0.036573`
- `lag_00__T_flash_alpha_mean`: contribution `+0.030544`
- `lag_00__kill_diff_last_3s`: contribution `+0.016998`
- `lag_00__CT_kills_last_3s`: contribution `+0.013330`
- `lag_15__CT_kills_last_3s`: contribution `+0.005820`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.030544`

### tick `132412`, seconds `43.00`, LSTM delta `+0.1999`

Top all feature movements:
- `lag_13__CT_place_TSIDELOWER`: contribution `+0.030570`
- `lag_00__kill_diff_last_3s`: contribution `+0.016998`
- `lag_00__T_place_SIDEENTRANCE`: contribution `+0.016482`
- `lag_00__CT_kills_last_3s`: contribution `+0.013330`
- `lag_13__T_kills_last_3s`: contribution `+0.011430`

Top utility-only movements:
- `lag_02__T4__flash_duration`: contribution `+0.004147`
- `lag_02__T_A_site_active_infernos`: contribution `+0.003425`

### tick `132508`, seconds `44.50`, LSTM delta `+0.1854`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.016998`
- `lag_00__T_place_SIDEENTRANCE`: contribution `+0.016482`
- `lag_00__CT_kills_last_3s`: contribution `+0.013330`
- `lag_10__kill_diff_last_3s`: contribution `+0.012290`
- `lag_05__T4__flash_duration`: contribution `+0.008593`

Top utility-only movements:
- `lag_05__T4__flash_duration`: contribution `+0.008593`
- `lag_05__T_A_site_active_infernos`: contribution `+0.003273`

### tick `133148`, seconds `54.50`, LSTM delta `-0.1649`

Top all feature movements:
- `lag_00__CT_place_TSIDEUPPER`: contribution `-0.036809`
- `lag_00__kill_diff_last_3s`: contribution `-0.016998`
- `lag_05__CT_place_SIDEENTRANCE`: contribution `-0.016237`
- `lag_01__CT_place_TSIDEUPPER`: contribution `-0.014345`
- `lag_05__CT_place_TSIDEUPPER`: contribution `-0.013547`

Top utility-only movements:
- No utility movement among the top local contributors.
