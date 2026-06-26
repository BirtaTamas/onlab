# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-eternal-fire-vs-fluxo-bo3-Kqy3ohBVu1ANumI6Qdn26R/eternal-fire-vs-fluxo-m2-dust2.csv`
- round_num: `16`

## Largest probability jumps

- tick `128295`, seconds `33.50`, LSTM `0.0928`, delta `-0.2267`
- tick `127719`, seconds `24.50`, LSTM `0.2824`, delta `+0.1904`
- tick `126183`, seconds `0.50`, LSTM `0.0411`, delta `-0.0472`
- tick `127911`, seconds `27.50`, LSTM `0.3086`, delta `-0.0329`
- tick `127815`, seconds `26.00`, LSTM `0.3158`, delta `+0.0324`
- tick `130279`, seconds `64.50`, LSTM `0.0161`, delta `-0.0322`
- tick `128199`, seconds `32.00`, LSTM `0.3307`, delta `+0.0295`
- tick `128135`, seconds `31.00`, LSTM `0.2862`, delta `-0.0278`
- tick `128103`, seconds `30.50`, LSTM `0.3141`, delta `+0.0274`
- tick `128583`, seconds `38.00`, LSTM `0.0764`, delta `-0.0269`

## Top 15 local ridge features

- `lag_00__CT_place_TUNNELSTAIRS`: coefficient `0.001521`, |coef| `0.001521`
- `lag_00__CT_place_OUTSIDELONG`: coefficient `0.001478`, |coef| `0.001478`
- `lag_00__T_place_TUNNELSTAIRS`: coefficient `-0.001448`, |coef| `0.001448`
- `lag_15__CT_place_LOWERTUNNEL`: coefficient `0.001396`, |coef| `0.001396`
- `lag_15__CT_place_TUNNELSTAIRS`: coefficient `-0.001377`, |coef| `0.001377`
- `lag_00__kill_diff_last_3s`: coefficient `0.001270`, |coef| `0.001270`
- `lag_02__T_place_TUNNELSTAIRS`: coefficient `0.001227`, |coef| `0.001227`
- `lag_05__CT_place_TUNNELSTAIRS`: coefficient `0.001115`, |coef| `0.001115`
- `lag_02__T4__flash_duration`: coefficient `-0.001063`, |coef| `0.001063`
- `lag_06__T_place_OUTSIDELONG`: coefficient `0.001054`, |coef| `0.001054`
- `lag_14__T3__duck_amount`: coefficient `0.001029`, |coef| `0.001029`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001028`, |coef| `0.001028`
- `lag_14__CT_place_PIT`: coefficient `-0.001013`, |coef| `0.001013`
- `lag_02__T3__duck_amount`: coefficient `0.001003`, |coef| `0.001003`
- `lag_01__CT_place_OUTSIDELONG`: coefficient `-0.000993`, |coef| `0.000993`

## Top 10 utility ridge features

- `lag_02__T4__flash_duration`: coefficient `-0.001063` (lowers CT win probability)
- `lag_03__CT2__flash_duration`: coefficient `0.000981` (raises CT win probability)
- `lag_00__CT2__molly`: coefficient `0.000881` (raises CT win probability)
- `lag_15__CT2__flash_duration`: coefficient `0.000795` (raises CT win probability)
- `lag_09__T_B_site_active_infernos`: coefficient `0.000781` (raises CT win probability)
- `lag_05__T_B_site_active_infernos`: coefficient `0.000765` (raises CT win probability)
- `lag_00__T5__utility_total`: coefficient `-0.000722` (lowers CT win probability)
- `lag_02__T_flash_duration_sum`: coefficient `-0.000720` (lowers CT win probability)
- `lag_02__T2__flash_duration`: coefficient `-0.000716` (lowers CT win probability)
- `lag_09__T1__molly`: coefficient `-0.000682` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_TUNNELSTAIRS`: coefficient `0.001521` (raises CT win probability)
- `lag_00__CT_place_OUTSIDELONG`: coefficient `0.001478` (raises CT win probability)
- `lag_00__T_place_TUNNELSTAIRS`: coefficient `-0.001448` (lowers CT win probability)
- `lag_15__CT_place_LOWERTUNNEL`: coefficient `0.001396` (raises CT win probability)
- `lag_15__CT_place_TUNNELSTAIRS`: coefficient `-0.001377` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001270` (raises CT win probability)
- `lag_02__T_place_TUNNELSTAIRS`: coefficient `0.001227` (raises CT win probability)
- `lag_05__CT_place_TUNNELSTAIRS`: coefficient `0.001115` (raises CT win probability)
- `lag_06__T_place_OUTSIDELONG`: coefficient `0.001054` (raises CT win probability)
- `lag_14__T3__duck_amount`: coefficient `0.001029` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `128295`, seconds `33.50`, LSTM delta `-0.2267`

Top all feature movements:
- `lag_15__CT_place_TUNNELSTAIRS`: contribution `-0.019389`
- `lag_05__CT_place_TUNNELSTAIRS`: contribution `-0.015710`
- `lag_00__CT_place_OUTSIDELONG`: contribution `-0.014987`
- `lag_03__CT_place_TUNNELSTAIRS`: contribution `-0.010852`
- `lag_15__CT_place_LOWERTUNNEL`: contribution `-0.010264`

Top utility-only movements:
- `lag_02__T4__flash_duration`: contribution `-0.006479`
- `lag_02__T_flash_duration_sum`: contribution `-0.002966`
- `lag_02__T2__flash_duration`: contribution `-0.002868`
- `lag_15__CT2__flash_duration`: contribution `-0.002627`

### tick `127719`, seconds `24.50`, LSTM delta `+0.1904`

Top all feature movements:
- `lag_00__T_place_TUNNELSTAIRS`: contribution `+0.010112`
- `lag_02__T_place_TUNNELSTAIRS`: contribution `+0.008568`
- `lag_14__CT_place_PIT`: contribution `+0.004363`
- `lag_02__T3__duck_amount`: contribution `+0.003783`
- `lag_07__T2__duck_amount`: contribution `+0.003323`

Top utility-only movements:
- `lag_03__CT2__flash_duration`: contribution `+0.003242`
- `lag_00__CT2__molly`: contribution `+0.002171`
- `lag_05__T_B_site_active_infernos`: contribution `+0.002164`

### tick `126183`, seconds `0.50`, LSTM delta `-0.0472`

Top all feature movements:
- `lag_01__T_place_TSPAWN`: contribution `-0.001852`
- `lag_01__CT_place_CTSPAWN`: contribution `-0.001851`
- `lag_00__T_velocity_mean`: contribution `-0.001392`
- `lag_01__utility_inv_diff`: contribution `-0.001161`
- `lag_00__CT_velocity_mean`: contribution `-0.000918`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `-0.001161`
- `lag_01__flash_inv_diff`: contribution `-0.000801`
- `lag_01__molly_inv_diff`: contribution `-0.000787`
- `lag_01__T_utility_inv`: contribution `-0.000660`
- `lag_01__smoke_inv_diff`: contribution `-0.000653`

### tick `127911`, seconds `27.50`, LSTM delta `-0.0329`

Top all feature movements:
- `lag_03__CT_place_TUNNELSTAIRS`: contribution `-0.010852`
- `lag_03__CT_place_LOWERTUNNEL`: contribution `-0.006689`
- `lag_03__CT2__flash_duration`: contribution `-0.003242`
- `lag_02__T3__duck_amount`: contribution `-0.003111`
- `lag_00__kill_diff_last_3s`: contribution `-0.003056`

Top utility-only movements:
- `lag_03__CT2__flash_duration`: contribution `-0.003242`

### tick `127815`, seconds `26.00`, LSTM delta `+0.0324`

Top all feature movements:
- `lag_00__CT_place_TUNNELSTAIRS`: contribution `+0.021419`
- `lag_00__CT_place_LOWERTUNNEL`: contribution `+0.002574`
- `lag_08__T4__duck_amount`: contribution `-0.002178`
- `lag_05__T3__is_walking`: contribution `-0.001590`
- `lag_03__T_place_TUNNELSTAIRS`: contribution `-0.001495`

Top utility-only movements:
- `lag_08__T_B_site_active_infernos`: contribution `+0.001173`
- `lag_06__CT2__flash_duration`: contribution `+0.001165`
- `lag_03__CT2__molly`: contribution `+0.000840`
