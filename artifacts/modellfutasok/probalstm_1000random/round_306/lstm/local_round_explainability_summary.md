# Local Round Explainability

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-falcons-vs-g2-bo3-Xf_UKx9fB2btv0Vy_VBVUC/falcons-vs-g2-m3-mirage.csv`
- round_num: `22`

## Largest probability jumps

- tick `165752`, seconds `64.00`, LSTM `0.1182`, delta `-0.2723`
- tick `164312`, seconds `41.50`, LSTM `0.4874`, delta `-0.1973`
- tick `163704`, seconds `32.00`, LSTM `0.8879`, delta `+0.1973`
- tick `163896`, seconds `35.00`, LSTM `0.7152`, delta `-0.1890`
- tick `163576`, seconds `30.00`, LSTM `0.6873`, delta `+0.1404`
- tick `163544`, seconds `29.50`, LSTM `0.5469`, delta `+0.1390`
- tick `164344`, seconds `42.00`, LSTM `0.3708`, delta `-0.1165`
- tick `163192`, seconds `24.00`, LSTM `0.4960`, delta `-0.1000`
- tick `163128`, seconds `23.00`, LSTM `0.6206`, delta `+0.0768`
- tick `163512`, seconds `29.00`, LSTM `0.4078`, delta `+0.0705`

## Top 15 local ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.005178`, |coef| `0.005178`
- `lag_00__damage_diff_last_5s`: coefficient `0.005074`, |coef| `0.005074`
- `lag_01__CT_place_SHOP`: coefficient `0.005030`, |coef| `0.005030`
- `lag_00__kill_diff_last_3s`: coefficient `0.004915`, |coef| `0.004915`
- `lag_00__CT2__alive`: coefficient `0.004580`, |coef| `0.004580`
- `lag_00__CT2__armor`: coefficient `0.004127`, |coef| `0.004127`
- `lag_00__T_damage_last_5s`: coefficient `-0.003900`, |coef| `0.003900`
- `lag_00__CT2__hp`: coefficient `0.003378`, |coef| `0.003378`
- `lag_05__CT2__is_walking`: coefficient `0.003074`, |coef| `0.003074`
- `lag_00__T1__is_walking`: coefficient `0.002856`, |coef| `0.002856`
- `lag_05__CT1__duck_amount`: coefficient `-0.002721`, |coef| `0.002721`
- `lag_12__T1__duck_amount`: coefficient `-0.002706`, |coef| `0.002706`
- `lag_04__kill_diff_last_3s`: coefficient `0.002585`, |coef| `0.002585`
- `lag_11__CT2__duck_amount`: coefficient `-0.002555`, |coef| `0.002555`
- `lag_06__CT2__is_walking`: coefficient `-0.002514`, |coef| `0.002514`

## Top 10 utility ridge features

- `lag_00__CT4__flash`: coefficient `0.001231` (raises CT win probability)
- `lag_09__T1__flash_duration`: coefficient `-0.001204` (lowers CT win probability)
- `lag_12__CT2__flash_duration`: coefficient `-0.001119` (lowers CT win probability)
- `lag_13__CT_active_smokes`: coefficient `0.001027` (raises CT win probability)
- `lag_01__CT4__flash`: coefficient `0.000998` (raises CT win probability)
- `lag_00__CT4__utility_total`: coefficient `0.000991` (raises CT win probability)
- `lag_13__T2__flash_duration`: coefficient `-0.000932` (lowers CT win probability)
- `lag_14__CT_A_site_active_smokes`: coefficient `0.000931` (raises CT win probability)
- `lag_14__CT_active_smokes`: coefficient `0.000921` (raises CT win probability)
- `lag_15__T_B_site_active_smokes`: coefficient `0.000918` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.005178` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.005074` (raises CT win probability)
- `lag_01__CT_place_SHOP`: coefficient `0.005030` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.004915` (raises CT win probability)
- `lag_00__CT2__alive`: coefficient `0.004580` (raises CT win probability)
- `lag_00__CT2__armor`: coefficient `0.004127` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.003900` (lowers CT win probability)
- `lag_00__CT2__hp`: coefficient `0.003378` (raises CT win probability)
- `lag_05__CT2__is_walking`: coefficient `0.003074` (raises CT win probability)
- `lag_00__T1__is_walking`: coefficient `0.002856` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `165752`, seconds `64.00`, LSTM delta `-0.2723`

Top all feature movements:
- `lag_01__CT_place_SHOP`: contribution `-0.025229`
- `lag_00__T_kills_last_3s`: contribution `-0.016405`
- `lag_00__kill_diff_last_3s`: contribution `-0.011829`
- `lag_00__CT2__alive`: contribution `-0.011090`
- `lag_11__CT2__duck_amount`: contribution `-0.009733`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `164312`, seconds `41.50`, LSTM delta `-0.1973`

Top all feature movements:
- `lag_08__T_place_TRUCK`: contribution `-0.041745`
- `lag_00__T_kills_last_3s`: contribution `-0.016405`
- `lag_00__kill_diff_last_3s`: contribution `-0.011829`
- `lag_00__damage_diff_last_5s`: contribution `-0.008356`
- `lag_09__T1__duck_amount`: contribution `+0.007748`

Top utility-only movements:
- `lag_00__CT4__flash`: contribution `-0.004270`

### tick `163704`, seconds `32.00`, LSTM delta `+0.1973`

Top all feature movements:
- `lag_14__CT_shots_fired_sum`: contribution `+0.036874`
- `lag_01__CT_place_SHOP`: contribution `-0.025229`
- `lag_14__CT2__shots_fired`: contribution `+0.018273`
- `lag_00__kill_diff_last_3s`: contribution `+0.011829`
- `lag_04__CT2__shots_fired`: contribution `+0.008937`

Top utility-only movements:
- `lag_12__CT2__flash_duration`: contribution `+0.007942`
- `lag_13__T2__flash_duration`: contribution `+0.004100`

### tick `163896`, seconds `35.00`, LSTM delta `-0.1890`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.023659`
- `lag_00__damage_diff_last_5s`: contribution `-0.018773`
- `lag_00__T_kills_last_3s`: contribution `-0.016405`
- `lag_07__CT_place_SHOP`: contribution `-0.009382`
- `lag_01__CT2__duck_amount`: contribution `-0.009182`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `163576`, seconds `30.00`, LSTM delta `+0.1404`

Top all feature movements:
- `lag_10__CT_shots_fired_sum`: contribution `+0.015705`
- `lag_00__kill_diff_last_3s`: contribution `+0.011829`
- `lag_00__damage_diff_last_5s`: contribution `+0.010073`
- `lag_11__CT2__duck_amount`: contribution `+0.009733`
- `lag_10__CT_place_SHOP`: contribution `-0.008550`

Top utility-only movements:
- `lag_08__CT2__flash_duration`: contribution `+0.005841`
