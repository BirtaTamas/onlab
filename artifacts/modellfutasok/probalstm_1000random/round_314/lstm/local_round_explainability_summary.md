# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-mouz-vs-falcons-bo3-xBECUqZMcQ8GCwi-GUyz8e/mouz-vs-falcons-m1-train.csv`
- round_num: `21`

## Largest probability jumps

- tick `215469`, seconds `94.50`, LSTM `0.5818`, delta `+0.4099`
- tick `210701`, seconds `20.00`, LSTM `0.3983`, delta `+0.3617`
- tick `216525`, seconds `111.00`, LSTM `0.5053`, delta `-0.3179`
- tick `210541`, seconds `17.50`, LSTM `0.1230`, delta `-0.2633`
- tick `210477`, seconds `16.50`, LSTM `0.4597`, delta `-0.2543`
- tick `216557`, seconds `111.50`, LSTM `0.7294`, delta `+0.2241`
- tick `216589`, seconds `112.00`, LSTM `0.5695`, delta `-0.1599`
- tick `210221`, seconds `12.50`, LSTM `0.6915`, delta `+0.1578`
- tick `210317`, seconds `14.00`, LSTM `0.8412`, delta `+0.1297`
- tick `212397`, seconds `46.50`, LSTM `0.3970`, delta `+0.0981`

## Top 15 local ridge features

- `lag_07__CT_place_TMAIN`: coefficient `-0.007743`, |coef| `0.007743`
- `lag_12__T_bomb_zone_count`: coefficient `-0.004871`, |coef| `0.004871`
- `lag_00__kill_diff_last_3s`: coefficient `0.004745`, |coef| `0.004745`
- `lag_08__CT_place_TMAIN`: coefficient `-0.003842`, |coef| `0.003842`
- `lag_00__damage_diff_last_5s`: coefficient `0.003697`, |coef| `0.003697`
- `lag_00__CT_kills_last_3s`: coefficient `0.003639`, |coef| `0.003639`
- `lag_06__CT_place_ELECTRICALBOX`: coefficient `-0.003503`, |coef| `0.003503`
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.003336`, |coef| `0.003336`
- `lag_12__T4__flash_duration`: coefficient `-0.003317`, |coef| `0.003317`
- `lag_00__CT_place_BACKOFB`: coefficient `-0.003154`, |coef| `0.003154`
- `lag_00__T_place_ALLEY`: coefficient `-0.002735`, |coef| `0.002735`
- `lag_01__CT_place_BACKOFB`: coefficient `-0.002673`, |coef| `0.002673`
- `lag_00__T_duck_amount_mean`: coefficient `-0.002499`, |coef| `0.002499`
- `lag_12__T2__has_bomb`: coefficient `-0.002487`, |coef| `0.002487`
- `lag_01__CT_place_TSIDEUPPER`: coefficient `0.002436`, |coef| `0.002436`

## Top 10 utility ridge features

- `lag_12__T4__flash_duration`: coefficient `-0.003317` (lowers CT win probability)
- `lag_02__CT5__smoke`: coefficient `0.002211` (raises CT win probability)
- `lag_00__T4__flash_duration`: coefficient `0.001951` (raises CT win probability)
- `lag_05__T4__flash_duration`: coefficient `0.001730` (raises CT win probability)
- `lag_12__T_flash_duration_sum`: coefficient `-0.001633` (lowers CT win probability)
- `lag_12__T_B_site_active_infernos`: coefficient `-0.001587` (lowers CT win probability)
- `lag_11__T4__flash_duration`: coefficient `-0.001538` (lowers CT win probability)
- `lag_00__T2__flash`: coefficient `-0.001349` (lowers CT win probability)
- `lag_03__CT1__flash_duration`: coefficient `0.001349` (raises CT win probability)
- `lag_10__T4__flash_duration`: coefficient `-0.001233` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_07__CT_place_TMAIN`: coefficient `-0.007743` (lowers CT win probability)
- `lag_12__T_bomb_zone_count`: coefficient `-0.004871` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.004745` (raises CT win probability)
- `lag_08__CT_place_TMAIN`: coefficient `-0.003842` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003697` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003639` (raises CT win probability)
- `lag_06__CT_place_ELECTRICALBOX`: coefficient `-0.003503` (lowers CT win probability)
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.003336` (raises CT win probability)
- `lag_00__CT_place_BACKOFB`: coefficient `-0.003154` (lowers CT win probability)
- `lag_00__T_place_ALLEY`: coefficient `-0.002735` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `215469`, seconds `94.50`, LSTM delta `+0.4099`

Top all feature movements:
- `lag_07__CT_place_TMAIN`: contribution `+0.085802`
- `lag_12__T_bomb_zone_count`: contribution `+0.028355`
- `lag_00__kill_diff_last_3s`: contribution `+0.011422`
- `lag_00__CT_kills_last_3s`: contribution `+0.010505`
- `lag_00__CT1__duck_amount`: contribution `+0.008959`

Top utility-only movements:
- `lag_02__CT5__smoke`: contribution `+0.004850`
- `lag_12__T_B_site_active_infernos`: contribution `+0.004486`

### tick `210701`, seconds `20.00`, LSTM delta `+0.3617`

Top all feature movements:
- `lag_06__CT_place_ELECTRICALBOX`: contribution `+0.040719`
- `lag_00__damage_diff_last_5s`: contribution `+0.012426`
- `lag_00__kill_diff_last_3s`: contribution `+0.011422`
- `lag_00__CT_kills_last_3s`: contribution `+0.010505`
- `lag_00__CT_shots_fired_sum`: contribution `+0.007433`

Top utility-only movements:
- `lag_02__T_A_site_active_infernos`: contribution `+0.007333`
- `lag_08__CT1__flash_duration`: contribution `+0.006320`
- `lag_13__CT2__flash_duration`: contribution `+0.004676`
- `lag_09__CT2__flash_duration`: contribution `+0.004380`
- `lag_13__T5__flash_duration`: contribution `+0.004379`

### tick `216525`, seconds `111.00`, LSTM delta `-0.3179`

Top all feature movements:
- `lag_06__CT_place_ELECTRICALBOX`: contribution `-0.040719`
- `lag_11__CT_place_ELECTRICALBOX`: contribution `-0.027375`
- `lag_12__T4__flash_duration`: contribution `-0.025916`
- `lag_00__kill_diff_last_3s`: contribution `-0.011422`
- `lag_05__CT_place_ELECTRICALBOX`: contribution `-0.010817`

Top utility-only movements:
- `lag_12__T4__flash_duration`: contribution `-0.025916`
- `lag_12__T_flash_duration_sum`: contribution `-0.005188`

### tick `210541`, seconds `17.50`, LSTM delta `-0.2633`

Top all feature movements:
- `lag_08__CT_place_ELECTRICALBOX`: contribution `-0.014092`
- `lag_00__damage_diff_last_5s`: contribution `-0.013093`
- `lag_00__kill_diff_last_3s`: contribution `-0.011422`
- `lag_12__CT_place_ELECTRICALBOX`: contribution `+0.008550`
- `lag_01__CT_place_ELECTRICALBOX`: contribution `-0.008114`

Top utility-only movements:
- `lag_03__CT1__flash_duration`: contribution `-0.007647`
- `lag_11__CT3__flash_duration`: contribution `-0.005655`
- `lag_11__T_B_site_active_infernos`: contribution `-0.005330`
- `lag_11__T_A_site_active_infernos`: contribution `-0.005061`
- `lag_07__T3__flash_duration`: contribution `-0.004531`

### tick `210477`, seconds `16.50`, LSTM delta `-0.2543`

Top all feature movements:
- `lag_06__CT_place_ELECTRICALBOX`: contribution `-0.040719`
- `lag_10__CT_place_ELECTRICALBOX`: contribution `-0.013379`
- `lag_00__kill_diff_last_3s`: contribution `-0.011422`
- `lag_12__CT_place_ELECTRICALBOX`: contribution `-0.008550`
- `lag_00__damage_diff_last_5s`: contribution `-0.008339`

Top utility-only movements:
- `lag_13__T3__flash_duration`: contribution `-0.006175`
- `lag_05__T3__flash_duration`: contribution `-0.005402`
- `lag_09__T_A_site_active_infernos`: contribution `-0.003909`
