# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-saw-vs-big-bo3-Eh5yMCium2D2NNwnLk7jHb/saw-vs-big-m1-ancient.csv`
- round_num: `9`

## Largest probability jumps

- tick `83822`, seconds `28.50`, LSTM `0.1775`, delta `-0.3906`
- tick `83438`, seconds `22.50`, LSTM `0.4877`, delta `-0.2848`
- tick `83086`, seconds `17.00`, LSTM `0.4160`, delta `+0.2416`
- tick `83342`, seconds `21.00`, LSTM `0.7248`, delta `+0.1731`
- tick `82958`, seconds `15.00`, LSTM `0.3039`, delta `-0.1528`
- tick `82990`, seconds `15.50`, LSTM `0.2230`, delta `-0.0809`
- tick `83854`, seconds `29.00`, LSTM `0.1143`, delta `-0.0632`
- tick `83150`, seconds `18.00`, LSTM `0.5016`, delta `+0.0536`
- tick `83886`, seconds `29.50`, LSTM `0.0614`, delta `-0.0529`
- tick `83374`, seconds `21.50`, LSTM `0.7763`, delta `+0.0515`

## Top 15 local ridge features

- `lag_12__CT_shots_fired_sum`: coefficient `0.004024`, |coef| `0.004024`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002919`, |coef| `0.002919`
- `lag_00__T3__is_scoped`: coefficient `0.002522`, |coef| `0.002522`
- `lag_15__CT3__shots_fired`: coefficient `-0.002310`, |coef| `0.002310`
- `lag_00__kill_diff_last_3s`: coefficient `0.002229`, |coef| `0.002229`
- `lag_14__CT3__shots_fired`: coefficient `-0.002167`, |coef| `0.002167`
- `lag_10__T5__flash_duration`: coefficient `0.002158`, |coef| `0.002158`
- `lag_12__CT_place_TSIDEUPPER`: coefficient `0.002071`, |coef| `0.002071`
- `lag_15__T4__flash_duration`: coefficient `0.002056`, |coef| `0.002056`
- `lag_09__T3__is_scoped`: coefficient `-0.001992`, |coef| `0.001992`
- `lag_08__CT_shots_fired_sum`: coefficient `-0.001967`, |coef| `0.001967`
- `lag_00__T_kills_last_3s`: coefficient `-0.001864`, |coef| `0.001864`
- `lag_13__CT3__shots_fired`: coefficient `-0.001856`, |coef| `0.001856`
- `lag_15__T_shots_fired_sum`: coefficient `0.001669`, |coef| `0.001669`
- `lag_08__CT4__shots_fired`: coefficient `-0.001653`, |coef| `0.001653`

## Top 10 utility ridge features

- `lag_10__T5__flash_duration`: coefficient `0.002158` (raises CT win probability)
- `lag_15__T4__flash_duration`: coefficient `0.002056` (raises CT win probability)
- `lag_15__CT1__flash_duration`: coefficient `0.001320` (raises CT win probability)
- `lag_11__CT_B_site_active_infernos`: coefficient `-0.001318` (lowers CT win probability)
- `lag_11__T1__flash_duration`: coefficient `-0.001281` (lowers CT win probability)
- `lag_15__T_flash_duration_sum`: coefficient `0.001235` (raises CT win probability)
- `lag_00__CT_B_site_active_infernos`: coefficient `0.001211` (raises CT win probability)
- `lag_03__T4__flash_duration`: coefficient `0.001142` (raises CT win probability)
- `lag_12__T1__flash_duration`: coefficient `-0.000998` (lowers CT win probability)
- `lag_08__CT1__flash_duration`: coefficient `-0.000985` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_12__CT_shots_fired_sum`: coefficient `0.004024` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002919` (raises CT win probability)
- `lag_00__T3__is_scoped`: coefficient `0.002522` (raises CT win probability)
- `lag_15__CT3__shots_fired`: coefficient `-0.002310` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002229` (raises CT win probability)
- `lag_14__CT3__shots_fired`: coefficient `-0.002167` (lowers CT win probability)
- `lag_12__CT_place_TSIDEUPPER`: coefficient `0.002071` (raises CT win probability)
- `lag_09__T3__is_scoped`: coefficient `-0.001992` (lowers CT win probability)
- `lag_08__CT_shots_fired_sum`: coefficient `-0.001967` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001864` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `83822`, seconds `28.50`, LSTM delta `-0.3906`

Top all feature movements:
- `lag_12__CT_shots_fired_sum`: contribution `-0.055907`
- `lag_00__T3__is_scoped`: contribution `-0.016181`
- `lag_12__CT_place_TSIDEUPPER`: contribution `-0.015564`
- `lag_09__T3__is_scoped`: contribution `-0.012775`
- `lag_10__T5__flash_duration`: contribution `-0.012561`

Top utility-only movements:
- `lag_10__T5__flash_duration`: contribution `-0.012561`
- `lag_15__T4__flash_duration`: contribution `-0.012285`
- `lag_15__CT1__flash_duration`: contribution `-0.003460`

### tick `83438`, seconds `22.50`, LSTM delta `-0.2848`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.040558`
- `lag_00__CT_place_TSIDEUPPER`: contribution `-0.009246`
- `lag_15__T_shots_fired_sum`: contribution `-0.008758`
- `lag_05__CT_place_SIDEHALL`: contribution `-0.008511`
- `lag_03__T4__flash_duration`: contribution `-0.006822`

Top utility-only movements:
- `lag_03__T4__flash_duration`: contribution `-0.006822`
- `lag_11__CT_B_site_active_infernos`: contribution `-0.004527`
- `lag_00__CT_B_site_active_infernos`: contribution `-0.004159`
- `lag_09__T4__flash_duration`: contribution `-0.003839`
- `lag_09__CT1__flash_duration`: contribution `-0.003760`

### tick `83086`, seconds `17.00`, LSTM delta `+0.2416`

Top all feature movements:
- `lag_08__CT_shots_fired_sum`: contribution `+0.031435`
- `lag_08__CT4__shots_fired`: contribution `+0.014249`
- `lag_00__CT_shots_fired_sum`: contribution `+0.008112`
- `lag_00__kill_diff_last_3s`: contribution `+0.005364`
- `lag_12__CT1__flash_duration`: contribution `+0.005064`

Top utility-only movements:
- `lag_12__CT1__flash_duration`: contribution `+0.005064`
- `lag_00__CT_B_site_active_infernos`: contribution `+0.004159`
- `lag_15__T_flash_duration_sum`: contribution `+0.004128`
- `lag_14__CT_B_site_active_infernos`: contribution `+0.003128`
- `lag_15__T1__flash_duration`: contribution `+0.002779`

### tick `83342`, seconds `21.00`, LSTM delta `+0.1731`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.010139`
- `lag_02__CT_place_SIDEHALL`: contribution `+0.008784`
- `lag_06__CT1__flash_duration`: contribution `+0.005559`
- `lag_08__CT_shots_fired_sum`: contribution `-0.005467`
- `lag_00__kill_diff_last_3s`: contribution `+0.005364`

Top utility-only movements:
- `lag_06__CT1__flash_duration`: contribution `+0.005559`
- `lag_06__T4__flash_duration`: contribution `+0.004981`
- `lag_11__CT_B_site_active_infernos`: contribution `+0.004527`
- `lag_05__T5__flash_duration`: contribution `+0.003646`
- `lag_00__T4__flash_duration`: contribution `+0.002334`

### tick `82958`, seconds `15.00`, LSTM delta `-0.1528`

Top all feature movements:
- `lag_04__CT_shots_fired_sum`: contribution `-0.015037`
- `lag_11__T1__flash_duration`: contribution `-0.008653`
- `lag_04__CT4__shots_fired`: contribution `-0.007581`
- `lag_00__T_kills_last_3s`: contribution `-0.005906`
- `lag_08__CT1__flash_duration`: contribution `-0.005895`

Top utility-only movements:
- `lag_11__T1__flash_duration`: contribution `-0.008653`
- `lag_08__CT1__flash_duration`: contribution `-0.005895`
- `lag_14__CT_B_site_active_infernos`: contribution `+0.003128`
