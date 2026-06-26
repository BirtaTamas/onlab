# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-eternal-fire-vs-falcons-bo3-Bm3FkXiO5h_cvpKxUnOmaW/eternal-fire-vs-falcons-m1-inferno.csv`
- round_num: `10`

## Largest probability jumps

- tick `87911`, seconds `16.00`, LSTM `0.2486`, delta `-0.2412`
- tick `88071`, seconds `18.50`, LSTM `0.0678`, delta `-0.1440`
- tick `87943`, seconds `16.50`, LSTM `0.1865`, delta `-0.0620`
- tick `89383`, seconds `39.00`, LSTM `0.0274`, delta `-0.0573`
- tick `87207`, seconds `5.00`, LSTM `0.4813`, delta `+0.0383`
- tick `89511`, seconds `41.00`, LSTM `0.0059`, delta `-0.0332`
- tick `89351`, seconds `38.50`, LSTM `0.0847`, delta `+0.0314`
- tick `87751`, seconds `13.50`, LSTM `0.4839`, delta `-0.0201`
- tick `89447`, seconds `40.00`, LSTM `0.0451`, delta `+0.0192`
- tick `89159`, seconds `35.50`, LSTM `0.0516`, delta `-0.0178`

## Top 15 local ridge features

- `lag_07__CT_place_APARTMENTS`: coefficient `-0.001656`, |coef| `0.001656`
- `lag_10__CT_place_BALCONY`: coefficient `-0.001625`, |coef| `0.001625`
- `lag_08__CT_place_BALCONY`: coefficient `0.001522`, |coef| `0.001522`
- `lag_10__CT2__flash_duration`: coefficient `-0.001507`, |coef| `0.001507`
- `lag_08__CT_place_APARTMENTS`: coefficient `-0.001397`, |coef| `0.001397`
- `lag_07__CT_place_BALCONY`: coefficient `0.001353`, |coef| `0.001353`
- `lag_03__T3__flash_duration`: coefficient `-0.001295`, |coef| `0.001295`
- `lag_01__CT4__is_scoped`: coefficient `-0.001099`, |coef| `0.001099`
- `lag_00__T_kills_last_3s`: coefficient `-0.001074`, |coef| `0.001074`
- `lag_09__CT_place_BALCONY`: coefficient `-0.000971`, |coef| `0.000971`
- `lag_00__CT4__utility_total`: coefficient `0.000929`, |coef| `0.000929`
- `lag_10__CT_place_TOPOFMID`: coefficient `-0.000903`, |coef| `0.000903`
- `lag_00__kill_diff_last_3s`: coefficient `0.000898`, |coef| `0.000898`
- `lag_00__T_damage_last_5s`: coefficient `-0.000859`, |coef| `0.000859`
- `lag_04__T3__flash_duration`: coefficient `-0.000839`, |coef| `0.000839`

## Top 10 utility ridge features

- `lag_10__CT2__flash_duration`: coefficient `-0.001507` (lowers CT win probability)
- `lag_03__T3__flash_duration`: coefficient `-0.001295` (lowers CT win probability)
- `lag_00__CT4__utility_total`: coefficient `0.000929` (raises CT win probability)
- `lag_04__T3__flash_duration`: coefficient `-0.000839` (lowers CT win probability)
- `lag_00__CT4__molly`: coefficient `0.000820` (raises CT win probability)
- `lag_02__CT_A_site_active_infernos`: coefficient `-0.000762` (lowers CT win probability)
- `lag_15__CT2__flash_duration`: coefficient `-0.000736` (lowers CT win probability)
- `lag_00__CT4__smoke`: coefficient `0.000727` (raises CT win probability)
- `lag_08__T3__flash_duration`: coefficient `-0.000715` (lowers CT win probability)
- `lag_10__CT_flash_duration_sum`: coefficient `-0.000668` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_07__CT_place_APARTMENTS`: coefficient `-0.001656` (lowers CT win probability)
- `lag_10__CT_place_BALCONY`: coefficient `-0.001625` (lowers CT win probability)
- `lag_08__CT_place_BALCONY`: coefficient `0.001522` (raises CT win probability)
- `lag_08__CT_place_APARTMENTS`: coefficient `-0.001397` (lowers CT win probability)
- `lag_07__CT_place_BALCONY`: coefficient `0.001353` (raises CT win probability)
- `lag_01__CT4__is_scoped`: coefficient `-0.001099` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001074` (lowers CT win probability)
- `lag_09__CT_place_BALCONY`: coefficient `-0.000971` (lowers CT win probability)
- `lag_10__CT_place_TOPOFMID`: coefficient `-0.000903` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000898` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `87911`, seconds `16.00`, LSTM delta `-0.2412`

Top all feature movements:
- `lag_07__CT_place_APARTMENTS`: contribution `-0.012725`
- `lag_10__CT2__flash_duration`: contribution `-0.010647`
- `lag_10__CT_place_BALCONY`: contribution `-0.010429`
- `lag_08__CT_place_BALCONY`: contribution `-0.009766`
- `lag_07__CT_place_BALCONY`: contribution `-0.008684`

Top utility-only movements:
- `lag_10__CT2__flash_duration`: contribution `-0.010647`
- `lag_03__T3__flash_duration`: contribution `-0.006483`
- `lag_02__CT_A_site_active_infernos`: contribution `-0.002688`
- `lag_00__CT4__utility_total`: contribution `-0.002593`

### tick `88071`, seconds `18.50`, LSTM delta `-0.1440`

Top all feature movements:
- `lag_12__CT_place_APARTMENTS`: contribution `-0.005513`
- `lag_15__CT2__flash_duration`: contribution `-0.005198`
- `lag_13__CT_place_BALCONY`: contribution `-0.003777`
- `lag_08__T3__flash_duration`: contribution `-0.003579`
- `lag_02__CT2__flash_duration`: contribution `-0.003506`

Top utility-only movements:
- `lag_15__CT2__flash_duration`: contribution `-0.005198`
- `lag_08__T3__flash_duration`: contribution `-0.003579`
- `lag_02__CT2__flash_duration`: contribution `-0.003506`

### tick `87943`, seconds `16.50`, LSTM delta `-0.0620`

Top all feature movements:
- `lag_08__CT_place_APARTMENTS`: contribution `-0.010735`
- `lag_10__CT_place_BALCONY`: contribution `-0.010429`
- `lag_08__CT_place_BALCONY`: contribution `-0.009766`
- `lag_09__CT_place_BALCONY`: contribution `+0.006229`
- `lag_11__CT2__flash_duration`: contribution `-0.004351`

Top utility-only movements:
- `lag_11__CT2__flash_duration`: contribution `-0.004351`
- `lag_04__T3__flash_duration`: contribution `-0.004204`

### tick `89383`, seconds `39.00`, LSTM delta `-0.0573`

Top all feature movements:
- `lag_00__CT3__flash_duration`: contribution `-0.005065`
- `lag_00__T3__flash_duration`: contribution `-0.004441`
- `lag_00__T_kills_last_3s`: contribution `-0.003401`
- `lag_00__CT_shots_fired_sum`: contribution `-0.003302`
- `lag_15__T2__is_scoped`: contribution `-0.002462`

Top utility-only movements:
- `lag_00__CT3__flash_duration`: contribution `-0.005065`
- `lag_00__T3__flash_duration`: contribution `-0.004441`
- `lag_00__T5__flash_duration`: contribution `-0.001522`

### tick `87207`, seconds `5.00`, LSTM delta `+0.0383`

Top all feature movements:
- `lag_00__CT_place_LIBRARY`: contribution `+0.009802`
- `lag_00__T_place_LOWERMID`: contribution `+0.004434`
- `lag_01__CT_place_LIBRARY`: contribution `+0.002372`
- `lag_01__T_place_LOWERMID`: contribution `+0.002274`
- `lag_05__CT_place_LIBRARY`: contribution `+0.002028`

Top utility-only movements:
- `lag_10__CT3__molly`: contribution `+0.000663`
