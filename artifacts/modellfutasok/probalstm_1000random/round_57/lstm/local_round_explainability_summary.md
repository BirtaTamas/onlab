# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-heroic-vs-aurora-bo3-0XrprgXu_t-aBJHUPpJYb4/heroic-vs-aurora-m1-overpass.csv`
- round_num: `11`

## Largest probability jumps

- tick `89992`, seconds `43.00`, LSTM `0.8466`, delta `+0.1434`
- tick `88584`, seconds `21.00`, LSTM `0.7649`, delta `+0.0801`
- tick `90024`, seconds `43.50`, LSTM `0.8855`, delta `+0.0389`
- tick `88264`, seconds `16.00`, LSTM `0.6888`, delta `+0.0347`
- tick `90408`, seconds `49.50`, LSTM `0.9567`, delta `+0.0332`
- tick `88072`, seconds `13.00`, LSTM `0.6546`, delta `-0.0284`
- tick `89256`, seconds `31.50`, LSTM `0.7686`, delta `-0.0282`
- tick `89224`, seconds `31.00`, LSTM `0.7968`, delta `-0.0249`
- tick `88552`, seconds `20.50`, LSTM `0.6848`, delta `+0.0222`
- tick `88808`, seconds `24.50`, LSTM `0.7789`, delta `-0.0214`

## Top 15 local ridge features

- `lag_06__T2__is_scoped`: coefficient `0.001838`, |coef| `0.001838`
- `lag_07__T2__is_scoped`: coefficient `0.001531`, |coef| `0.001531`
- `lag_12__T2__flash_duration`: coefficient `-0.001255`, |coef| `0.001255`
- `lag_13__T4__flash_duration`: coefficient `-0.000999`, |coef| `0.000999`
- `lag_05__CT5__duck_amount`: coefficient `-0.000894`, |coef| `0.000894`
- `lag_07__CT_shots_fired_sum`: coefficient `-0.000822`, |coef| `0.000822`
- `lag_09__CT_place_CANAL`: coefficient `0.000820`, |coef| `0.000820`
- `lag_00__CT_place_RESTROOM`: coefficient `0.000766`, |coef| `0.000766`
- `lag_00__T_scoped_count`: coefficient `-0.000735`, |coef| `0.000735`
- `lag_05__T2__is_scoped`: coefficient `0.000729`, |coef| `0.000729`
- `lag_08__T2__is_scoped`: coefficient `0.000722`, |coef| `0.000722`
- `lag_11__T2__is_scoped`: coefficient `-0.000704`, |coef| `0.000704`
- `lag_00__T2__has_bomb`: coefficient `-0.000703`, |coef| `0.000703`
- `lag_00__CT_kills_last_3s`: coefficient `0.000682`, |coef| `0.000682`
- `lag_12__T_flash_duration_sum`: coefficient `-0.000668`, |coef| `0.000668`

## Top 10 utility ridge features

- `lag_12__T2__flash_duration`: coefficient `-0.001255` (lowers CT win probability)
- `lag_13__T4__flash_duration`: coefficient `-0.000999` (lowers CT win probability)
- `lag_12__T_flash_duration_sum`: coefficient `-0.000668` (lowers CT win probability)
- `lag_13__T_flash_duration_sum`: coefficient `-0.000629` (lowers CT win probability)
- `lag_11__T2__flash_duration`: coefficient `-0.000594` (lowers CT win probability)
- `lag_12__CT1__molly`: coefficient `-0.000593` (lowers CT win probability)
- `lag_00__T2__smoke`: coefficient `-0.000560` (lowers CT win probability)
- `lag_13__T2__flash_duration`: coefficient `-0.000539` (lowers CT win probability)
- `lag_11__CT3__smoke`: coefficient `-0.000524` (lowers CT win probability)
- `lag_14__T4__flash_duration`: coefficient `-0.000462` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_06__T2__is_scoped`: coefficient `0.001838` (raises CT win probability)
- `lag_07__T2__is_scoped`: coefficient `0.001531` (raises CT win probability)
- `lag_05__CT5__duck_amount`: coefficient `-0.000894` (lowers CT win probability)
- `lag_07__CT_shots_fired_sum`: coefficient `-0.000822` (lowers CT win probability)
- `lag_09__CT_place_CANAL`: coefficient `0.000820` (raises CT win probability)
- `lag_00__CT_place_RESTROOM`: coefficient `0.000766` (raises CT win probability)
- `lag_00__T_scoped_count`: coefficient `-0.000735` (lowers CT win probability)
- `lag_05__T2__is_scoped`: coefficient `0.000729` (raises CT win probability)
- `lag_08__T2__is_scoped`: coefficient `0.000722` (raises CT win probability)
- `lag_11__T2__is_scoped`: coefficient `-0.000704` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `89992`, seconds `43.00`, LSTM delta `+0.1434`

Top all feature movements:
- `lag_06__T2__is_scoped`: contribution `+0.016198`
- `lag_12__T2__flash_duration`: contribution `+0.007006`
- `lag_11__T2__is_scoped`: contribution `+0.006207`
- `lag_13__T4__flash_duration`: contribution `+0.005113`
- `lag_09__CT_place_CANAL`: contribution `+0.004986`

Top utility-only movements:
- `lag_12__T2__flash_duration`: contribution `+0.007006`
- `lag_13__T4__flash_duration`: contribution `+0.005113`

### tick `88584`, seconds `21.00`, LSTM delta `+0.0801`

Top all feature movements:
- `lag_07__T2__is_scoped`: contribution `+0.013492`
- `lag_15__T_place_PIPE`: contribution `+0.007377`
- `lag_10__CT_place_RESTROOM`: contribution `+0.006725`
- `lag_00__T_place_LOWERPARK`: contribution `+0.002624`
- `lag_01__T2__is_scoped`: contribution `+0.002542`

Top utility-only movements:
- `lag_12__CT3__flash_duration`: contribution `+0.001524`

### tick `90024`, seconds `43.50`, LSTM delta `+0.0389`

Top all feature movements:
- `lag_07__T2__is_scoped`: contribution `+0.013492`
- `lag_11__T2__is_scoped`: contribution `-0.006207`
- `lag_13__T2__flash_duration`: contribution `+0.003007`
- `lag_14__T4__flash_duration`: contribution `+0.002365`
- `lag_10__T2__is_scoped`: contribution `-0.002176`

Top utility-only movements:
- `lag_13__T2__flash_duration`: contribution `+0.003007`
- `lag_14__T4__flash_duration`: contribution `+0.002365`
- `lag_13__T_flash_duration_sum`: contribution `+0.001438`

### tick `88264`, seconds `16.00`, LSTM delta `+0.0347`

Top all feature movements:
- `lag_00__CT_place_RESTROOM`: contribution `+0.010921`
- `lag_11__CT_place_RESTROOM`: contribution `+0.006077`
- `lag_07__CT_place_CANAL`: contribution `-0.003749`
- `lag_10__T_place_PIPE`: contribution `+0.003508`
- `lag_05__T_place_PIPE`: contribution `+0.002733`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `90408`, seconds `49.50`, LSTM delta `+0.0332`

Top all feature movements:
- `lag_03__CT_place_PIPE`: contribution `+0.025950`
- `lag_00__T_place_LOWERPARK`: contribution `+0.002624`
- `lag_13__T_place_LOWERPARK`: contribution `-0.002230`
- `lag_00__CT_kills_last_3s`: contribution `+0.001969`
- `lag_03__CT_place_CANAL`: contribution `-0.001620`

Top utility-only movements:
- No utility movement among the top local contributors.
