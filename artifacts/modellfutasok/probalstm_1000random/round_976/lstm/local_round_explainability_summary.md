# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-aurora-vs-astralis-bo3-EH-le-_LuObR5nGefXVoZY/aurora-vs-astralis-m3-overpass.csv`
- round_num: `22`

## Largest probability jumps

- tick `203506`, seconds `95.50`, LSTM `0.8263`, delta `+0.3150`
- tick `203410`, seconds `94.00`, LSTM `0.6080`, delta `-0.2532`
- tick `201778`, seconds `68.50`, LSTM `0.7840`, delta `+0.1563`
- tick `202578`, seconds `81.00`, LSTM `0.8754`, delta `+0.1348`
- tick `203570`, seconds `96.50`, LSTM `0.8498`, delta `+0.1019`
- tick `203122`, seconds `89.50`, LSTM `0.8216`, delta `-0.1016`
- tick `203538`, seconds `96.00`, LSTM `0.7479`, delta `-0.0785`
- tick `203442`, seconds `94.50`, LSTM `0.5510`, delta `-0.0570`
- tick `203666`, seconds `98.00`, LSTM `0.9511`, delta `+0.0466`
- tick `200370`, seconds `46.50`, LSTM `0.6519`, delta `-0.0441`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002317`, |coef| `0.002317`
- `lag_10__CT_place_BACKOFA`: coefficient `0.002021`, |coef| `0.002021`
- `lag_07__CT_place_STAIRS`: coefficient `-0.001941`, |coef| `0.001941`
- `lag_05__T4__is_scoped`: coefficient `-0.001909`, |coef| `0.001909`
- `lag_12__CT_place_STAIRS`: coefficient `-0.001791`, |coef| `0.001791`
- `lag_00__CT_kills_last_3s`: coefficient `0.001618`, |coef| `0.001618`
- `lag_12__CT_place_LOWERPARK`: coefficient `-0.001550`, |coef| `0.001550`
- `lag_00__CT_place_BACKOFA`: coefficient `0.001463`, |coef| `0.001463`
- `lag_04__CT_place_LOWERPARK`: coefficient `0.001455`, |coef| `0.001455`
- `lag_08__CT_place_BACKOFA`: coefficient `0.001426`, |coef| `0.001426`
- `lag_00__T_place_LOWERPARK`: coefficient `-0.001404`, |coef| `0.001404`
- `lag_00__damage_diff_last_5s`: coefficient `0.001404`, |coef| `0.001404`
- `lag_08__CT_place_STAIRS`: coefficient `-0.001394`, |coef| `0.001394`
- `lag_14__CT3__flash_duration`: coefficient `0.001341`, |coef| `0.001341`
- `lag_00__CT_flashed_players`: coefficient `0.001324`, |coef| `0.001324`

## Top 10 utility ridge features

- `lag_14__CT3__flash_duration`: coefficient `0.001341` (raises CT win probability)
- `lag_04__CT2__flash_duration`: coefficient `0.001134` (raises CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `0.000957` (raises CT win probability)
- `lag_15__CT2__flash_duration`: coefficient `-0.000920` (lowers CT win probability)
- `lag_03__CT3__flash_duration`: coefficient `-0.000829` (lowers CT win probability)
- `lag_01__CT2__flash_duration`: coefficient `0.000718` (raises CT win probability)
- `lag_05__CT3__flash_duration`: coefficient `-0.000712` (lowers CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `0.000670` (raises CT win probability)
- `lag_02__CT2__flash_duration`: coefficient `0.000648` (raises CT win probability)
- `lag_06__CT3__flash_duration`: coefficient `-0.000498` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002317` (raises CT win probability)
- `lag_10__CT_place_BACKOFA`: coefficient `0.002021` (raises CT win probability)
- `lag_07__CT_place_STAIRS`: coefficient `-0.001941` (lowers CT win probability)
- `lag_05__T4__is_scoped`: coefficient `-0.001909` (lowers CT win probability)
- `lag_12__CT_place_STAIRS`: coefficient `-0.001791` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001618` (raises CT win probability)
- `lag_12__CT_place_LOWERPARK`: coefficient `-0.001550` (lowers CT win probability)
- `lag_00__CT_place_BACKOFA`: coefficient `0.001463` (raises CT win probability)
- `lag_04__CT_place_LOWERPARK`: coefficient `0.001455` (raises CT win probability)
- `lag_08__CT_place_BACKOFA`: coefficient `0.001426` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `203506`, seconds `95.50`, LSTM delta `+0.3150`

Top all feature movements:
- `lag_12__CT_place_STAIRS`: contribution `+0.027879`
- `lag_07__CT_place_STAIRS`: contribution `+0.015107`
- `lag_08__CT_place_BACKOFA`: contribution `+0.013767`
- `lag_08__CT_place_STAIRS`: contribution `+0.010851`
- `lag_04__CT_place_BACKOFA`: contribution `+0.010657`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `203410`, seconds `94.00`, LSTM delta `-0.2532`

Top all feature movements:
- `lag_07__CT_place_STAIRS`: contribution `-0.030214`
- `lag_10__CT_place_BACKOFA`: contribution `-0.019515`
- `lag_11__CT_place_BACKOFA`: contribution `-0.012511`
- `lag_01__CT_place_BACKOFA`: contribution `-0.012347`
- `lag_04__CT_place_BACKOFA`: contribution `-0.010657`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `201778`, seconds `68.50`, LSTM delta `+0.1563`

Top all feature movements:
- `lag_08__CT_place_BACKOFA`: contribution `+0.013767`
- `lag_08__CT_place_STAIRS`: contribution `+0.010851`
- `lag_04__CT2__flash_duration`: contribution `+0.008939`
- `lag_00__CT_flashed_players`: contribution `+0.005799`
- `lag_00__kill_diff_last_3s`: contribution `+0.005576`

Top utility-only movements:
- `lag_04__CT2__flash_duration`: contribution `+0.008939`

### tick `202578`, seconds `81.00`, LSTM delta `+0.1348`

Top all feature movements:
- `lag_14__CT3__flash_duration`: contribution `+0.008319`
- `lag_15__CT2__flash_duration`: contribution `+0.007253`
- `lag_04__CT_place_LOWERPARK`: contribution `+0.006501`
- `lag_00__T_place_LOWERPARK`: contribution `+0.005660`
- `lag_00__kill_diff_last_3s`: contribution `+0.005576`

Top utility-only movements:
- `lag_14__CT3__flash_duration`: contribution `+0.008319`
- `lag_15__CT2__flash_duration`: contribution `+0.007253`
- `lag_03__CT3__flash_duration`: contribution `+0.005140`

### tick `203570`, seconds `96.50`, LSTM delta `+0.1019`

Top all feature movements:
- `lag_12__CT_place_STAIRS`: contribution `-0.027879`
- `lag_10__CT_place_BACKOFA`: contribution `+0.019515`
- `lag_09__CT_place_BACKOFA`: contribution `+0.012533`
- `lag_05__T4__is_scoped`: contribution `+0.008869`
- `lag_15__CT_place_BACKOFA`: contribution `-0.007778`

Top utility-only movements:
- No utility movement among the top local contributors.
