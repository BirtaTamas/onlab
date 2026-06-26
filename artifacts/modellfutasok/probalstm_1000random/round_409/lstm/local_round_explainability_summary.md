# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-aurora-vs-astralis-bo3-EH-le-_LuObR5nGefXVoZY/aurora-vs-astralis-m3-overpass.csv`
- round_num: `2`

## Largest probability jumps

- tick `14510`, seconds `66.00`, LSTM `0.7749`, delta `+0.1969`
- tick `17038`, seconds `105.50`, LSTM `0.8902`, delta `+0.0985`
- tick `16494`, seconds `97.00`, LSTM `0.7454`, delta `-0.0735`
- tick `13486`, seconds `50.00`, LSTM `0.6121`, delta `+0.0557`
- tick `16750`, seconds `101.00`, LSTM `0.7023`, delta `-0.0505`
- tick `15182`, seconds `76.50`, LSTM `0.7314`, delta `-0.0468`
- tick `16910`, seconds `103.50`, LSTM `0.7189`, delta `+0.0429`
- tick `15470`, seconds `81.00`, LSTM `0.7190`, delta `-0.0427`
- tick `17006`, seconds `105.00`, LSTM `0.7917`, delta `+0.0427`
- tick `13518`, seconds `50.50`, LSTM `0.5702`, delta `-0.0419`

## Top 15 local ridge features

- `lag_14__CT_place_LOBBY`: coefficient `0.002780`, |coef| `0.002780`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002432`, |coef| `0.002432`
- `lag_15__T4__flash_duration`: coefficient `-0.002050`, |coef| `0.002050`
- `lag_04__T_place_RESTROOM`: coefficient `-0.001671`, |coef| `0.001671`
- `lag_13__CT_place_LOBBY`: coefficient `0.001298`, |coef| `0.001298`
- `lag_12__CT_place_LOBBY`: coefficient `0.001290`, |coef| `0.001290`
- `lag_00__CT2__duck_amount`: coefficient `0.001216`, |coef| `0.001216`
- `lag_15__T5__is_walking`: coefficient `-0.001201`, |coef| `0.001201`
- `lag_02__T_flashes_last_5s`: coefficient `-0.001190`, |coef| `0.001190`
- `lag_01__CT_shots_fired_sum`: coefficient `0.001088`, |coef| `0.001088`
- `lag_14__T4__flash_duration`: coefficient `-0.001032`, |coef| `0.001032`
- `lag_00__CT_kills_last_3s`: coefficient `0.001013`, |coef| `0.001013`
- `lag_12__T_flashes_last_5s`: coefficient `0.001006`, |coef| `0.001006`
- `lag_00__CT_place_BACKOFA`: coefficient `0.000987`, |coef| `0.000987`
- `lag_06__CT_place_CANAL`: coefficient `-0.000982`, |coef| `0.000982`

## Top 10 utility ridge features

- `lag_15__T4__flash_duration`: coefficient `-0.002050` (lowers CT win probability)
- `lag_02__T_flashes_last_5s`: coefficient `-0.001190` (lowers CT win probability)
- `lag_14__T4__flash_duration`: coefficient `-0.001032` (lowers CT win probability)
- `lag_12__T_flashes_last_5s`: coefficient `0.001006` (raises CT win probability)
- `lag_10__T_flashes_last_5s`: coefficient `0.000965` (raises CT win probability)
- `lag_14__T_flashes_last_5s`: coefficient `-0.000923` (lowers CT win probability)
- `lag_13__T4__flash_duration`: coefficient `-0.000849` (lowers CT win probability)
- `lag_15__T_flash_duration_sum`: coefficient `-0.000843` (lowers CT win probability)
- `lag_08__T_flashes_last_5s`: coefficient `0.000822` (raises CT win probability)
- `lag_00__T4__flash_duration`: coefficient `0.000748` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_14__CT_place_LOBBY`: coefficient `0.002780` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002432` (raises CT win probability)
- `lag_04__T_place_RESTROOM`: coefficient `-0.001671` (lowers CT win probability)
- `lag_13__CT_place_LOBBY`: coefficient `0.001298` (raises CT win probability)
- `lag_12__CT_place_LOBBY`: coefficient `0.001290` (raises CT win probability)
- `lag_00__CT2__duck_amount`: coefficient `0.001216` (raises CT win probability)
- `lag_15__T5__is_walking`: coefficient `-0.001201` (lowers CT win probability)
- `lag_01__CT_shots_fired_sum`: coefficient `0.001088` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001013` (raises CT win probability)
- `lag_00__CT_place_BACKOFA`: coefficient `0.000987` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `14510`, seconds `66.00`, LSTM delta `+0.1969`

Top all feature movements:
- `lag_04__T_place_RESTROOM`: contribution `+0.032229`
- `lag_14__CT_place_LOBBY`: contribution `+0.022757`
- `lag_15__T4__flash_duration`: contribution `+0.014628`
- `lag_00__CT_shots_fired_sum`: contribution `+0.008449`
- `lag_04__CT_place_BACKOFA`: contribution `+0.008199`

Top utility-only movements:
- `lag_15__T4__flash_duration`: contribution `+0.014628`
- `lag_15__T_flash_duration_sum`: contribution `+0.002446`

### tick `17038`, seconds `105.50`, LSTM delta `+0.0985`

Top all feature movements:
- `lag_04__T_place_RESTROOM`: contribution `+0.032229`
- `lag_01__T_place_RESTROOM`: contribution `+0.011985`
- `lag_11__CT_place_STORAGEROOM`: contribution `+0.009181`
- `lag_11__CT_place_LOBBY`: contribution `-0.004870`
- `lag_14__T_place_RESTROOM`: contribution `+0.004680`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `16494`, seconds `97.00`, LSTM delta `-0.0735`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.027037`
- `lag_00__CT2__shots_fired`: contribution `-0.006129`
- `lag_06__CT_place_CANAL`: contribution `-0.005969`
- `lag_00__CT2__duck_amount`: contribution `-0.004632`
- `lag_02__CT_place_CANAL`: contribution `-0.004521`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `13486`, seconds `50.00`, LSTM delta `+0.0557`

Top all feature movements:
- `lag_02__T_flashes_last_5s`: contribution `+0.010780`
- `lag_12__T_flashes_last_5s`: contribution `+0.009113`
- `lag_00__CT_shots_fired_sum`: contribution `+0.008449`
- `lag_02__T_place_LOWERPARK`: contribution `+0.003041`
- `lag_01__CT_shots_fired_sum`: contribution `+0.003024`

Top utility-only movements:
- `lag_02__T_flashes_last_5s`: contribution `+0.010780`
- `lag_12__T_flashes_last_5s`: contribution `+0.009113`

### tick `16750`, seconds `101.00`, LSTM delta `-0.0505`

Top all feature movements:
- `lag_02__CT_place_STORAGEROOM`: contribution `-0.016839`
- `lag_00__T_place_RESTROOM`: contribution `-0.015222`
- `lag_05__T_place_RESTROOM`: contribution `-0.002856`
- `lag_05__CT4__shots_fired`: contribution `-0.002660`
- `lag_04__CT_place_WATER`: contribution `-0.002553`

Top utility-only movements:
- No utility movement among the top local contributors.
