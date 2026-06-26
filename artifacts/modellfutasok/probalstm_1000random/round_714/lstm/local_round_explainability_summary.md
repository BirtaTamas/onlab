# Local Round Explainability

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-furia-vs-g2-bo3-QMek4tXQesgbTlulfGKOmD/furia-vs-g2-m1-inferno.csv`
- round_num: `8`

## Largest probability jumps

- tick `70898`, seconds `82.00`, LSTM `0.7582`, delta `+0.4851`
- tick `67826`, seconds `34.00`, LSTM `0.1412`, delta `-0.2760`
- tick `70290`, seconds `72.50`, LSTM `0.3050`, delta `+0.2532`
- tick `67762`, seconds `33.00`, LSTM `0.3788`, delta `+0.2474`
- tick `67602`, seconds `30.50`, LSTM `0.3107`, delta `+0.1923`
- tick `67666`, seconds `31.50`, LSTM `0.1464`, delta `-0.1869`
- tick `67474`, seconds `28.50`, LSTM `0.1898`, delta `-0.1620`
- tick `70322`, seconds `73.00`, LSTM `0.3978`, delta `+0.0928`
- tick `67954`, seconds `36.00`, LSTM `0.0908`, delta `-0.0734`
- tick `70546`, seconds `76.50`, LSTM `0.3784`, delta `-0.0619`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.006870`, |coef| `0.006870`
- `lag_00__CT_kills_last_3s`: coefficient `0.006748`, |coef| `0.006748`
- `lag_08__T_place_BALCONY`: coefficient `0.004677`, |coef| `0.004677`
- `lag_11__T_place_BALCONY`: coefficient `0.004583`, |coef| `0.004583`
- `lag_15__T_place_BALCONY`: coefficient `0.004169`, |coef| `0.004169`
- `lag_00__damage_diff_last_5s`: coefficient `0.003713`, |coef| `0.003713`
- `lag_00__T_place_BALCONY`: coefficient `-0.003550`, |coef| `0.003550`
- `lag_12__T4__duck_amount`: coefficient `0.003424`, |coef| `0.003424`
- `lag_10__CT3__is_walking`: coefficient `-0.003155`, |coef| `0.003155`
- `lag_13__CT_kills_last_3s`: coefficient `-0.003035`, |coef| `0.003035`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002988`, |coef| `0.002988`
- `lag_01__CT_kills_last_3s`: coefficient `0.002871`, |coef| `0.002871`
- `lag_14__CT3__duck_amount`: coefficient `0.002795`, |coef| `0.002795`
- `lag_00__CT5__shots_fired`: coefficient `0.002775`, |coef| `0.002775`
- `lag_01__kill_diff_last_3s`: coefficient `0.002752`, |coef| `0.002752`

## Top 10 utility ridge features

- `lag_00__T1__smoke`: coefficient `-0.002161` (lowers CT win probability)
- `lag_12__CT5__smoke`: coefficient `-0.002149` (lowers CT win probability)
- `lag_00__T4__smoke`: coefficient `-0.002066` (lowers CT win probability)
- `lag_08__T5__flash_duration`: coefficient `0.001747` (raises CT win probability)
- `lag_00__smoke_inv_diff`: coefficient `0.001724` (raises CT win probability)
- `lag_02__T1__smoke`: coefficient `-0.001600` (lowers CT win probability)
- `lag_01__T1__smoke`: coefficient `-0.001533` (lowers CT win probability)
- `lag_00__T_smoke_inv`: coefficient `-0.001491` (lowers CT win probability)
- `lag_00__T1__utility_total`: coefficient `-0.001468` (lowers CT win probability)
- `lag_14__T5__flash_duration`: coefficient `0.001405` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.006870` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.006748` (raises CT win probability)
- `lag_08__T_place_BALCONY`: coefficient `0.004677` (raises CT win probability)
- `lag_11__T_place_BALCONY`: coefficient `0.004583` (raises CT win probability)
- `lag_15__T_place_BALCONY`: coefficient `0.004169` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003713` (raises CT win probability)
- `lag_00__T_place_BALCONY`: coefficient `-0.003550` (lowers CT win probability)
- `lag_12__T4__duck_amount`: coefficient `0.003424` (raises CT win probability)
- `lag_10__CT3__is_walking`: coefficient `-0.003155` (lowers CT win probability)
- `lag_13__CT_kills_last_3s`: coefficient `-0.003035` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `70898`, seconds `82.00`, LSTM delta `+0.4851`

Top all feature movements:
- `lag_08__T_place_BALCONY`: contribution `+0.064313`
- `lag_11__T_place_BALCONY`: contribution `+0.063021`
- `lag_04__T_place_BALCONY`: contribution `+0.025815`
- `lag_00__CT_kills_last_3s`: contribution `+0.019483`
- `lag_00__kill_diff_last_3s`: contribution `+0.016537`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `67826`, seconds `34.00`, LSTM delta `-0.2760`

Top all feature movements:
- `lag_11__T_shots_fired_sum`: contribution `-0.018440`
- `lag_00__kill_diff_last_3s`: contribution `-0.016537`
- `lag_11__T4__shots_fired`: contribution `-0.008333`
- `lag_01__CT_kills_last_3s`: contribution `-0.008290`
- `lag_01__kill_diff_last_3s`: contribution `-0.006625`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `70290`, seconds `72.50`, LSTM delta `+0.2532`

Top all feature movements:
- `lag_00__T_place_BALCONY`: contribution `+0.048820`
- `lag_00__CT_kills_last_3s`: contribution `+0.019483`
- `lag_00__kill_diff_last_3s`: contribution `+0.016537`
- `lag_00__CT_shots_fired_sum`: contribution `+0.012457`
- `lag_02__T_place_BALCONY`: contribution `+0.011127`

Top utility-only movements:
- `lag_00__T1__smoke`: contribution `+0.004665`

### tick `67762`, seconds `33.00`, LSTM delta `+0.2474`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.019483`
- `lag_00__kill_diff_last_3s`: contribution `+0.016537`
- `lag_00__CT_shots_fired_sum`: contribution `-0.008304`
- `lag_09__T_shots_fired_sum`: contribution `+0.007655`
- `lag_02__T4__duck_amount`: contribution `+0.007603`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `67602`, seconds `30.50`, LSTM delta `+0.1923`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.019483`
- `lag_00__kill_diff_last_3s`: contribution `+0.016537`
- `lag_13__T_place_SECONDMID`: contribution `+0.008599`
- `lag_12__T5__flash_duration`: contribution `+0.007825`
- `lag_11__T4__duck_amount`: contribution `+0.007699`

Top utility-only movements:
- `lag_12__T5__flash_duration`: contribution `+0.007825`
- `lag_11__CT_B_site_active_infernos`: contribution `+0.003947`
