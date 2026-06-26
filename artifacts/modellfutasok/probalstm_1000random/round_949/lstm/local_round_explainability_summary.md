# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-heroic-bo3-VpF2znQtwzecEgVsCr-4Wn/astralis-vs-heroic-m2-inferno.csv`
- round_num: `14`

## Largest probability jumps

- tick `110668`, seconds `98.00`, LSTM `0.6589`, delta `-0.2213`
- tick `109900`, seconds `86.00`, LSTM `0.7757`, delta `+0.0959`
- tick `109196`, seconds `75.00`, LSTM `0.8222`, delta `-0.0774`
- tick `109932`, seconds `86.50`, LSTM `0.8519`, delta `+0.0762`
- tick `107788`, seconds `53.00`, LSTM `0.9273`, delta `+0.0555`
- tick `110828`, seconds `100.50`, LSTM `0.7010`, delta `+0.0546`
- tick `109644`, seconds `82.00`, LSTM `0.6392`, delta `-0.0538`
- tick `107308`, seconds `45.50`, LSTM `0.8779`, delta `+0.0529`
- tick `109260`, seconds `76.00`, LSTM `0.7467`, delta `-0.0419`
- tick `108332`, seconds `61.50`, LSTM `0.9098`, delta `-0.0415`

## Top 15 local ridge features

- `lag_00__CT_place_SECONDMID`: coefficient `0.002694`, |coef| `0.002694`
- `lag_08__CT_place_SECONDMID`: coefficient `-0.001991`, |coef| `0.001991`
- `lag_08__CT_place_ARCH`: coefficient `-0.001890`, |coef| `0.001890`
- `lag_08__CT_place_LOWERMID`: coefficient `0.001805`, |coef| `0.001805`
- `lag_00__kill_diff_last_3s`: coefficient `0.001691`, |coef| `0.001691`
- `lag_00__damage_diff_last_5s`: coefficient `0.001580`, |coef| `0.001580`
- `lag_00__spread_diff`: coefficient `0.001567`, |coef| `0.001567`
- `lag_03__T_place_BALCONY`: coefficient `0.001514`, |coef| `0.001514`
- `lag_01__CT_shots_fired_sum`: coefficient `0.001493`, |coef| `0.001493`
- `lag_08__CT_place_TRAMP`: coefficient `0.001410`, |coef| `0.001410`
- `lag_06__T3__flash_duration`: coefficient `-0.001291`, |coef| `0.001291`
- `lag_00__T_kills_last_3s`: coefficient `-0.001287`, |coef| `0.001287`
- `lag_00__T_spread_xy`: coefficient `-0.001273`, |coef| `0.001273`
- `lag_11__T3__flash_duration`: coefficient `0.001180`, |coef| `0.001180`
- `lag_15__T_kills_last_3s`: coefficient `-0.001150`, |coef| `0.001150`

## Top 10 utility ridge features

- `lag_06__T3__flash_duration`: coefficient `-0.001291` (lowers CT win probability)
- `lag_11__T3__flash_duration`: coefficient `0.001180` (raises CT win probability)
- `lag_07__T3__flash_duration`: coefficient `-0.001029` (lowers CT win probability)
- `lag_05__T3__flash_duration`: coefficient `-0.000833` (lowers CT win probability)
- `lag_12__T3__flash_duration`: coefficient `0.000761` (raises CT win probability)
- `lag_14__T3__flash`: coefficient `-0.000647` (lowers CT win probability)
- `lag_10__T3__flash_duration`: coefficient `0.000597` (raises CT win probability)
- `lag_06__T_flash_duration_sum`: coefficient `-0.000530` (lowers CT win probability)
- `lag_11__T_flash_duration_sum`: coefficient `0.000496` (raises CT win probability)
- `lag_00__CT3__flash`: coefficient `0.000479` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_SECONDMID`: coefficient `0.002694` (raises CT win probability)
- `lag_08__CT_place_SECONDMID`: coefficient `-0.001991` (lowers CT win probability)
- `lag_08__CT_place_ARCH`: coefficient `-0.001890` (lowers CT win probability)
- `lag_08__CT_place_LOWERMID`: coefficient `0.001805` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001691` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001580` (raises CT win probability)
- `lag_00__spread_diff`: coefficient `0.001567` (raises CT win probability)
- `lag_03__T_place_BALCONY`: coefficient `0.001514` (raises CT win probability)
- `lag_01__CT_shots_fired_sum`: coefficient `0.001493` (raises CT win probability)
- `lag_08__CT_place_TRAMP`: coefficient `0.001410` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `110668`, seconds `98.00`, LSTM delta `-0.2213`

Top all feature movements:
- `lag_00__CT_place_SECONDMID`: contribution `-0.055241`
- `lag_08__CT_place_LOWERMID`: contribution `-0.049525`
- `lag_08__CT_place_SECONDMID`: contribution `-0.040822`
- `lag_00__spread_diff`: contribution `-0.005515`
- `lag_00__T_kills_last_3s`: contribution `-0.004076`

Top utility-only movements:
- `lag_00__CT3__flash`: contribution `-0.000885`

### tick `109900`, seconds `86.00`, LSTM delta `+0.0959`

Top all feature movements:
- `lag_08__CT_place_ARCH`: contribution `+0.007712`
- `lag_06__T3__flash_duration`: contribution `+0.003718`
- `lag_14__CT_place_ARCH`: contribution `+0.003708`
- `lag_11__T3__flash_duration`: contribution `+0.003398`
- `lag_10__T3__duck_amount`: contribution `+0.003246`

Top utility-only movements:
- `lag_06__T3__flash_duration`: contribution `+0.003718`
- `lag_11__T3__flash_duration`: contribution `+0.003398`

### tick `109196`, seconds `75.00`, LSTM delta `-0.0774`

Top all feature movements:
- `lag_11__T_place_GRAVEYARD`: contribution `-0.020290`
- `lag_02__CT_place_LOWERMID`: contribution `-0.009614`
- `lag_02__CT_place_TRAMP`: contribution `-0.004263`
- `lag_00__T_kills_last_3s`: contribution `-0.004076`
- `lag_00__kill_diff_last_3s`: contribution `-0.004069`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `109932`, seconds `86.50`, LSTM delta `+0.0762`

Top all feature movements:
- `lag_01__CT_shots_fired_sum`: contribution `+0.005188`
- `lag_00__T_spread_xy`: contribution `+0.005138`
- `lag_00__spread_diff`: contribution `+0.004479`
- `lag_09__CT_place_ARCH`: contribution `+0.004432`
- `lag_00__kill_diff_last_3s`: contribution `+0.004069`

Top utility-only movements:
- `lag_07__T3__flash_duration`: contribution `+0.002962`
- `lag_12__T3__flash_duration`: contribution `+0.002192`

### tick `107788`, seconds `53.00`, LSTM delta `+0.0555`

Top all feature movements:
- `lag_10__T_place_ARCH`: contribution `+0.013185`
- `lag_13__T_place_ARCH`: contribution `+0.004884`
- `lag_15__T_kills_last_3s`: contribution `+0.003644`
- `lag_01__CT_shots_fired_sum`: contribution `+0.003113`
- `lag_01__CT3__is_walking`: contribution `+0.002344`

Top utility-only movements:
- No utility movement among the top local contributors.
