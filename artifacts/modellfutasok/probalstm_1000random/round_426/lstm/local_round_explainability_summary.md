# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-falcons-vs-3dmax-bo3-XHM3Ovc8L9TfLFTYQFrGdT/falcons-vs-3dmax-m3-dust2.csv`
- round_num: `4`

## Largest probability jumps

- tick `19071`, seconds `80.50`, LSTM `0.8813`, delta `+0.2755`
- tick `14751`, seconds `13.00`, LSTM `0.8063`, delta `+0.2508`
- tick `18751`, seconds `75.50`, LSTM `0.7385`, delta `+0.1633`
- tick `18783`, seconds `76.00`, LSTM `0.6311`, delta `-0.1074`
- tick `19103`, seconds `81.00`, LSTM `0.7788`, delta `-0.1025`
- tick `18495`, seconds `71.50`, LSTM `0.6419`, delta `-0.0954`
- tick `14719`, seconds `12.50`, LSTM `0.5555`, delta `+0.0903`
- tick `18719`, seconds `75.00`, LSTM `0.5752`, delta `-0.0754`
- tick `14783`, seconds `13.50`, LSTM `0.7314`, delta `-0.0749`
- tick `19007`, seconds `79.50`, LSTM `0.5866`, delta `-0.0664`

## Top 15 local ridge features

- `lag_00__T_place_ARAMP`: coefficient `-0.003282`, |coef| `0.003282`
- `lag_00__CT_duck_amount_mean`: coefficient `0.002198`, |coef| `0.002198`
- `lag_15__T_he_last_5s`: coefficient `0.002128`, |coef| `0.002128`
- `lag_01__T_place_ARAMP`: coefficient `0.001704`, |coef| `0.001704`
- `lag_00__kill_diff_last_3s`: coefficient `0.001692`, |coef| `0.001692`
- `lag_00__CT3__duck_amount`: coefficient `0.001587`, |coef| `0.001587`
- `lag_10__CT_shots_fired_sum`: coefficient `-0.001523`, |coef| `0.001523`
- `lag_05__T_he_last_5s`: coefficient `-0.001504`, |coef| `0.001504`
- `lag_00__CT_kills_last_3s`: coefficient `0.001485`, |coef| `0.001485`
- `lag_00__CT1__is_walking`: coefficient `-0.001472`, |coef| `0.001472`
- `lag_11__T_place_ARAMP`: coefficient `0.001408`, |coef| `0.001408`
- `lag_00__CT4__duck_amount`: coefficient `0.001371`, |coef| `0.001371`
- `lag_10__CT5__shots_fired`: coefficient `-0.001322`, |coef| `0.001322`
- `lag_12__CT_flashes_last_5s`: coefficient `-0.001271`, |coef| `0.001271`
- `lag_14__T_he_last_5s`: coefficient `0.001253`, |coef| `0.001253`

## Top 10 utility ridge features

- `lag_15__T_he_last_5s`: coefficient `0.002128` (raises CT win probability)
- `lag_05__T_he_last_5s`: coefficient `-0.001504` (lowers CT win probability)
- `lag_12__CT_flashes_last_5s`: coefficient `-0.001271` (lowers CT win probability)
- `lag_14__T_he_last_5s`: coefficient `0.001253` (raises CT win probability)
- `lag_13__CT5__flash_duration`: coefficient `0.001231` (raises CT win probability)
- `lag_10__CT3__flash_duration`: coefficient `0.001193` (raises CT win probability)
- `lag_09__CT5__flash_duration`: coefficient `-0.001070` (lowers CT win probability)
- `lag_04__T4__flash_duration`: coefficient `-0.001065` (lowers CT win probability)
- `lag_02__T1__flash_duration`: coefficient `-0.000998` (lowers CT win probability)
- `lag_13__CT1__flash_duration`: coefficient `0.000990` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_ARAMP`: coefficient `-0.003282` (lowers CT win probability)
- `lag_00__CT_duck_amount_mean`: coefficient `0.002198` (raises CT win probability)
- `lag_01__T_place_ARAMP`: coefficient `0.001704` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001692` (raises CT win probability)
- `lag_00__CT3__duck_amount`: coefficient `0.001587` (raises CT win probability)
- `lag_10__CT_shots_fired_sum`: coefficient `-0.001523` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001485` (raises CT win probability)
- `lag_00__CT1__is_walking`: coefficient `-0.001472` (lowers CT win probability)
- `lag_11__T_place_ARAMP`: coefficient `0.001408` (raises CT win probability)
- `lag_00__CT4__duck_amount`: coefficient `0.001371` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `19071`, seconds `80.50`, LSTM delta `+0.2755`

Top all feature movements:
- `lag_00__T_place_ARAMP`: contribution `+0.029698`
- `lag_10__CT_shots_fired_sum`: contribution `+0.015868`
- `lag_11__T_place_ARAMP`: contribution `+0.012743`
- `lag_10__CT5__shots_fired`: contribution `+0.010490`
- `lag_10__CT_place_LOWERTUNNEL`: contribution `+0.008200`

Top utility-only movements:
- `lag_09__CT5__flash_duration`: contribution `+0.006242`
- `lag_02__T1__flash_duration`: contribution `+0.005919`
- `lag_13__CT5__flash_duration`: contribution `+0.005357`
- `lag_13__CT1__flash_duration`: contribution `+0.005282`
- `lag_03__T3__flash_duration`: contribution `+0.004777`

### tick `14751`, seconds `13.00`, LSTM delta `+0.2508`

Top all feature movements:
- `lag_15__T_he_last_5s`: contribution `+0.027775`
- `lag_05__T_he_last_5s`: contribution `+0.019633`
- `lag_12__CT_flashes_last_5s`: contribution `+0.013971`
- `lag_05__T_place_TUNNELSTAIRS`: contribution `+0.008092`
- `lag_10__CT3__flash_duration`: contribution `+0.007629`

Top utility-only movements:
- `lag_15__T_he_last_5s`: contribution `+0.027775`
- `lag_05__T_he_last_5s`: contribution `+0.019633`
- `lag_12__CT_flashes_last_5s`: contribution `+0.013971`
- `lag_10__CT3__flash_duration`: contribution `+0.007629`
- `lag_13__CT5__flash_duration`: contribution `+0.006101`

### tick `18751`, seconds `75.50`, LSTM delta `+0.1633`

Top all feature movements:
- `lag_00__T_place_ARAMP`: contribution `+0.029698`
- `lag_01__T_place_ARAMP`: contribution `+0.015416`
- `lag_00__CT_place_LOWERTUNNEL`: contribution `+0.008598`
- `lag_08__T_place_ARAMP`: contribution `+0.008206`
- `lag_10__CT_place_SHORTSTAIRS`: contribution `+0.005861`

Top utility-only movements:
- `lag_12__CT5__flash_duration`: contribution `+0.005846`
- `lag_12__CT_flash_duration_sum`: contribution `+0.004234`
- `lag_00__T4__flash_duration`: contribution `+0.003629`
- `lag_03__CT1__flash_duration`: contribution `-0.003229`
- `lag_12__T4__flash_duration`: contribution `+0.002986`

### tick `18783`, seconds `76.00`, LSTM delta `-0.1074`

Top all feature movements:
- `lag_01__T_place_ARAMP`: contribution `-0.015416`
- `lag_11__CT_place_EXTENDEDA`: contribution `-0.010155`
- `lag_13__CT5__flash_duration`: contribution `+0.007923`
- `lag_01__CT_shots_fired_sum`: contribution `-0.005770`
- `lag_01__CT5__shots_fired`: contribution `-0.005237`

Top utility-only movements:
- `lag_13__CT5__flash_duration`: contribution `+0.007923`
- `lag_13__CT_flash_duration_sum`: contribution `+0.005037`
- `lag_13__T4__flash_duration`: contribution `-0.004735`
- `lag_04__CT5__flash_duration`: contribution `-0.003988`
- `lag_13__CT4__flash_duration`: contribution `-0.003232`

### tick `19103`, seconds `81.00`, LSTM delta `-0.1025`

Top all feature movements:
- `lag_01__T_place_ARAMP`: contribution `-0.015416`
- `lag_11__T_place_ARAMP`: contribution `-0.012743`
- `lag_11__CT_shots_fired_sum`: contribution `-0.010470`
- `lag_00__CT4__duck_amount`: contribution `-0.005037`
- `lag_00__CT_duck_amount_mean`: contribution `-0.004740`

Top utility-only movements:
- `lag_10__CT5__flash_duration`: contribution `-0.003267`
- `lag_10__CT_flash_duration_sum`: contribution `-0.002129`
