# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-og-vs-tyloo-ancient-6bJQWEKo0L9rHQMGqH72Vs/og-vs-tyloo-ancient.csv`
- round_num: `9`

## Largest probability jumps

- tick `64349`, seconds `79.50`, LSTM `0.1558`, delta `-0.1785`
- tick `61021`, seconds `27.50`, LSTM `0.7067`, delta `+0.1480`
- tick `61149`, seconds `29.50`, LSTM `0.7737`, delta `+0.1252`
- tick `64125`, seconds `76.00`, LSTM `0.4119`, delta `-0.0605`
- tick `61565`, seconds `36.00`, LSTM `0.6388`, delta `-0.0595`
- tick `62205`, seconds `46.00`, LSTM `0.5503`, delta `-0.0493`
- tick `61085`, seconds `28.50`, LSTM `0.6791`, delta `-0.0489`
- tick `61309`, seconds `32.00`, LSTM `0.6900`, delta `-0.0488`
- tick `64669`, seconds `84.50`, LSTM `0.0095`, delta `-0.0428`
- tick `64381`, seconds `80.00`, LSTM `0.1187`, delta `-0.0371`

## Top 15 local ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.001941`, |coef| `0.001941`
- `lag_00__kill_diff_last_3s`: coefficient `0.001893`, |coef| `0.001893`
- `lag_09__CT4__flash`: coefficient `0.001839`, |coef| `0.001839`
- `lag_00__T_B_site_active_infernos`: coefficient `-0.001655`, |coef| `0.001655`
- `lag_00__CT5__alive`: coefficient `0.001501`, |coef| `0.001501`
- `lag_00__CT_place_HOUSE`: coefficient `0.001495`, |coef| `0.001495`
- `lag_09__CT4__alive`: coefficient `0.001458`, |coef| `0.001458`
- `lag_14__T2__duck_amount`: coefficient `-0.001402`, |coef| `0.001402`
- `lag_09__CT4__shots_fired`: coefficient `-0.001378`, |coef| `0.001378`
- `lag_09__CT_burning_players`: coefficient `0.001374`, |coef| `0.001374`
- `lag_12__T5__is_walking`: coefficient `-0.001353`, |coef| `0.001353`
- `lag_00__CT5__has_defuser`: coefficient `0.001331`, |coef| `0.001331`
- `lag_00__CT5__armor`: coefficient `0.001330`, |coef| `0.001330`
- `lag_01__CT5__is_walking`: coefficient `0.001307`, |coef| `0.001307`
- `lag_09__CT4__armor`: coefficient `0.001301`, |coef| `0.001301`

## Top 10 utility ridge features

- `lag_09__CT4__flash`: coefficient `0.001839` (raises CT win probability)
- `lag_00__T_B_site_active_infernos`: coefficient `-0.001655` (lowers CT win probability)
- `lag_02__T3__molly`: coefficient `0.001257` (raises CT win probability)
- `lag_00__T_active_infernos`: coefficient `-0.001219` (lowers CT win probability)
- `lag_03__T4__flash_duration`: coefficient `0.001167` (raises CT win probability)
- `lag_15__CT4__flash_duration`: coefficient `0.001120` (raises CT win probability)
- `lag_09__CT4__utility_total`: coefficient `0.000992` (raises CT win probability)
- `lag_06__CT4__flash_duration`: coefficient `-0.000982` (lowers CT win probability)
- `lag_02__CT4__flash`: coefficient `0.000886` (raises CT win probability)
- `lag_00__T4__flash_duration`: coefficient `-0.000805` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.001941` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001893` (raises CT win probability)
- `lag_00__CT5__alive`: coefficient `0.001501` (raises CT win probability)
- `lag_00__CT_place_HOUSE`: coefficient `0.001495` (raises CT win probability)
- `lag_09__CT4__alive`: coefficient `0.001458` (raises CT win probability)
- `lag_14__T2__duck_amount`: coefficient `-0.001402` (lowers CT win probability)
- `lag_09__CT4__shots_fired`: coefficient `-0.001378` (lowers CT win probability)
- `lag_09__CT_burning_players`: coefficient `0.001374` (raises CT win probability)
- `lag_12__T5__is_walking`: coefficient `-0.001353` (lowers CT win probability)
- `lag_00__CT5__has_defuser`: coefficient `0.001331` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `64349`, seconds `79.50`, LSTM delta `-0.1785`

Top all feature movements:
- `lag_09__CT4__flash`: contribution `-0.006376`
- `lag_00__T_kills_last_3s`: contribution `-0.006149`
- `lag_00__CT_place_HOUSE`: contribution `-0.005282`
- `lag_07__CT2__duck_amount`: contribution `-0.004903`
- `lag_00__T_shots_fired_sum`: contribution `-0.004743`

Top utility-only movements:
- `lag_09__CT4__flash`: contribution `-0.006376`
- `lag_00__T_B_site_active_infernos`: contribution `-0.004678`

### tick `61021`, seconds `27.50`, LSTM delta `+0.1480`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.008537`
- `lag_03__T4__flash_duration`: contribution `+0.007904`
- `lag_15__CT4__flash_duration`: contribution `+0.005837`
- `lag_00__T4__flash_duration`: contribution `+0.005448`
- `lag_06__CT4__flash_duration`: contribution `+0.005119`

Top utility-only movements:
- `lag_03__T4__flash_duration`: contribution `+0.007904`
- `lag_15__CT4__flash_duration`: contribution `+0.005837`
- `lag_00__T4__flash_duration`: contribution `+0.005448`
- `lag_06__CT4__flash_duration`: contribution `+0.005119`
- `lag_15__CT5__flash_duration`: contribution `+0.004943`

### tick `61149`, seconds `29.50`, LSTM delta `+0.1252`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.052173`
- `lag_02__T_shots_fired_sum`: contribution `+0.005683`
- `lag_03__CT4__shots_fired`: contribution `+0.004773`
- `lag_07__T4__flash_duration`: contribution `+0.003797`
- `lag_11__T_place_TSIDELOWER`: contribution `+0.003569`

Top utility-only movements:
- `lag_07__T4__flash_duration`: contribution `+0.003797`
- `lag_03__CT_B_site_active_infernos`: contribution `+0.002755`
- `lag_10__CT4__flash_duration`: contribution `+0.002128`
- `lag_07__CT5__flash_duration`: contribution `+0.001933`

### tick `64125`, seconds `76.00`, LSTM delta `-0.0605`

Top all feature movements:
- `lag_14__T2__duck_amount`: contribution `-0.003619`
- `lag_11__T_place_TSIDELOWER`: contribution `-0.003569`
- `lag_00__T2__duck_amount`: contribution `-0.003396`
- `lag_07__T2__duck_amount`: contribution `-0.003149`
- `lag_12__T5__is_walking`: contribution `-0.003137`

Top utility-only movements:
- `lag_02__CT4__flash`: contribution `-0.003071`

### tick `61565`, seconds `36.00`, LSTM delta `-0.0595`

Top all feature movements:
- `lag_13__T_shots_fired_sum`: contribution `-0.035365`
- `lag_00__T_shots_fired_sum`: contribution `-0.005692`
- `lag_13__T2__shots_fired`: contribution `-0.005102`
- `lag_13__T1__shots_fired`: contribution `-0.002616`
- `lag_14__T_shots_fired_sum`: contribution `+0.002299`

Top utility-only movements:
- `lag_08__CT5__flash_duration`: contribution `+0.001609`
