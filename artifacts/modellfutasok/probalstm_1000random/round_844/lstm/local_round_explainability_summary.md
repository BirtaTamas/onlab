# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-gamerlegion-vs-complexity-bo3-A8nOd44IyEYHGVOxrkExMv/gamerlegion-vs-complexity-m1-inferno.csv`
- round_num: `10`

## Largest probability jumps

- tick `79445`, seconds `71.00`, LSTM `0.7546`, delta `+0.2021`
- tick `78005`, seconds `48.50`, LSTM `0.8249`, delta `+0.1983`
- tick `78037`, seconds `49.00`, LSTM `0.6890`, delta `-0.1359`
- tick `80437`, seconds `86.50`, LSTM `0.9525`, delta `+0.1257`
- tick `75861`, seconds `15.00`, LSTM `0.7020`, delta `+0.1097`
- tick `76757`, seconds `29.00`, LSTM `0.5897`, delta `-0.0783`
- tick `80469`, seconds `87.00`, LSTM `0.8876`, delta `-0.0649`
- tick `81173`, seconds `98.00`, LSTM `0.6915`, delta `-0.0630`
- tick `81109`, seconds `97.00`, LSTM `0.7497`, delta `-0.0568`
- tick `79093`, seconds `65.50`, LSTM `0.5689`, delta `-0.0542`

## Top 15 local ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.003271`, |coef| `0.003271`
- `lag_00__CT1__duck_amount`: coefficient `0.002693`, |coef| `0.002693`
- `lag_00__kill_diff_last_3s`: coefficient `0.002674`, |coef| `0.002674`
- `lag_00__T_duck_amount_mean`: coefficient `-0.002672`, |coef| `0.002672`
- `lag_00__CT_kills_last_3s`: coefficient `0.002552`, |coef| `0.002552`
- `lag_02__T_flashed_players`: coefficient `0.002393`, |coef| `0.002393`
- `lag_00__T2__duck_amount`: coefficient `-0.002300`, |coef| `0.002300`
- `lag_00__CT_duck_amount_mean`: coefficient `0.002154`, |coef| `0.002154`
- `lag_00__CT_place_QUAD`: coefficient `0.002028`, |coef| `0.002028`
- `lag_00__CT_place_BACKALLEY`: coefficient `0.001812`, |coef| `0.001812`
- `lag_04__CT_flashes_last_5s`: coefficient `0.001734`, |coef| `0.001734`
- `lag_13__T_shots_fired_sum`: coefficient `0.001703`, |coef| `0.001703`
- `lag_02__CT5__is_walking`: coefficient `-0.001685`, |coef| `0.001685`
- `lag_11__T2__shots_fired`: coefficient `-0.001671`, |coef| `0.001671`
- `lag_09__CT_place_BALCONY`: coefficient `-0.001525`, |coef| `0.001525`

## Top 10 utility ridge features

- `lag_04__CT_flashes_last_5s`: coefficient `0.001734` (raises CT win probability)
- `lag_00__T1__flash`: coefficient `-0.001149` (lowers CT win probability)
- `lag_15__CT_flashes_last_5s`: coefficient `0.001143` (raises CT win probability)
- `lag_00__CT_flashes_last_5s`: coefficient `0.001102` (raises CT win probability)
- `lag_08__CT_flashes_last_5s`: coefficient `0.001035` (raises CT win probability)
- `lag_10__CT_flashes_last_5s`: coefficient `0.000839` (raises CT win probability)
- `lag_14__CT_flashes_last_5s`: coefficient `0.000797` (raises CT win probability)
- `lag_05__T_flashes_last_5s`: coefficient `0.000777` (raises CT win probability)
- `lag_09__CT_flashes_last_5s`: coefficient `0.000766` (raises CT win probability)
- `lag_12__CT5__molly`: coefficient `0.000740` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.003271` (raises CT win probability)
- `lag_00__CT1__duck_amount`: coefficient `0.002693` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002674` (raises CT win probability)
- `lag_00__T_duck_amount_mean`: coefficient `-0.002672` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002552` (raises CT win probability)
- `lag_02__T_flashed_players`: coefficient `0.002393` (raises CT win probability)
- `lag_00__T2__duck_amount`: coefficient `-0.002300` (lowers CT win probability)
- `lag_00__CT_duck_amount_mean`: coefficient `0.002154` (raises CT win probability)
- `lag_00__CT_place_QUAD`: coefficient `0.002028` (raises CT win probability)
- `lag_00__CT_place_BACKALLEY`: coefficient `0.001812` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `79445`, seconds `71.00`, LSTM delta `+0.2021`

Top all feature movements:
- `lag_11__T2__shots_fired`: contribution `+0.010814`
- `lag_09__T_shots_fired_sum`: contribution `+0.010533`
- `lag_09__CT_place_BALCONY`: contribution `+0.009790`
- `lag_09__T4__shots_fired`: contribution `+0.007767`
- `lag_00__CT_kills_last_3s`: contribution `+0.007367`

Top utility-only movements:
- `lag_00__T1__flash`: contribution `+0.003196`

### tick `78005`, seconds `48.50`, LSTM delta `+0.1983`

Top all feature movements:
- `lag_00__CT_place_QUAD`: contribution `+0.015987`
- `lag_02__T_flashed_players`: contribution `+0.013853`
- `lag_00__CT_shots_fired_sum`: contribution `+0.011362`
- `lag_00__CT1__duck_amount`: contribution `+0.010275`
- `lag_00__T2__duck_amount`: contribution `+0.008794`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `78037`, seconds `49.00`, LSTM delta `-0.1359`

Top all feature movements:
- `lag_02__T_flashed_players`: contribution `-0.013853`
- `lag_00__CT_shots_fired_sum`: contribution `-0.013634`
- `lag_00__T_duck_amount_mean`: contribution `-0.010360`
- `lag_00__CT1__duck_amount`: contribution `-0.010275`
- `lag_00__T2__duck_amount`: contribution `-0.008794`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `80437`, seconds `86.50`, LSTM delta `+0.1257`

Top all feature movements:
- `lag_04__CT_flashes_last_5s`: contribution `+0.019070`
- `lag_00__CT_shots_fired_sum`: contribution `+0.011362`
- `lag_00__T_duck_amount_mean`: contribution `+0.010539`
- `lag_00__CT_kills_last_3s`: contribution `+0.007367`
- `lag_00__kill_diff_last_3s`: contribution `+0.006437`

Top utility-only movements:
- `lag_04__CT_flashes_last_5s`: contribution `+0.019070`

### tick `75861`, seconds `15.00`, LSTM delta `+0.1097`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.007367`
- `lag_00__kill_diff_last_3s`: contribution `+0.006437`
- `lag_13__CT1__duck_amount`: contribution `+0.004246`
- `lag_07__CT1__is_scoped`: contribution `+0.003981`
- `lag_14__bomb_events_last_5s`: contribution `+0.003912`

Top utility-only movements:
- No utility movement among the top local contributors.
