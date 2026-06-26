# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-vitality-vs-astralis-bo3-Zley6FZuKcttfrliAqsvWJ/astralis-vs-vitality-m1-inferno.csv`
- round_num: `15`

## Largest probability jumps

- tick `110190`, seconds `38.50`, LSTM `0.1981`, delta `-0.2342`
- tick `110126`, seconds `37.50`, LSTM `0.4387`, delta `+0.2153`
- tick `110574`, seconds `44.50`, LSTM `0.0583`, delta `-0.1228`
- tick `110478`, seconds `43.00`, LSTM `0.1222`, delta `-0.0861`
- tick `110542`, seconds `44.00`, LSTM `0.1811`, delta `+0.0672`
- tick `107758`, seconds `0.50`, LSTM `0.0818`, delta `-0.0658`
- tick `111054`, seconds `52.00`, LSTM `0.0753`, delta `+0.0621`
- tick `111086`, seconds `52.50`, LSTM `0.0196`, delta `-0.0557`
- tick `109806`, seconds `32.50`, LSTM `0.2193`, delta `+0.0459`
- tick `110222`, seconds `39.00`, LSTM `0.2394`, delta `+0.0413`

## Top 15 local ridge features

- `lag_10__CT_place_QUAD`: coefficient `0.003707`, |coef| `0.003707`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001806`, |coef| `0.001806`
- `lag_02__T5__duck_amount`: coefficient `0.001665`, |coef| `0.001665`
- `lag_00__kill_diff_last_3s`: coefficient `0.001607`, |coef| `0.001607`
- `lag_12__CT_place_QUAD`: coefficient `-0.001478`, |coef| `0.001478`
- `lag_08__CT_place_QUAD`: coefficient `-0.001452`, |coef| `0.001452`
- `lag_00__CT_place_BANANA`: coefficient `0.001314`, |coef| `0.001314`
- `lag_08__CT_place_TOPOFMID`: coefficient `0.001281`, |coef| `0.001281`
- `lag_04__T4__duck_amount`: coefficient `0.001267`, |coef| `0.001267`
- `lag_01__T3__is_walking`: coefficient `0.001247`, |coef| `0.001247`
- `lag_00__T_kills_last_3s`: coefficient `-0.001165`, |coef| `0.001165`
- `lag_02__T4__duck_amount`: coefficient `-0.001088`, |coef| `0.001088`
- `lag_10__T3__is_walking`: coefficient `-0.001086`, |coef| `0.001086`
- `lag_09__T_place_TRAMP`: coefficient `0.001063`, |coef| `0.001063`
- `lag_00__damage_diff_last_5s`: coefficient `0.001040`, |coef| `0.001040`

## Top 10 utility ridge features

- `lag_00__T2__utility_total`: coefficient `-0.000526` (lowers CT win probability)
- `lag_09__T5__smoke`: coefficient `0.000514` (raises CT win probability)
- `lag_02__T2__utility_total`: coefficient `0.000503` (raises CT win probability)
- `lag_07__T5__smoke`: coefficient `-0.000493` (lowers CT win probability)
- `lag_00__T2__molly`: coefficient `-0.000476` (lowers CT win probability)
- `lag_00__T2__smoke`: coefficient `-0.000469` (lowers CT win probability)
- `lag_02__T2__molly`: coefficient `0.000443` (raises CT win probability)
- `lag_02__T2__smoke`: coefficient `0.000439` (raises CT win probability)
- `lag_01__T5__utility_total`: coefficient `-0.000362` (lowers CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000355` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_10__CT_place_QUAD`: coefficient `0.003707` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001806` (lowers CT win probability)
- `lag_02__T5__duck_amount`: coefficient `0.001665` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001607` (raises CT win probability)
- `lag_12__CT_place_QUAD`: coefficient `-0.001478` (lowers CT win probability)
- `lag_08__CT_place_QUAD`: coefficient `-0.001452` (lowers CT win probability)
- `lag_00__CT_place_BANANA`: coefficient `0.001314` (raises CT win probability)
- `lag_08__CT_place_TOPOFMID`: coefficient `0.001281` (raises CT win probability)
- `lag_04__T4__duck_amount`: coefficient `0.001267` (raises CT win probability)
- `lag_01__T3__is_walking`: coefficient `0.001247` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `110190`, seconds `38.50`, LSTM delta `-0.2342`

Top all feature movements:
- `lag_10__CT_place_QUAD`: contribution `-0.029214`
- `lag_12__CT_place_QUAD`: contribution `-0.011646`
- `lag_00__T_shots_fired_sum`: contribution `-0.006771`
- `lag_02__T5__duck_amount`: contribution `-0.006321`
- `lag_04__T4__duck_amount`: contribution `-0.004683`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `110126`, seconds `37.50`, LSTM delta `+0.2153`

Top all feature movements:
- `lag_10__CT_place_QUAD`: contribution `+0.029214`
- `lag_08__CT_place_QUAD`: contribution `+0.011441`
- `lag_02__T5__duck_amount`: contribution `+0.006321`
- `lag_04__T4__duck_amount`: contribution `+0.004683`
- `lag_08__CT_place_TOPOFMID`: contribution `+0.004648`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `110574`, seconds `44.50`, LSTM delta `-0.1228`

Top all feature movements:
- `lag_01__T_shots_fired_sum`: contribution `-0.014536`
- `lag_00__T_shots_fired_sum`: contribution `-0.005417`
- `lag_01__T1__shots_fired`: contribution `-0.003998`
- `lag_00__CT_place_BANANA`: contribution `-0.003889`
- `lag_00__kill_diff_last_3s`: contribution `-0.003868`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `110478`, seconds `43.00`, LSTM delta `-0.0861`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.013542`
- `lag_08__CT_place_TOPOFMID`: contribution `-0.004648`
- `lag_11__CT1__is_scoped`: contribution `-0.003634`
- `lag_10__T3__is_walking`: contribution `+0.002521`
- `lag_11__CT3__duck_amount`: contribution `-0.002395`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `110542`, seconds `44.00`, LSTM delta `+0.0672`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.027084`
- `lag_01__T_shots_fired_sum`: contribution `+0.007268`
- `lag_10__T_shots_fired_sum`: contribution `+0.002711`
- `lag_10__T3__is_walking`: contribution `+0.002521`
- `lag_01__T1__shots_fired`: contribution `+0.002332`

Top utility-only movements:
- No utility movement among the top local contributors.
