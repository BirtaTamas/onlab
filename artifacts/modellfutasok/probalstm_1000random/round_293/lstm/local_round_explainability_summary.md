# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-vitality-vs-the-mongolz-bo3-vhm_UWcBfYfYcOeLh9JIDA/vitality-vs-the-mongolz-m1-mirage.csv`
- round_num: `7`

## Largest probability jumps

- tick `67032`, seconds `61.50`, LSTM `0.5422`, delta `+0.2961`
- tick `67000`, seconds `61.00`, LSTM `0.2461`, delta `-0.2794`
- tick `67384`, seconds `67.00`, LSTM `0.8624`, delta `+0.2471`
- tick `67800`, seconds `73.50`, LSTM `0.9274`, delta `+0.2223`
- tick `67992`, seconds `76.50`, LSTM `0.7208`, delta `-0.2160`
- tick `67768`, seconds `73.00`, LSTM `0.7051`, delta `-0.1877`
- tick `65080`, seconds `31.00`, LSTM `0.4732`, delta `+0.1672`
- tick `64152`, seconds `16.50`, LSTM `0.3104`, delta `-0.1559`
- tick `64856`, seconds `27.50`, LSTM `0.2671`, delta `-0.0639`
- tick `68408`, seconds `83.00`, LSTM `0.5727`, delta `-0.0573`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004092`, |coef| `0.004092`
- `lag_00__damage_diff_last_5s`: coefficient `0.003148`, |coef| `0.003148`
- `lag_00__CT_kills_last_3s`: coefficient `0.003066`, |coef| `0.003066`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002497`, |coef| `0.002497`
- `lag_08__T2__duck_amount`: coefficient `-0.002318`, |coef| `0.002318`
- `lag_08__T3__is_scoped`: coefficient `-0.002196`, |coef| `0.002196`
- `lag_01__T_shots_fired_sum`: coefficient `-0.002161`, |coef| `0.002161`
- `lag_00__CT_damage_last_5s`: coefficient `0.002027`, |coef| `0.002027`
- `lag_00__T_kills_last_3s`: coefficient `-0.002022`, |coef| `0.002022`
- `lag_13__CT_kills_last_3s`: coefficient `0.001639`, |coef| `0.001639`
- `lag_02__CT_place_JUNGLE`: coefficient `0.001620`, |coef| `0.001620`
- `lag_03__CT_place_JUNGLE`: coefficient `0.001592`, |coef| `0.001592`
- `lag_01__T3__is_scoped`: coefficient `-0.001550`, |coef| `0.001550`
- `lag_05__CT5__duck_amount`: coefficient `-0.001548`, |coef| `0.001548`
- `lag_02__T_place_PALACEINTERIOR`: coefficient `0.001515`, |coef| `0.001515`

## Top 10 utility ridge features

- `lag_11__T3__flash_duration`: coefficient `-0.001172` (lowers CT win probability)
- `lag_00__T_flash_duration_sum`: coefficient `-0.001171` (lowers CT win probability)
- `lag_10__T4__flash_duration`: coefficient `-0.001083` (lowers CT win probability)
- `lag_07__CT3__flash_duration`: coefficient `-0.001034` (lowers CT win probability)
- `lag_01__T_flash_duration_sum`: coefficient `-0.000968` (lowers CT win probability)
- `lag_00__T4__flash_duration`: coefficient `-0.000925` (lowers CT win probability)
- `lag_10__T3__flash_duration`: coefficient `-0.000891` (lowers CT win probability)
- `lag_08__CT3__flash_duration`: coefficient `-0.000883` (lowers CT win probability)
- `lag_00__T2__molly`: coefficient `-0.000825` (lowers CT win probability)
- `lag_09__T4__flash_duration`: coefficient `-0.000824` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004092` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003148` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003066` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002497` (raises CT win probability)
- `lag_08__T2__duck_amount`: coefficient `-0.002318` (lowers CT win probability)
- `lag_08__T3__is_scoped`: coefficient `-0.002196` (lowers CT win probability)
- `lag_01__T_shots_fired_sum`: coefficient `-0.002161` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002027` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002022` (lowers CT win probability)
- `lag_13__CT_kills_last_3s`: coefficient `0.001639` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `67032`, seconds `61.50`, LSTM delta `+0.2961`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.010407`
- `lag_01__T3__is_scoped`: contribution `+0.009941`
- `lag_00__kill_diff_last_3s`: contribution `+0.009850`
- `lag_08__T2__duck_amount`: contribution `+0.008863`
- `lag_00__CT_kills_last_3s`: contribution `+0.008852`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `67000`, seconds `61.00`, LSTM delta `-0.2794`

Top all feature movements:
- `lag_08__T3__is_scoped`: contribution `-0.014086`
- `lag_00__kill_diff_last_3s`: contribution `-0.009850`
- `lag_08__T2__duck_amount`: contribution `-0.008863`
- `lag_12__CT_place_JUNGLE`: contribution `-0.008728`
- `lag_00__T3__is_scoped`: contribution `-0.007986`

Top utility-only movements:
- `lag_15__CT3__flash_duration`: contribution `-0.005261`

### tick `67384`, seconds `67.00`, LSTM delta `+0.2471`

Top all feature movements:
- `lag_02__CT_place_JUNGLE`: contribution `+0.010396`
- `lag_00__kill_diff_last_3s`: contribution `+0.009850`
- `lag_12__T3__is_scoped`: contribution `+0.009591`
- `lag_00__CT_kills_last_3s`: contribution `+0.008852`
- `lag_00__damage_diff_last_5s`: contribution `+0.007101`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `67800`, seconds `73.50`, LSTM delta `+0.2223`

Top all feature movements:
- `lag_01__T_shots_fired_sum`: contribution `+0.016199`
- `lag_00__kill_diff_last_3s`: contribution `+0.009850`
- `lag_00__CT_kills_last_3s`: contribution `+0.008852`
- `lag_00__CT_shots_fired_sum`: contribution `+0.008672`
- `lag_15__CT_place_JUNGLE`: contribution `+0.008213`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `67992`, seconds `76.50`, LSTM delta `-0.2160`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.019700`
- `lag_10__CT_place_JUNGLE`: contribution `-0.009153`
- `lag_07__CT_place_STAIRS`: contribution `-0.008976`
- `lag_00__CT_kills_last_3s`: contribution `-0.008852`
- `lag_07__T_shots_fired_sum`: contribution `-0.008701`

Top utility-only movements:
- No utility movement among the top local contributors.
