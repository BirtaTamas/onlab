# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-3dmax-vs-betboom-anubis-9yOMu3EhAmKzkIxUzvijXH/3dmax-vs-betboom-anubis.csv`
- round_num: `6`

## Largest probability jumps

- tick `34975`, seconds `21.50`, LSTM `0.4748`, delta `+0.3499`
- tick `38975`, seconds `84.00`, LSTM `0.1423`, delta `-0.2500`
- tick `34687`, seconds `17.00`, LSTM `0.3429`, delta `+0.1417`
- tick `38943`, seconds `83.50`, LSTM `0.3923`, delta `-0.1338`
- tick `34815`, seconds `19.00`, LSTM `0.2132`, delta `-0.1318`
- tick `35583`, seconds `31.00`, LSTM `0.5141`, delta `-0.1242`
- tick `35455`, seconds `29.00`, LSTM `0.5055`, delta `-0.1004`
- tick `37759`, seconds `65.00`, LSTM `0.6679`, delta `+0.0898`
- tick `38879`, seconds `82.50`, LSTM `0.5797`, delta `-0.0880`
- tick `34655`, seconds `16.50`, LSTM `0.2012`, delta `+0.0760`

## Top 15 local ridge features

- `lag_00__CT4__duck_amount`: coefficient `0.006351`, |coef| `0.006351`
- `lag_00__CT_place_BRICKS`: coefficient `0.004022`, |coef| `0.004022`
- `lag_00__CT_duck_amount_mean`: coefficient `0.003255`, |coef| `0.003255`
- `lag_00__T_place_MIDDOORS`: coefficient `-0.002820`, |coef| `0.002820`
- `lag_07__CT_place_TSTAIRS`: coefficient `-0.002665`, |coef| `0.002665`
- `lag_00__CT_place_HEAVEN`: coefficient `0.002495`, |coef| `0.002495`
- `lag_03__T_place_BRIDGE`: coefficient `0.002273`, |coef| `0.002273`
- `lag_06__CT_place_TSTAIRS`: coefficient `-0.002245`, |coef| `0.002245`
- `lag_14__T4__is_scoped`: coefficient `0.002093`, |coef| `0.002093`
- `lag_03__T3__duck_amount`: coefficient `-0.002092`, |coef| `0.002092`
- `lag_00__T_place_MIDDLE`: coefficient `0.002040`, |coef| `0.002040`
- `lag_00__kill_diff_last_3s`: coefficient `0.001969`, |coef| `0.001969`
- `lag_11__T3__is_walking`: coefficient `0.001940`, |coef| `0.001940`
- `lag_04__CT_place_TSTAIRS`: coefficient `-0.001912`, |coef| `0.001912`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001909`, |coef| `0.001909`

## Top 10 utility ridge features

- `lag_07__T1__flash_duration`: coefficient `0.001811` (raises CT win probability)
- `lag_10__T1__flash_duration`: coefficient `0.001810` (raises CT win probability)
- `lag_00__CT5__flash_duration`: coefficient `0.001770` (raises CT win probability)
- `lag_01__T1__flash_duration`: coefficient `0.001339` (raises CT win probability)
- `lag_14__T2__flash_duration`: coefficient `0.001293` (raises CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `0.001256` (raises CT win probability)
- `lag_05__T2__flash_duration`: coefficient `0.001230` (raises CT win probability)
- `lag_06__T1__flash_duration`: coefficient `0.001064` (raises CT win probability)
- `lag_05__CT5__flash_duration`: coefficient `0.001063` (raises CT win probability)
- `lag_02__CT5__flash_duration`: coefficient `0.001052` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT4__duck_amount`: coefficient `0.006351` (raises CT win probability)
- `lag_00__CT_place_BRICKS`: coefficient `0.004022` (raises CT win probability)
- `lag_00__CT_duck_amount_mean`: coefficient `0.003255` (raises CT win probability)
- `lag_00__T_place_MIDDOORS`: coefficient `-0.002820` (lowers CT win probability)
- `lag_07__CT_place_TSTAIRS`: coefficient `-0.002665` (lowers CT win probability)
- `lag_00__CT_place_HEAVEN`: coefficient `0.002495` (raises CT win probability)
- `lag_03__T_place_BRIDGE`: coefficient `0.002273` (raises CT win probability)
- `lag_06__CT_place_TSTAIRS`: coefficient `-0.002245` (lowers CT win probability)
- `lag_14__T4__is_scoped`: coefficient `0.002093` (raises CT win probability)
- `lag_03__T3__duck_amount`: coefficient `-0.002092` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `34975`, seconds `21.50`, LSTM delta `+0.3499`

Top all feature movements:
- `lag_00__CT_place_BRICKS`: contribution `+0.077238`
- `lag_10__T1__flash_duration`: contribution `+0.013153`
- `lag_00__CT5__flash_duration`: contribution `+0.012331`
- `lag_07__CT_place_MAIN`: contribution `+0.010959`
- `lag_12__T_shots_fired_sum`: contribution `+0.009541`

Top utility-only movements:
- `lag_10__T1__flash_duration`: contribution `+0.013153`
- `lag_00__CT5__flash_duration`: contribution `+0.012331`
- `lag_00__CT4__flash_duration`: contribution `+0.007347`
- `lag_14__T2__flash_duration`: contribution `+0.006922`
- `lag_12__T1__flash_duration`: contribution `+0.006266`

### tick `38975`, seconds `84.00`, LSTM delta `-0.2500`

Top all feature movements:
- `lag_07__CT_place_TSTAIRS`: contribution `-0.070338`
- `lag_00__CT4__duck_amount`: contribution `-0.023323`
- `lag_00__CT_place_TSTAIRS`: contribution `-0.022939`
- `lag_00__CT_place_HEAVEN`: contribution `-0.013470`
- `lag_01__T_place_WALKWAY`: contribution `-0.010265`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `34687`, seconds `17.00`, LSTM delta `+0.1417`

Top all feature movements:
- `lag_07__T1__flash_duration`: contribution `+0.013094`
- `lag_01__T1__flash_duration`: contribution `+0.009725`
- `lag_14__T4__is_scoped`: contribution `+0.009723`
- `lag_13__CT_place_HEAVEN`: contribution `+0.008731`
- `lag_00__T_shots_fired_sum`: contribution `+0.007157`

Top utility-only movements:
- `lag_07__T1__flash_duration`: contribution `+0.013094`
- `lag_01__T1__flash_duration`: contribution `+0.009725`
- `lag_05__T2__flash_duration`: contribution `+0.006587`
- `lag_05__CT5__flash_duration`: contribution `+0.003277`
- `lag_07__T_flash_duration_sum`: contribution `+0.002818`

### tick `38943`, seconds `83.50`, LSTM delta `-0.1338`

Top all feature movements:
- `lag_06__CT_place_TSTAIRS`: contribution `-0.059256`
- `lag_00__T_place_WALKWAY`: contribution `-0.024095`
- `lag_02__T_place_WALKWAY`: contribution `-0.022186`
- `lag_11__CT2__duck_amount`: contribution `-0.006466`
- `lag_06__CT_place_CANAL`: contribution `-0.004212`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `34815`, seconds `19.00`, LSTM delta `-0.1318`

Top all feature movements:
- `lag_07__T1__flash_duration`: contribution `-0.012864`
- `lag_00__CT_place_CANAL`: contribution `-0.008398`
- `lag_11__CT_place_HEAVEN`: contribution `-0.007867`
- `lag_00__T_shots_fired_sum`: contribution `-0.005725`
- `lag_00__T_kills_last_3s`: contribution `-0.005224`

Top utility-only movements:
- `lag_07__T1__flash_duration`: contribution `-0.012864`
- `lag_05__T1__flash_duration`: contribution `-0.003857`
- `lag_05__T2__flash_duration`: contribution `-0.003835`
- `lag_07__T_flash_duration_sum`: contribution `-0.002769`
