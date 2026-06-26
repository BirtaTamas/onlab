# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-lynn-vision-vs-3dmax-bo3-peFr0yEP4eKTMrYfeYqBZK/lynn-vision-vs-3dmax-m3-train.csv`
- round_num: `13`

## Largest probability jumps

- tick `110204`, seconds `86.00`, LSTM `0.8196`, delta `+0.2679`
- tick `107196`, seconds `39.00`, LSTM `0.6679`, delta `-0.1699`
- tick `107068`, seconds `37.00`, LSTM `0.6795`, delta `+0.1653`
- tick `110268`, seconds `87.00`, LSTM `0.9257`, delta `+0.0974`
- tick `107100`, seconds `37.50`, LSTM `0.7684`, delta `+0.0889`
- tick `109724`, seconds `78.50`, LSTM `0.5302`, delta `-0.0866`
- tick `107260`, seconds `40.00`, LSTM `0.6031`, delta `-0.0574`
- tick `107132`, seconds `38.00`, LSTM `0.8177`, delta `+0.0493`
- tick `110652`, seconds `93.00`, LSTM `0.9294`, delta `-0.0412`
- tick `110780`, seconds `95.00`, LSTM `0.9604`, delta `+0.0360`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.003229`, |coef| `0.003229`
- `lag_00__kill_diff_last_3s`: coefficient `0.003204`, |coef| `0.003204`
- `lag_00__T_place_LONGDOG`: coefficient `-0.002200`, |coef| `0.002200`
- `lag_01__T_place_LONGDOG`: coefficient `-0.002035`, |coef| `0.002035`
- `lag_15__T_bomb_zone_count`: coefficient `0.002016`, |coef| `0.002016`
- `lag_09__CT5__flash_duration`: coefficient `0.001937`, |coef| `0.001937`
- `lag_00__damage_diff_last_5s`: coefficient `0.001871`, |coef| `0.001871`
- `lag_07__T_bomb_zone_count`: coefficient `-0.001866`, |coef| `0.001866`
- `lag_00__CT_damage_last_5s`: coefficient `0.001794`, |coef| `0.001794`
- `lag_15__CT5__flash_duration`: coefficient `-0.001754`, |coef| `0.001754`
- `lag_04__T_place_LONGDOG`: coefficient `0.001735`, |coef| `0.001735`
- `lag_00__CT_defusing_count`: coefficient `0.001674`, |coef| `0.001674`
- `lag_14__T2__flash_duration`: coefficient `0.001638`, |coef| `0.001638`
- `lag_06__T_place_BACKOFB`: coefficient `0.001597`, |coef| `0.001597`
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.001520`, |coef| `0.001520`

## Top 10 utility ridge features

- `lag_09__CT5__flash_duration`: coefficient `0.001937` (raises CT win probability)
- `lag_15__CT5__flash_duration`: coefficient `-0.001754` (lowers CT win probability)
- `lag_14__T2__flash_duration`: coefficient `0.001638` (raises CT win probability)
- `lag_06__T2__flash_duration`: coefficient `-0.001520` (lowers CT win probability)
- `lag_04__CT5__flash_duration`: coefficient `-0.001094` (lowers CT win probability)
- `lag_00__CT5__flash_duration`: coefficient `0.001004` (raises CT win probability)
- `lag_05__CT5__flash_duration`: coefficient `-0.000977` (lowers CT win probability)
- `lag_11__CT5__flash_duration`: coefficient `0.000966` (raises CT win probability)
- `lag_07__CT5__flash_duration`: coefficient `-0.000929` (lowers CT win probability)
- `lag_03__CT5__flash_duration`: coefficient `-0.000894` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.003229` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003204` (raises CT win probability)
- `lag_00__T_place_LONGDOG`: coefficient `-0.002200` (lowers CT win probability)
- `lag_01__T_place_LONGDOG`: coefficient `-0.002035` (lowers CT win probability)
- `lag_15__T_bomb_zone_count`: coefficient `0.002016` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001871` (raises CT win probability)
- `lag_07__T_bomb_zone_count`: coefficient `-0.001866` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001794` (raises CT win probability)
- `lag_04__T_place_LONGDOG`: coefficient `0.001735` (raises CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.001674` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `110204`, seconds `86.00`, LSTM delta `+0.2679`

Top all feature movements:
- `lag_09__CT5__flash_duration`: contribution `+0.012361`
- `lag_15__T_bomb_zone_count`: contribution `+0.011737`
- `lag_07__T_bomb_zone_count`: contribution `+0.010862`
- `lag_00__CT_kills_last_3s`: contribution `+0.009322`
- `lag_15__CT5__flash_duration`: contribution `+0.009245`

Top utility-only movements:
- `lag_09__CT5__flash_duration`: contribution `+0.012361`
- `lag_15__CT5__flash_duration`: contribution `+0.009245`
- `lag_14__T2__flash_duration`: contribution `+0.007758`
- `lag_06__T2__flash_duration`: contribution `+0.007196`

### tick `107196`, seconds `39.00`, LSTM delta `-0.1699`

Top all feature movements:
- `lag_04__T_place_LONGDOG`: contribution `-0.016143`
- `lag_00__T_place_LONGDOG`: contribution `-0.010239`
- `lag_01__T_place_LONGDOG`: contribution `-0.009470`
- `lag_00__CT_place_LONGDOG`: contribution `-0.008918`
- `lag_00__kill_diff_last_3s`: contribution `-0.007711`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `107068`, seconds `37.00`, LSTM delta `+0.1653`

Top all feature movements:
- `lag_00__T_place_LONGDOG`: contribution `+0.020477`
- `lag_00__CT_kills_last_3s`: contribution `+0.018644`
- `lag_00__kill_diff_last_3s`: contribution `+0.015422`
- `lag_09__T_place_LONGDOG`: contribution `+0.005222`
- `lag_00__CT_damage_last_5s`: contribution `+0.004106`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `110268`, seconds `87.00`, LSTM delta `+0.0974`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.009322`
- `lag_00__kill_diff_last_3s`: contribution `+0.007711`
- `lag_11__CT5__flash_duration`: contribution `+0.006164`
- `lag_07__CT_place_CONNECTOR`: contribution `+0.005146`
- `lag_00__CT1__duck_amount`: contribution `-0.004196`

Top utility-only movements:
- `lag_11__CT5__flash_duration`: contribution `+0.006164`
- `lag_08__T2__flash_duration`: contribution `+0.001896`

### tick `107100`, seconds `37.50`, LSTM delta `+0.0889`

Top all feature movements:
- `lag_01__T_place_LONGDOG`: contribution `+0.018940`
- `lag_01__CT_kills_last_3s`: contribution `+0.008672`
- `lag_01__kill_diff_last_3s`: contribution `+0.006190`
- `lag_09__T_place_LONGDOG`: contribution `+0.005222`
- `lag_13__T3__is_walking`: contribution `+0.002853`

Top utility-only movements:
- No utility movement among the top local contributors.
