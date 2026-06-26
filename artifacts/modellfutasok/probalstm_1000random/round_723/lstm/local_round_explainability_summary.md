# Local Round Explainability

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-faze-vs-bcgame-bo3-daIGc_M_y7Qq42fF9AoQhi/faze-vs-bcgame-m2-anubis.csv`
- round_num: `10`

## Largest probability jumps

- tick `63151`, seconds `41.00`, LSTM `0.8816`, delta `+0.1547`
- tick `62735`, seconds `34.50`, LSTM `0.7324`, delta `+0.1333`
- tick `62479`, seconds `30.50`, LSTM `0.6234`, delta `-0.1277`
- tick `62063`, seconds `24.00`, LSTM `0.7216`, delta `+0.0387`
- tick `63055`, seconds `39.50`, LSTM `0.7277`, delta `-0.0387`
- tick `64751`, seconds `66.00`, LSTM `0.9667`, delta `+0.0368`
- tick `62639`, seconds `33.00`, LSTM `0.5843`, delta `-0.0326`
- tick `61455`, seconds `14.50`, LSTM `0.6381`, delta `+0.0297`
- tick `60879`, seconds `5.50`, LSTM `0.5344`, delta `+0.0294`
- tick `60783`, seconds `4.00`, LSTM `0.5332`, delta `+0.0276`

## Top 15 local ridge features

- `lag_09__CT_place_HEAVEN`: coefficient `-0.002075`, |coef| `0.002075`
- `lag_13__T_place_MAIN`: coefficient `-0.001579`, |coef| `0.001579`
- `lag_00__kill_diff_last_3s`: coefficient `0.001472`, |coef| `0.001472`
- `lag_05__CT_place_BRICKS`: coefficient `0.001351`, |coef| `0.001351`
- `lag_00__damage_diff_last_5s`: coefficient `0.001337`, |coef| `0.001337`
- `lag_01__CT_place_FOUNTAIN`: coefficient `-0.001271`, |coef| `0.001271`
- `lag_03__CT_place_BRICKS`: coefficient `-0.001246`, |coef| `0.001246`
- `lag_10__CT_place_OUTSIDELONG`: coefficient `0.001216`, |coef| `0.001216`
- `lag_00__CT_kills_last_3s`: coefficient `0.001032`, |coef| `0.001032`
- `lag_00__CT5__is_scoped`: coefficient `-0.000985`, |coef| `0.000985`
- `lag_13__CT2__flash_duration`: coefficient `0.000912`, |coef| `0.000912`
- `lag_05__CT5__is_scoped`: coefficient `0.000901`, |coef| `0.000901`
- `lag_00__CT_damage_last_5s`: coefficient `0.000897`, |coef| `0.000897`
- `lag_09__CT_place_WALKWAY`: coefficient `0.000874`, |coef| `0.000874`
- `lag_05__CT2__flash_duration`: coefficient `-0.000859`, |coef| `0.000859`

## Top 10 utility ridge features

- `lag_13__CT2__flash_duration`: coefficient `0.000912` (raises CT win probability)
- `lag_05__CT2__flash_duration`: coefficient `-0.000859` (lowers CT win probability)
- `lag_04__CT2__flash_duration`: coefficient `0.000658` (raises CT win probability)
- `lag_00__T4__smoke`: coefficient `-0.000562` (lowers CT win probability)
- `lag_00__T5__flash`: coefficient `-0.000553` (lowers CT win probability)
- `lag_13__T5__flash`: coefficient `-0.000542` (lowers CT win probability)
- `lag_02__CT2__flash_duration`: coefficient `0.000464` (raises CT win probability)
- `lag_00__T5__utility_total`: coefficient `-0.000456` (lowers CT win probability)
- `lag_00__T_smoke_inv`: coefficient `-0.000437` (lowers CT win probability)
- `lag_14__CT2__flash_duration`: coefficient `0.000434` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_09__CT_place_HEAVEN`: coefficient `-0.002075` (lowers CT win probability)
- `lag_13__T_place_MAIN`: coefficient `-0.001579` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001472` (raises CT win probability)
- `lag_05__CT_place_BRICKS`: coefficient `0.001351` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001337` (raises CT win probability)
- `lag_01__CT_place_FOUNTAIN`: coefficient `-0.001271` (lowers CT win probability)
- `lag_03__CT_place_BRICKS`: coefficient `-0.001246` (lowers CT win probability)
- `lag_10__CT_place_OUTSIDELONG`: coefficient `0.001216` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001032` (raises CT win probability)
- `lag_00__CT5__is_scoped`: coefficient `-0.000985` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `63151`, seconds `41.00`, LSTM delta `+0.1547`

Top all feature movements:
- `lag_01__CT_place_FOUNTAIN`: contribution `+0.013367`
- `lag_10__CT_place_OUTSIDELONG`: contribution `+0.012333`
- `lag_09__CT_place_HEAVEN`: contribution `+0.011202`
- `lag_13__T_place_MAIN`: contribution `+0.010211`
- `lag_00__CT_place_OUTSIDELONG`: contribution `+0.006913`

Top utility-only movements:
- `lag_05__CT2__flash_duration`: contribution `+0.005556`

### tick `62735`, seconds `34.50`, LSTM delta `+0.1333`

Top all feature movements:
- `lag_05__CT_place_BRICKS`: contribution `+0.025946`
- `lag_03__CT_place_BRICKS`: contribution `+0.023925`
- `lag_04__CT2__flash_duration`: contribution `+0.004256`
- `lag_00__kill_diff_last_3s`: contribution `+0.003543`
- `lag_00__CT5__is_scoped`: contribution `+0.003524`

Top utility-only movements:
- `lag_04__CT2__flash_duration`: contribution `+0.004256`
- `lag_00__T5__flash`: contribution `+0.001569`

### tick `62479`, seconds `30.50`, LSTM delta `-0.1277`

Top all feature movements:
- `lag_09__CT_place_HEAVEN`: contribution `-0.011202`
- `lag_13__T_place_MAIN`: contribution `-0.010211`
- `lag_01__T_place_MAIN`: contribution `-0.005437`
- `lag_13__CT2__flash_duration`: contribution `-0.005000`
- `lag_09__CT_place_WALKWAY`: contribution `-0.004292`

Top utility-only movements:
- `lag_13__CT2__flash_duration`: contribution `-0.005000`

### tick `62063`, seconds `24.00`, LSTM delta `+0.0387`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.003543`
- `lag_00__CT5__is_scoped`: contribution `+0.003524`
- `lag_00__CT_kills_last_3s`: contribution `+0.002979`
- `lag_07__T_place_MAIN`: contribution `+0.002273`
- `lag_11__T1__duck_amount`: contribution `+0.002087`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `63055`, seconds `39.50`, LSTM delta `-0.0387`

Top all feature movements:
- `lag_15__CT_place_BRICKS`: contribution `-0.009046`
- `lag_13__CT_place_BRICKS`: contribution `-0.007754`
- `lag_01__T_place_MAIN`: contribution `-0.005437`
- `lag_07__CT_place_OUTSIDELONG`: contribution `-0.003887`
- `lag_00__damage_diff_last_5s`: contribution `-0.003017`

Top utility-only movements:
- `lag_02__CT2__flash_duration`: contribution `-0.003000`
- `lag_14__CT2__flash_duration`: contribution `+0.002803`
