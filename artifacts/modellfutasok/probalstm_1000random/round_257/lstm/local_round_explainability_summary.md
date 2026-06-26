# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-spirit-vs-the-huns-bo3-TWIJIxJZifB3vPv3OUvjVr/spirit-vs-the-huns-m2-dust2.csv`
- round_num: `2`

## Largest probability jumps

- tick `18639`, seconds `75.50`, LSTM `0.2735`, delta `-0.4248`
- tick `18575`, seconds `74.50`, LSTM `0.6388`, delta `+0.3212`
- tick `18415`, seconds `72.00`, LSTM `0.6026`, delta `-0.2078`
- tick `18447`, seconds `72.50`, LSTM `0.4660`, delta `-0.1366`
- tick `18479`, seconds `73.00`, LSTM `0.3498`, delta `-0.1161`
- tick `18383`, seconds `71.50`, LSTM `0.8104`, delta `-0.0670`
- tick `18895`, seconds `79.50`, LSTM `0.1332`, delta `-0.0651`
- tick `18607`, seconds `75.00`, LSTM `0.6983`, delta `+0.0595`
- tick `19311`, seconds `86.00`, LSTM `0.1378`, delta `-0.0515`
- tick `20655`, seconds `107.00`, LSTM `0.0296`, delta `-0.0457`

## Top 15 local ridge features

- `lag_13__CT_place_ARAMP`: coefficient `-0.004486`, |coef| `0.004486`
- `lag_05__CT_place_OUTSIDELONG`: coefficient `-0.004259`, |coef| `0.004259`
- `lag_03__CT_place_OUTSIDELONG`: coefficient `0.003676`, |coef| `0.003676`
- `lag_05__CT_place_ARAMP`: coefficient `0.003607`, |coef| `0.003607`
- `lag_00__kill_diff_last_3s`: coefficient `0.003371`, |coef| `0.003371`
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.003005`, |coef| `0.003005`
- `lag_00__T_macro_B`: coefficient `-0.003005`, |coef| `0.003005`
- `lag_00__T_kills_last_3s`: coefficient `-0.002972`, |coef| `0.002972`
- `lag_01__CT2__shots_fired`: coefficient `-0.002967`, |coef| `0.002967`
- `lag_02__CT2__shots_fired`: coefficient `-0.002897`, |coef| `0.002897`
- `lag_05__CT_shots_fired_sum`: coefficient `-0.002788`, |coef| `0.002788`
- `lag_08__T5__duck_amount`: coefficient `-0.002765`, |coef| `0.002765`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002682`, |coef| `0.002682`
- `lag_07__CT_shots_fired_sum`: coefficient `0.002619`, |coef| `0.002619`
- `lag_14__CT_place_ARAMP`: coefficient `-0.002507`, |coef| `0.002507`

## Top 10 utility ridge features

- `lag_00__CT3__smoke`: coefficient `0.001269` (raises CT win probability)
- `lag_00__CT2__smoke`: coefficient `0.001191` (raises CT win probability)
- `lag_00__CT3__utility_total`: coefficient `0.001183` (raises CT win probability)
- `lag_00__CT2__flash`: coefficient `0.001183` (raises CT win probability)
- `lag_00__CT2__utility_total`: coefficient `0.001134` (raises CT win probability)
- `lag_00__CT3__flash`: coefficient `0.001059` (raises CT win probability)
- `lag_07__CT2__smoke`: coefficient `0.000997` (raises CT win probability)
- `lag_01__CT2__smoke`: coefficient `0.000913` (raises CT win probability)
- `lag_07__CT2__utility_total`: coefficient `0.000888` (raises CT win probability)
- `lag_05__CT2__smoke`: coefficient `-0.000863` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_13__CT_place_ARAMP`: coefficient `-0.004486` (lowers CT win probability)
- `lag_05__CT_place_OUTSIDELONG`: coefficient `-0.004259` (lowers CT win probability)
- `lag_03__CT_place_OUTSIDELONG`: coefficient `0.003676` (raises CT win probability)
- `lag_05__CT_place_ARAMP`: coefficient `0.003607` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003371` (raises CT win probability)
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.003005` (lowers CT win probability)
- `lag_00__T_macro_B`: coefficient `-0.003005` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002972` (lowers CT win probability)
- `lag_01__CT2__shots_fired`: coefficient `-0.002967` (lowers CT win probability)
- `lag_02__CT2__shots_fired`: coefficient `-0.002897` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `18639`, seconds `75.50`, LSTM delta `-0.4248`

Top all feature movements:
- `lag_05__CT_place_OUTSIDELONG`: contribution `-0.043195`
- `lag_05__CT_place_ARAMP`: contribution `-0.022472`
- `lag_07__CT_shots_fired_sum`: contribution `-0.016375`
- `lag_00__CT_place_OUTSIDELONG`: contribution `-0.011219`
- `lag_00__T_kills_last_3s`: contribution `-0.009415`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `18575`, seconds `74.50`, LSTM delta `+0.3212`

Top all feature movements:
- `lag_03__CT_place_OUTSIDELONG`: contribution `+0.037290`
- `lag_13__CT_place_ARAMP`: contribution `+0.027942`
- `lag_05__CT_shots_fired_sum`: contribution `+0.017434`
- `lag_03__CT_place_ARAMP`: contribution `+0.011863`
- `lag_08__T5__duck_amount`: contribution `+0.008842`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `18415`, seconds `72.00`, LSTM delta `-0.2078`

Top all feature movements:
- `lag_13__CT_place_ARAMP`: contribution `-0.027942`
- `lag_00__CT_shots_fired_sum`: contribution `-0.016769`
- `lag_01__CT2__shots_fired`: contribution `-0.010324`
- `lag_00__T_kills_last_3s`: contribution `-0.009415`
- `lag_07__T5__duck_amount`: contribution `-0.008125`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `18447`, seconds `72.50`, LSTM delta `-0.1366`

Top all feature movements:
- `lag_14__CT_place_ARAMP`: contribution `-0.015617`
- `lag_08__T5__duck_amount`: contribution `-0.010501`
- `lag_02__CT_shots_fired_sum`: contribution `-0.010387`
- `lag_02__CT2__shots_fired`: contribution `-0.010080`
- `lag_09__CT_place_ARAMP`: contribution `-0.005598`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `18479`, seconds `73.00`, LSTM delta `-0.1161`

Top all feature movements:
- `lag_02__CT_shots_fired_sum`: contribution `+0.013354`
- `lag_00__CT_place_OUTSIDELONG`: contribution `+0.011219`
- `lag_10__CT_place_ARAMP`: contribution `-0.010989`
- `lag_00__CT_place_LONGDOORS`: contribution `-0.008226`
- `lag_03__CT2__shots_fired`: contribution `-0.007851`

Top utility-only movements:
- No utility movement among the top local contributors.
