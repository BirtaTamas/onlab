# Local Round Explainability

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-rare-atom-vs-nomads-bo3-2A6RLk5ZJnfAwsBhy_Qbbv/rare-atom-vs-nomads-m1-mirage.csv`
- round_num: `13`

## Largest probability jumps

- tick `85776`, seconds `15.50`, LSTM `0.1209`, delta `-0.1789`
- tick `85744`, seconds `15.00`, LSTM `0.2998`, delta `-0.0735`
- tick `88720`, seconds `61.50`, LSTM `0.0469`, delta `+0.0373`
- tick `88784`, seconds `62.50`, LSTM `0.0160`, delta `-0.0363`
- tick `85360`, seconds `9.00`, LSTM `0.4230`, delta `-0.0339`
- tick `87248`, seconds `38.50`, LSTM `0.0068`, delta `-0.0324`
- tick `85328`, seconds `8.50`, LSTM `0.4568`, delta `-0.0297`
- tick `85808`, seconds `16.00`, LSTM `0.0918`, delta `-0.0291`
- tick `85648`, seconds `13.50`, LSTM `0.3803`, delta `-0.0270`
- tick `85488`, seconds `11.00`, LSTM `0.4026`, delta `+0.0263`

## Top 15 local ridge features

- `lag_14__T_place_HOUSE`: coefficient `0.001497`, |coef| `0.001497`
- `lag_06__CT_place_TRUCK`: coefficient `0.001395`, |coef| `0.001395`
- `lag_15__T_place_BACKALLEY`: coefficient `-0.001351`, |coef| `0.001351`
- `lag_14__T_place_BACKALLEY`: coefficient `-0.001272`, |coef| `0.001272`
- `lag_11__T_place_BACKALLEY`: coefficient `-0.001253`, |coef| `0.001253`
- `lag_14__CT_place_UNDERPASS`: coefficient `-0.001243`, |coef| `0.001243`
- `lag_08__CT_place_TRUCK`: coefficient `-0.001237`, |coef| `0.001237`
- `lag_15__T_place_HOUSE`: coefficient `0.001222`, |coef| `0.001222`
- `lag_12__T_place_BACKALLEY`: coefficient `-0.001130`, |coef| `0.001130`
- `lag_12__CT5__duck_amount`: coefficient `-0.001104`, |coef| `0.001104`
- `lag_11__T_place_HOUSE`: coefficient `0.001101`, |coef| `0.001101`
- `lag_00__CT2__alive`: coefficient `0.000986`, |coef| `0.000986`
- `lag_04__CT_place_PALACEINTERIOR`: coefficient `-0.000977`, |coef| `0.000977`
- `lag_00__T_kills_last_3s`: coefficient `-0.000973`, |coef| `0.000973`
- `lag_12__T_place_HOUSE`: coefficient `0.000958`, |coef| `0.000958`

## Top 10 utility ridge features

- `lag_00__T5__smoke`: coefficient `0.000533` (raises CT win probability)
- `lag_00__T5__flash`: coefficient `0.000497` (raises CT win probability)
- `lag_00__T5__utility_total`: coefficient `0.000477` (raises CT win probability)
- `lag_01__T5__smoke`: coefficient `0.000336` (raises CT win probability)
- `lag_01__T5__utility_total`: coefficient `0.000318` (raises CT win probability)
- `lag_01__T5__flash`: coefficient `0.000312` (raises CT win probability)
- `lag_03__T5__flash`: coefficient `0.000254` (raises CT win probability)
- `lag_02__T5__flash`: coefficient `0.000244` (raises CT win probability)
- `lag_02__T5__utility_total`: coefficient `0.000243` (raises CT win probability)
- `lag_04__T5__flash`: coefficient `0.000237` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_14__T_place_HOUSE`: coefficient `0.001497` (raises CT win probability)
- `lag_06__CT_place_TRUCK`: coefficient `0.001395` (raises CT win probability)
- `lag_15__T_place_BACKALLEY`: coefficient `-0.001351` (lowers CT win probability)
- `lag_14__T_place_BACKALLEY`: coefficient `-0.001272` (lowers CT win probability)
- `lag_11__T_place_BACKALLEY`: coefficient `-0.001253` (lowers CT win probability)
- `lag_14__CT_place_UNDERPASS`: coefficient `-0.001243` (lowers CT win probability)
- `lag_08__CT_place_TRUCK`: coefficient `-0.001237` (lowers CT win probability)
- `lag_15__T_place_HOUSE`: coefficient `0.001222` (raises CT win probability)
- `lag_12__T_place_BACKALLEY`: coefficient `-0.001130` (lowers CT win probability)
- `lag_12__CT5__duck_amount`: coefficient `-0.001104` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `85776`, seconds `15.50`, LSTM delta `-0.1789`

Top all feature movements:
- `lag_06__CT_place_TRUCK`: contribution `-0.009000`
- `lag_08__CT_place_TRUCK`: contribution `-0.007977`
- `lag_14__CT_place_UNDERPASS`: contribution `-0.007206`
- `lag_14__T_place_HOUSE`: contribution `-0.006582`
- `lag_15__T_place_HOUSE`: contribution `-0.005374`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `85744`, seconds `15.00`, LSTM delta `-0.0735`

Top all feature movements:
- `lag_14__T_place_HOUSE`: contribution `-0.006582`
- `lag_15__T_place_HOUSE`: contribution `-0.005374`
- `lag_11__T_place_HOUSE`: contribution `-0.004843`
- `lag_13__CT_place_UNDERPASS`: contribution `-0.004266`
- `lag_15__T_place_BACKALLEY`: contribution `-0.004086`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `88720`, seconds `61.50`, LSTM delta `+0.0373`

Top all feature movements:
- `lag_02__T_place_LADDER`: contribution `+0.011533`
- `lag_00__T_place_UNDERPASS`: contribution `+0.003576`
- `lag_00__kill_diff_last_3s`: contribution `+0.001982`
- `lag_05__CT_place_SNIPERSNEST`: contribution `+0.001657`
- `lag_05__T5__duck_amount`: contribution `-0.001531`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `88784`, seconds `62.50`, LSTM delta `-0.0363`

Top all feature movements:
- `lag_04__T_place_LADDER`: contribution `-0.010525`
- `lag_00__CT_place_JUNGLE`: contribution `-0.003279`
- `lag_00__T_kills_last_3s`: contribution `-0.003084`
- `lag_06__T5__duck_amount`: contribution `+0.002956`
- `lag_07__CT_place_SNIPERSNEST`: contribution `-0.002793`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `85360`, seconds `9.00`, LSTM delta `-0.0339`

Top all feature movements:
- `lag_01__T_place_HOUSE`: contribution `-0.003710`
- `lag_02__T_place_HOUSE`: contribution `-0.003540`
- `lag_03__T_place_HOUSE`: contribution `-0.003166`
- `lag_01__CT_place_UNDERPASS`: contribution `-0.002438`
- `lag_05__CT_place_SNIPERSNEST`: contribution `+0.001657`

Top utility-only movements:
- No utility movement among the top local contributors.
