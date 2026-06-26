# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-virtuspro-vs-b8-inferno-RJncpU8XKWGlyue1SsisvY/virtus-pro-vs-b8-inferno.csv`
- round_num: `15`

## Largest probability jumps

- tick `137193`, seconds `20.00`, LSTM `0.0688`, delta `-0.2872`
- tick `136649`, seconds `11.50`, LSTM `0.4611`, delta `+0.1393`
- tick `137129`, seconds `19.00`, LSTM `0.3187`, delta `-0.1295`
- tick `136393`, seconds `7.50`, LSTM `0.3321`, delta `-0.0688`
- tick `137097`, seconds `18.50`, LSTM `0.4482`, delta `-0.0638`
- tick `136969`, seconds `16.50`, LSTM `0.4997`, delta `-0.0399`
- tick `137161`, seconds `19.50`, LSTM `0.3561`, delta `+0.0373`
- tick `136009`, seconds `1.50`, LSTM `0.3363`, delta `+0.0268`
- tick `136201`, seconds `4.50`, LSTM `0.3712`, delta `+0.0221`
- tick `136297`, seconds `6.00`, LSTM `0.4329`, delta `+0.0215`

## Top 15 local ridge features

- `lag_00__CT_utility_damage_last_5s`: coefficient `0.002124`, |coef| `0.002124`
- `lag_07__CT_utility_damage_last_5s`: coefficient `0.001789`, |coef| `0.001789`
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.001742`, |coef| `0.001742`
- `lag_07__utility_damage_diff_last_5s`: coefficient `0.001465`, |coef| `0.001465`
- `lag_00__CT_place_QUAD`: coefficient `0.001302`, |coef| `0.001302`
- `lag_13__CT_place_QUAD`: coefficient `-0.001271`, |coef| `0.001271`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001153`, |coef| `0.001153`
- `lag_12__CT_place_QUAD`: coefficient `-0.001129`, |coef| `0.001129`
- `lag_01__CT3__shots_fired`: coefficient `-0.001091`, |coef| `0.001091`
- `lag_00__CT3__flash_duration`: coefficient `0.000970`, |coef| `0.000970`
- `lag_02__CT_shots_fired_sum`: coefficient `-0.000960`, |coef| `0.000960`
- `lag_02__CT3__shots_fired`: coefficient `-0.000951`, |coef| `0.000951`
- `lag_12__T_place_LOWERMID`: coefficient `0.000935`, |coef| `0.000935`
- `lag_08__CT3__flash_duration`: coefficient `-0.000924`, |coef| `0.000924`
- `lag_00__CT3__shots_fired`: coefficient `-0.000883`, |coef| `0.000883`

## Top 10 utility ridge features

- `lag_00__CT_utility_damage_last_5s`: coefficient `0.002124` (raises CT win probability)
- `lag_07__CT_utility_damage_last_5s`: coefficient `0.001789` (raises CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.001742` (raises CT win probability)
- `lag_07__utility_damage_diff_last_5s`: coefficient `0.001465` (raises CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `0.000970` (raises CT win probability)
- `lag_08__CT3__flash_duration`: coefficient `-0.000924` (lowers CT win probability)
- `lag_05__CT_utility_damage_last_5s`: coefficient `0.000880` (raises CT win probability)
- `lag_02__CT2__flash_duration`: coefficient `0.000802` (raises CT win probability)
- `lag_15__CT_utility_damage_last_5s`: coefficient `-0.000735` (lowers CT win probability)
- `lag_05__utility_damage_diff_last_5s`: coefficient `0.000715` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_QUAD`: coefficient `0.001302` (raises CT win probability)
- `lag_13__CT_place_QUAD`: coefficient `-0.001271` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001153` (raises CT win probability)
- `lag_12__CT_place_QUAD`: coefficient `-0.001129` (lowers CT win probability)
- `lag_01__CT3__shots_fired`: coefficient `-0.001091` (lowers CT win probability)
- `lag_02__CT_shots_fired_sum`: coefficient `-0.000960` (lowers CT win probability)
- `lag_02__CT3__shots_fired`: coefficient `-0.000951` (lowers CT win probability)
- `lag_12__T_place_LOWERMID`: coefficient `0.000935` (raises CT win probability)
- `lag_00__CT3__shots_fired`: coefficient `-0.000883` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.000877` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `137193`, seconds `20.00`, LSTM delta `-0.2872`

Top all feature movements:
- `lag_07__CT_utility_damage_last_5s`: contribution `-0.030522`
- `lag_07__utility_damage_diff_last_5s`: contribution `-0.020506`
- `lag_00__CT_place_QUAD`: contribution `-0.010262`
- `lag_13__CT_place_QUAD`: contribution `-0.010017`
- `lag_00__T_shots_fired_sum`: contribution `-0.009207`

Top utility-only movements:
- `lag_07__CT_utility_damage_last_5s`: contribution `-0.030522`
- `lag_07__utility_damage_diff_last_5s`: contribution `-0.020506`
- `lag_00__CT3__flash_duration`: contribution `-0.006698`
- `lag_08__CT3__flash_duration`: contribution `-0.006383`
- `lag_02__CT2__flash_duration`: contribution `-0.004299`

### tick `136649`, seconds `11.50`, LSTM delta `+0.1393`

Top all feature movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.036231`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.024377`
- `lag_12__T_place_LOWERMID`: contribution `+0.006220`
- `lag_08__CT_place_RUINS`: contribution `+0.005498`
- `lag_14__CT_place_LIBRARY`: contribution `+0.004718`

Top utility-only movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.036231`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.024377`

### tick `137129`, seconds `19.00`, LSTM delta `-0.1295`

Top all feature movements:
- `lag_05__CT_utility_damage_last_5s`: contribution `-0.015013`
- `lag_15__CT_utility_damage_last_5s`: contribution `-0.012545`
- `lag_05__utility_damage_diff_last_5s`: contribution `-0.010002`
- `lag_15__utility_damage_diff_last_5s`: contribution `-0.008623`
- `lag_01__CT_place_QUAD`: contribution `-0.006094`

Top utility-only movements:
- `lag_05__CT_utility_damage_last_5s`: contribution `-0.015013`
- `lag_15__CT_utility_damage_last_5s`: contribution `-0.012545`
- `lag_05__utility_damage_diff_last_5s`: contribution `-0.010002`
- `lag_15__utility_damage_diff_last_5s`: contribution `-0.008623`
- `lag_06__CT3__flash_duration`: contribution `-0.002944`

### tick `136393`, seconds `7.50`, LSTM delta `-0.0688`

Top all feature movements:
- `lag_00__CT_place_RUINS`: contribution `-0.005499`
- `lag_00__T_place_LOWERMID`: contribution `-0.004872`
- `lag_11__CT_place_LIBRARY`: contribution `-0.004773`
- `lag_02__CT_place_LIBRARY`: contribution `-0.003469`
- `lag_04__T_place_LOWERMID`: contribution `-0.003230`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `137097`, seconds `18.50`, LSTM delta `-0.0638`

Top all feature movements:
- `lag_00__CT_place_QUAD`: contribution `-0.010262`
- `lag_04__CT_utility_damage_last_5s`: contribution `-0.007576`
- `lag_00__CT_shots_fired_sum`: contribution `-0.007208`
- `lag_00__T_shots_fired_sum`: contribution `-0.005261`
- `lag_04__utility_damage_diff_last_5s`: contribution `-0.005083`

Top utility-only movements:
- `lag_04__CT_utility_damage_last_5s`: contribution `-0.007576`
- `lag_04__utility_damage_diff_last_5s`: contribution `-0.005083`
- `lag_14__CT_utility_damage_last_5s`: contribution `-0.003445`
- `lag_14__utility_damage_diff_last_5s`: contribution `-0.002400`
- `lag_02__CT2__flash_duration`: contribution `+0.001811`
