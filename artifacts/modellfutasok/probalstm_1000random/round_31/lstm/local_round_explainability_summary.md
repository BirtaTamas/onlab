# Local Round Explainability

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-vitality-vs-g2-bo5-ieXHvClzA7f_aJ_85fPFqK/vitality-vs-g2-m1-dust2.csv`
- round_num: `16`

## Largest probability jumps

- tick `144918`, seconds `79.00`, LSTM `0.4657`, delta `+0.2382`
- tick `143894`, seconds `63.00`, LSTM `0.4321`, delta `+0.2122`
- tick `144150`, seconds `67.00`, LSTM `0.2204`, delta `-0.1862`
- tick `145686`, seconds `91.00`, LSTM `0.5401`, delta `+0.1828`
- tick `145142`, seconds `82.50`, LSTM `0.4035`, delta `-0.1700`
- tick `142902`, seconds `47.50`, LSTM `0.5079`, delta `-0.1661`
- tick `143254`, seconds `53.00`, LSTM `0.3907`, delta `-0.1232`
- tick `144726`, seconds `76.00`, LSTM `0.1765`, delta `+0.1114`
- tick `142870`, seconds `47.00`, LSTM `0.6740`, delta `-0.0830`
- tick `143318`, seconds `54.00`, LSTM `0.2868`, delta `-0.0711`

## Top 15 local ridge features

- `lag_00__CT_defusing_count`: coefficient `0.004386`, |coef| `0.004386`
- `lag_03__CT_place_EXTENDEDA`: coefficient `0.003195`, |coef| `0.003195`
- `lag_00__CT_duck_amount_mean`: coefficient `0.002807`, |coef| `0.002807`
- `lag_00__CT_place_EXTENDEDA`: coefficient `0.002512`, |coef| `0.002512`
- `lag_00__kill_diff_last_3s`: coefficient `0.002510`, |coef| `0.002510`
- `lag_03__CT_place_SHORTSTAIRS`: coefficient `-0.002240`, |coef| `0.002240`
- `lag_02__T_duck_amount_mean`: coefficient `0.002149`, |coef| `0.002149`
- `lag_11__T_kills_last_3s`: coefficient `-0.002095`, |coef| `0.002095`
- `lag_14__CT_shots_fired_sum`: coefficient `0.002058`, |coef| `0.002058`
- `lag_10__CT_place_SHORTSTAIRS`: coefficient `0.002000`, |coef| `0.002000`
- `lag_06__CT_shots_fired_sum`: coefficient `-0.001991`, |coef| `0.001991`
- `lag_09__CT_place_OUTSIDELONG`: coefficient `0.001898`, |coef| `0.001898`
- `lag_01__CT_defusing_count`: coefficient `0.001885`, |coef| `0.001885`
- `lag_00__T_kills_last_3s`: coefficient `-0.001812`, |coef| `0.001812`
- `lag_04__CT_place_OUTSIDELONG`: coefficient `-0.001784`, |coef| `0.001784`

## Top 10 utility ridge features

- `lag_00__CT_smokes_last_5s`: coefficient `0.001270` (raises CT win probability)
- `lag_00__CT_mollies_last_5s`: coefficient `0.000581` (raises CT win probability)
- `lag_08__T3__flash_duration`: coefficient `0.000547` (raises CT win probability)
- `lag_03__CT4__smoke`: coefficient `-0.000525` (lowers CT win probability)
- `lag_08__CT3__flash_duration`: coefficient `0.000512` (raises CT win probability)
- `lag_15__CT4__smoke`: coefficient `-0.000508` (lowers CT win probability)
- `lag_07__CT1__flash_duration`: coefficient `0.000470` (raises CT win probability)
- `lag_02__CT4__smoke`: coefficient `-0.000467` (lowers CT win probability)
- `lag_13__T_utility_damage_last_5s`: coefficient `-0.000463` (lowers CT win probability)
- `lag_09__CT2__flash_duration`: coefficient `0.000431` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_defusing_count`: coefficient `0.004386` (raises CT win probability)
- `lag_03__CT_place_EXTENDEDA`: coefficient `0.003195` (raises CT win probability)
- `lag_00__CT_duck_amount_mean`: coefficient `0.002807` (raises CT win probability)
- `lag_00__CT_place_EXTENDEDA`: coefficient `0.002512` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002510` (raises CT win probability)
- `lag_03__CT_place_SHORTSTAIRS`: coefficient `-0.002240` (lowers CT win probability)
- `lag_02__T_duck_amount_mean`: coefficient `0.002149` (raises CT win probability)
- `lag_11__T_kills_last_3s`: coefficient `-0.002095` (lowers CT win probability)
- `lag_14__CT_shots_fired_sum`: coefficient `0.002058` (raises CT win probability)
- `lag_10__CT_place_SHORTSTAIRS`: coefficient `0.002000` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `144918`, seconds `79.00`, LSTM delta `+0.2382`

Top all feature movements:
- `lag_03__CT_place_EXTENDEDA`: contribution `+0.017934`
- `lag_03__CT_place_SHORTSTAIRS`: contribution `+0.012489`
- `lag_13__CT_place_SHORTSTAIRS`: contribution `+0.009667`
- `lag_08__T_place_CTSPAWN`: contribution `+0.008351`
- `lag_13__T_place_CTSPAWN`: contribution `+0.008168`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `143894`, seconds `63.00`, LSTM delta `+0.2122`

Top all feature movements:
- `lag_06__CT_shots_fired_sum`: contribution `+0.029054`
- `lag_00__T_place_UNDERA`: contribution `+0.023256`
- `lag_09__CT_place_OUTSIDELONG`: contribution `+0.019249`
- `lag_04__CT_place_OUTSIDELONG`: contribution `+0.018094`
- `lag_06__CT5__shots_fired`: contribution `+0.016431`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `144150`, seconds `67.00`, LSTM delta `-0.1862`

Top all feature movements:
- `lag_14__CT_shots_fired_sum`: contribution `-0.030022`
- `lag_14__CT5__shots_fired`: contribution `-0.014341`
- `lag_08__T_place_UNDERA`: contribution `-0.013282`
- `lag_12__CT_place_OUTSIDELONG`: contribution `-0.011046`
- `lag_06__T4__shots_fired`: contribution `-0.007913`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `145686`, seconds `91.00`, LSTM delta `+0.1828`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `+0.042520`
- `lag_03__CT_place_EXTENDEDA`: contribution `+0.017934`
- `lag_00__CT_duck_amount_mean`: contribution `+0.011242`
- `lag_11__T_kills_last_3s`: contribution `+0.006639`
- `lag_07__CT_duck_amount_mean`: contribution `+0.004738`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `145142`, seconds `82.50`, LSTM delta `-0.1700`

Top all feature movements:
- `lag_00__CT_place_EXTENDEDA`: contribution `-0.014102`
- `lag_02__T_duck_amount_mean`: contribution `-0.012501`
- `lag_10__CT_place_SHORTSTAIRS`: contribution `-0.011150`
- `lag_10__CT_place_EXTENDEDA`: contribution `-0.007361`
- `lag_07__T_place_MIDDOORS`: contribution `-0.007112`

Top utility-only movements:
- No utility movement among the top local contributors.
