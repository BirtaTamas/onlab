# Local Round Explainability

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-faze-vs-g2-bo3-ldI7_iFRuThMOXF8zIbBwX/faze-vs-g2-m1-inferno.csv`
- round_num: `1`

## Largest probability jumps

- tick `13848`, seconds `69.00`, LSTM `0.2112`, delta `-0.2962`
- tick `14712`, seconds `82.50`, LSTM `0.0926`, delta `-0.2932`
- tick `11800`, seconds `37.00`, LSTM `0.3115`, delta `+0.1831`
- tick `11032`, seconds `25.00`, LSTM `0.3443`, delta `-0.1714`
- tick `14584`, seconds `80.50`, LSTM `0.3198`, delta `+0.1527`
- tick `11992`, seconds `40.00`, LSTM `0.4945`, delta `+0.1347`
- tick `14488`, seconds `79.00`, LSTM `0.2468`, delta `-0.1002`
- tick `11064`, seconds `25.50`, LSTM `0.2759`, delta `-0.0684`
- tick `13880`, seconds `69.50`, LSTM `0.1433`, delta `-0.0680`
- tick `11160`, seconds `27.00`, LSTM `0.1304`, delta `-0.0678`

## Top 15 local ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.003944`, |coef| `0.003944`
- `lag_00__kill_diff_last_3s`: coefficient `0.003663`, |coef| `0.003663`
- `lag_00__CT_place_TOPOFMID`: coefficient `0.003601`, |coef| `0.003601`
- `lag_00__T_damage_last_5s`: coefficient `-0.003507`, |coef| `0.003507`
- `lag_10__CT_place_TOPOFMID`: coefficient `-0.003457`, |coef| `0.003457`
- `lag_09__T_place_QUAD`: coefficient `-0.003445`, |coef| `0.003445`
- `lag_00__damage_diff_last_5s`: coefficient `0.003266`, |coef| `0.003266`
- `lag_00__CT5__alive`: coefficient `0.002750`, |coef| `0.002750`
- `lag_00__CT5__hp`: coefficient `0.002715`, |coef| `0.002715`
- `lag_00__CT5__armor`: coefficient `0.002576`, |coef| `0.002576`
- `lag_00__CT1__is_walking`: coefficient `0.002336`, |coef| `0.002336`
- `lag_10__CT_place_ARCH`: coefficient `0.002332`, |coef| `0.002332`
- `lag_11__T3__duck_amount`: coefficient `-0.002312`, |coef| `0.002312`
- `lag_06__CT5__duck_amount`: coefficient `-0.002296`, |coef| `0.002296`
- `lag_05__T_place_QUAD`: coefficient `0.002223`, |coef| `0.002223`

## Top 10 utility ridge features

- `lag_00__CT_utility_damage_last_5s`: coefficient `0.002167` (raises CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.001777` (raises CT win probability)
- `lag_06__CT_utility_damage_last_5s`: coefficient `0.001380` (raises CT win probability)
- `lag_08__T5__flash_duration`: coefficient `-0.001367` (lowers CT win probability)
- `lag_09__T4__flash_duration`: coefficient `-0.001334` (lowers CT win probability)
- `lag_14__T5__flash_duration`: coefficient `-0.001259` (lowers CT win probability)
- `lag_06__utility_damage_diff_last_5s`: coefficient `0.001123` (raises CT win probability)
- `lag_15__T4__flash_duration`: coefficient `-0.001050` (lowers CT win probability)
- `lag_00__CT_flash_alpha_mean`: coefficient `-0.001029` (lowers CT win probability)
- `lag_11__T3__flash_duration`: coefficient `-0.001008` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.003944` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003663` (raises CT win probability)
- `lag_00__CT_place_TOPOFMID`: coefficient `0.003601` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.003507` (lowers CT win probability)
- `lag_10__CT_place_TOPOFMID`: coefficient `-0.003457` (lowers CT win probability)
- `lag_09__T_place_QUAD`: coefficient `-0.003445` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003266` (raises CT win probability)
- `lag_00__CT5__alive`: coefficient `0.002750` (raises CT win probability)
- `lag_00__CT5__hp`: coefficient `0.002715` (raises CT win probability)
- `lag_00__CT5__armor`: coefficient `0.002576` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `13848`, seconds `69.00`, LSTM delta `-0.2962`

Top all feature movements:
- `lag_00__CT_place_TOPOFMID`: contribution `-0.013066`
- `lag_10__CT_place_TOPOFMID`: contribution `-0.012543`
- `lag_00__T_kills_last_3s`: contribution `-0.012496`
- `lag_10__CT_place_ARCH`: contribution `-0.009517`
- `lag_00__kill_diff_last_3s`: contribution `-0.008816`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `14712`, seconds `82.50`, LSTM delta `-0.2932`

Top all feature movements:
- `lag_09__T_place_QUAD`: contribution `-0.165945`
- `lag_07__T_place_QUAD`: contribution `-0.050622`
- `lag_00__T_kills_last_3s`: contribution `-0.012496`
- `lag_00__kill_diff_last_3s`: contribution `-0.008816`
- `lag_00__T_damage_last_5s`: contribution `-0.008409`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `11800`, seconds `37.00`, LSTM delta `+0.1831`

Top all feature movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.032437`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.021824`
- `lag_00__damage_diff_last_5s`: contribution `+0.010021`
- `lag_08__T5__flash_duration`: contribution `+0.008900`
- `lag_09__T4__flash_duration`: contribution `+0.008267`

Top utility-only movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.032437`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.021824`
- `lag_08__T5__flash_duration`: contribution `+0.008900`
- `lag_09__T4__flash_duration`: contribution `+0.008267`
- `lag_11__T3__flash_duration`: contribution `+0.004680`

### tick `11032`, seconds `25.00`, LSTM delta `-0.1714`

Top all feature movements:
- `lag_10__T_place_UPSTAIRS`: contribution `-0.018820`
- `lag_00__T_kills_last_3s`: contribution `-0.012496`
- `lag_12__CT_place_QUAD`: contribution `-0.011611`
- `lag_00__kill_diff_last_3s`: contribution `-0.008816`
- `lag_15__T_place_UPSTAIRS`: contribution `-0.008784`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `14584`, seconds `80.50`, LSTM delta `+0.1527`

Top all feature movements:
- `lag_05__T_place_QUAD`: contribution `+0.107064`
- `lag_02__T_place_QUAD`: contribution `+0.028722`
- `lag_00__kill_diff_last_3s`: contribution `+0.008816`
- `lag_03__T_place_QUAD`: contribution `-0.004693`
- `lag_05__CT1__is_walking`: contribution `-0.004197`

Top utility-only movements:
- No utility movement among the top local contributors.
