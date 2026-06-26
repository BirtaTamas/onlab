# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-og-vs-nrg-bo3-GH6ZBFOA9sfdeCxgnhHN9f/og-vs-nrg-m2-nuke.csv`
- round_num: `6`

## Largest probability jumps

- tick `47576`, seconds `97.50`, LSTM `0.6278`, delta `+0.2479`
- tick `44952`, seconds `56.50`, LSTM `0.6653`, delta `+0.2094`
- tick `45496`, seconds `65.00`, LSTM `0.4411`, delta `-0.1938`
- tick `47768`, seconds `100.50`, LSTM `0.4739`, delta `-0.1934`
- tick `46936`, seconds `87.50`, LSTM `0.6302`, delta `+0.1908`
- tick `46904`, seconds `87.00`, LSTM `0.4393`, delta `+0.1796`
- tick `49336`, seconds `125.00`, LSTM `0.0893`, delta `-0.1594`
- tick `49304`, seconds `124.50`, LSTM `0.2487`, delta `-0.1439`
- tick `47192`, seconds `91.50`, LSTM `0.5270`, delta `-0.1345`
- tick `45624`, seconds `67.00`, LSTM `0.4343`, delta `+0.1066`

## Top 15 local ridge features

- `lag_00__T_place_DECON`: coefficient `-0.004137`, |coef| `0.004137`
- `lag_00__kill_diff_last_3s`: coefficient `0.003143`, |coef| `0.003143`
- `lag_00__damage_diff_last_5s`: coefficient `0.002981`, |coef| `0.002981`
- `lag_09__CT_defusing_count`: coefficient `-0.002774`, |coef| `0.002774`
- `lag_10__CT_defusing_count`: coefficient `-0.002683`, |coef| `0.002683`
- `lag_00__T_kills_last_3s`: coefficient `-0.002671`, |coef| `0.002671`
- `lag_00__T_place_OBSERVATION`: coefficient `-0.002334`, |coef| `0.002334`
- `lag_00__T_damage_last_5s`: coefficient `-0.002103`, |coef| `0.002103`
- `lag_06__CT_place_SQUEAKY`: coefficient `-0.002095`, |coef| `0.002095`
- `lag_00__CT_defusing_count`: coefficient `0.002059`, |coef| `0.002059`
- `lag_06__T_place_SECRET`: coefficient `0.002047`, |coef| `0.002047`
- `lag_00__T_shots_fired_sum`: coefficient `-0.002036`, |coef| `0.002036`
- `lag_08__T_place_VENTS`: coefficient `0.001976`, |coef| `0.001976`
- `lag_00__CT5__is_walking`: coefficient `0.001958`, |coef| `0.001958`
- `lag_04__CT_place_RAMP`: coefficient `0.001914`, |coef| `0.001914`

## Top 10 utility ridge features

- `lag_00__CT3__flash`: coefficient `0.001040` (raises CT win probability)
- `lag_03__T_active_infernos`: coefficient `0.000615` (raises CT win probability)
- `lag_00__CT3__utility_total`: coefficient `0.000579` (raises CT win probability)
- `lag_10__T2__flash_duration`: coefficient `-0.000551` (lowers CT win probability)
- `lag_06__CT4__flash_duration`: coefficient `0.000519` (raises CT win probability)
- `lag_14__T4__flash_duration`: coefficient `-0.000507` (lowers CT win probability)
- `lag_03__T_A_site_active_infernos`: coefficient `0.000503` (raises CT win probability)
- `lag_01__CT3__flash`: coefficient `0.000487` (raises CT win probability)
- `lag_03__T_B_site_active_infernos`: coefficient `0.000478` (raises CT win probability)
- `lag_11__T2__flash_duration`: coefficient `-0.000468` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_DECON`: coefficient `-0.004137` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003143` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002981` (raises CT win probability)
- `lag_09__CT_defusing_count`: coefficient `-0.002774` (lowers CT win probability)
- `lag_10__CT_defusing_count`: coefficient `-0.002683` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002671` (lowers CT win probability)
- `lag_00__T_place_OBSERVATION`: coefficient `-0.002334` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002103` (lowers CT win probability)
- `lag_06__CT_place_SQUEAKY`: coefficient `-0.002095` (lowers CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.002059` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `47576`, seconds `97.50`, LSTM delta `+0.2479`

Top all feature movements:
- `lag_00__CT_place_TROPHY`: contribution `+0.028017`
- `lag_08__T_place_VENTS`: contribution `+0.026654`
- `lag_01__T_place_DECON`: contribution `-0.026555`
- `lag_04__CT_place_VENDING`: contribution `+0.025456`
- `lag_06__T_place_VENTS`: contribution `+0.024559`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `44952`, seconds `56.50`, LSTM delta `+0.2094`

Top all feature movements:
- `lag_02__CT_place_VENDING`: contribution `+0.031302`
- `lag_06__CT_place_SQUEAKY`: contribution `+0.027855`
- `lag_06__T_place_SECRET`: contribution `+0.010772`
- `lag_00__kill_diff_last_3s`: contribution `+0.007566`
- `lag_00__T_place_TROPHY`: contribution `+0.006857`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `45496`, seconds `65.00`, LSTM delta `-0.1938`

Top all feature movements:
- `lag_06__CT_place_VENDING`: contribution `-0.025952`
- `lag_06__T_place_SECRET`: contribution `-0.021544`
- `lag_11__CT_place_TROPHY`: contribution `-0.020752`
- `lag_15__T_place_GARAGE`: contribution `-0.016612`
- `lag_15__CT_place_VENDING`: contribution `-0.014521`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `47768`, seconds `100.50`, LSTM delta `-0.1934`

Top all feature movements:
- `lag_06__CT_place_VENDING`: contribution `-0.025952`
- `lag_02__CT_place_TROPHY`: contribution `-0.021560`
- `lag_00__kill_diff_last_3s`: contribution `-0.015132`
- `lag_14__T_place_VENTS`: contribution `-0.011566`
- `lag_12__T_place_VENTS`: contribution `-0.010141`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `46936`, seconds `87.50`, LSTM delta `+0.1908`

Top all feature movements:
- `lag_00__T_place_OBSERVATION`: contribution `+0.039517`
- `lag_05__T_place_ADMIN`: contribution `+0.030397`
- `lag_15__T_place_ADMIN`: contribution `+0.029399`
- `lag_15__T_place_HELL`: contribution `+0.020979`
- `lag_14__T_place_OBSERVATION`: contribution `+0.013916`

Top utility-only movements:
- `lag_06__CT4__flash_duration`: contribution `+0.003039`
