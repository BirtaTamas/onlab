# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m2-dust2.csv`
- round_num: `25`

## Largest probability jumps

- tick `228324`, seconds `17.00`, LSTM `0.6253`, delta `+0.2015`
- tick `228900`, seconds `26.00`, LSTM `0.6317`, delta `+0.1571`
- tick `231876`, seconds `72.50`, LSTM `0.8305`, delta `+0.1315`
- tick `231908`, seconds `73.00`, LSTM `0.9294`, delta `+0.0989`
- tick `228868`, seconds `25.50`, LSTM `0.4746`, delta `-0.0737`
- tick `231140`, seconds `61.00`, LSTM `0.7154`, delta `-0.0415`
- tick `231492`, seconds `66.50`, LSTM `0.7176`, delta `+0.0357`
- tick `231940`, seconds `73.50`, LSTM `0.9650`, delta `+0.0356`
- tick `228516`, seconds `20.00`, LSTM `0.6316`, delta `-0.0355`
- tick `229188`, seconds `30.50`, LSTM `0.6446`, delta `+0.0341`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001912`, |coef| `0.001912`
- `lag_00__kill_diff_last_3s`: coefficient `0.001724`, |coef| `0.001724`
- `lag_00__damage_diff_last_5s`: coefficient `0.001606`, |coef| `0.001606`
- `lag_08__T2__is_scoped`: coefficient `-0.001600`, |coef| `0.001600`
- `lag_10__CT_place_LOWERTUNNEL`: coefficient `0.001533`, |coef| `0.001533`
- `lag_11__CT_place_ARAMP`: coefficient `0.001529`, |coef| `0.001529`
- `lag_05__T_place_EXTENDEDA`: coefficient `0.001492`, |coef| `0.001492`
- `lag_00__CT_damage_last_5s`: coefficient `0.001395`, |coef| `0.001395`
- `lag_07__CT_place_ARAMP`: coefficient `-0.001350`, |coef| `0.001350`
- `lag_12__T_place_SHORTSTAIRS`: coefficient `0.001340`, |coef| `0.001340`
- `lag_00__CT1__flash_duration`: coefficient `0.001321`, |coef| `0.001321`
- `lag_04__T_place_TUNNELSTAIRS`: coefficient `0.001317`, |coef| `0.001317`
- `lag_00__CT5__is_walking`: coefficient `-0.001288`, |coef| `0.001288`
- `lag_10__CT_place_LONGDOORS`: coefficient `-0.001284`, |coef| `0.001284`
- `lag_00__CT4__is_walking`: coefficient `-0.001231`, |coef| `0.001231`

## Top 10 utility ridge features

- `lag_00__CT1__flash_duration`: coefficient `0.001321` (raises CT win probability)
- `lag_00__CT5__flash_duration`: coefficient `0.001088` (raises CT win probability)
- `lag_06__T2__flash_duration`: coefficient `-0.000971` (lowers CT win probability)
- `lag_06__T_A_site_active_infernos`: coefficient `0.000957` (raises CT win probability)
- `lag_14__T_utility_damage_last_5s`: coefficient `0.000855` (raises CT win probability)
- `lag_04__T_utility_damage_last_5s`: coefficient `-0.000758` (lowers CT win probability)
- `lag_06__T_active_infernos`: coefficient `0.000714` (raises CT win probability)
- `lag_15__CT1__flash_duration`: coefficient `0.000703` (raises CT win probability)
- `lag_05__CT_B_site_active_infernos`: coefficient `-0.000698` (lowers CT win probability)
- `lag_07__T_A_site_active_infernos`: coefficient `0.000693` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001912` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001724` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001606` (raises CT win probability)
- `lag_08__T2__is_scoped`: coefficient `-0.001600` (lowers CT win probability)
- `lag_10__CT_place_LOWERTUNNEL`: coefficient `0.001533` (raises CT win probability)
- `lag_11__CT_place_ARAMP`: coefficient `0.001529` (raises CT win probability)
- `lag_05__T_place_EXTENDEDA`: coefficient `0.001492` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001395` (raises CT win probability)
- `lag_07__CT_place_ARAMP`: coefficient `-0.001350` (lowers CT win probability)
- `lag_12__T_place_SHORTSTAIRS`: coefficient `0.001340` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `228324`, seconds `17.00`, LSTM delta `+0.2015`

Top all feature movements:
- `lag_08__T2__is_scoped`: contribution `+0.014102`
- `lag_10__CT_place_LOWERTUNNEL`: contribution `+0.011267`
- `lag_04__T_place_TUNNELSTAIRS`: contribution `+0.009197`
- `lag_10__CT_place_LONGDOORS`: contribution `+0.005621`
- `lag_00__CT_kills_last_3s`: contribution `+0.005521`

Top utility-only movements:
- `lag_06__T2__flash_duration`: contribution `+0.005122`
- `lag_14__T_utility_damage_last_5s`: contribution `+0.003540`
- `lag_04__T_utility_damage_last_5s`: contribution `+0.003137`
- `lag_05__CT_B_site_active_infernos`: contribution `+0.002397`

### tick `228900`, seconds `26.00`, LSTM delta `+0.1571`

Top all feature movements:
- `lag_00__CT1__flash_duration`: contribution `+0.009245`
- `lag_00__CT5__flash_duration`: contribution `+0.006991`
- `lag_14__T_place_TUNNELSTAIRS`: contribution `+0.006805`
- `lag_00__CT_kills_last_3s`: contribution `+0.005521`
- `lag_11__CT_place_EXTENDEDA`: contribution `+0.004534`

Top utility-only movements:
- `lag_00__CT1__flash_duration`: contribution `+0.009245`
- `lag_00__CT5__flash_duration`: contribution `+0.006991`

### tick `231876`, seconds `72.50`, LSTM delta `+0.1315`

Top all feature movements:
- `lag_11__CT_place_ARAMP`: contribution `+0.009525`
- `lag_07__CT_place_ARAMP`: contribution `+0.008411`
- `lag_05__T_place_EXTENDEDA`: contribution `+0.007396`
- `lag_02__T_place_EXTENDEDA`: contribution `+0.005876`
- `lag_12__T_place_SHORTSTAIRS`: contribution `+0.005630`

Top utility-only movements:
- `lag_06__T_A_site_active_infernos`: contribution `+0.002848`

### tick `231908`, seconds `73.00`, LSTM delta `+0.0989`

Top all feature movements:
- `lag_08__CT_place_ARAMP`: contribution `+0.007390`
- `lag_03__T_place_EXTENDEDA`: contribution `+0.006047`
- `lag_06__T_place_EXTENDEDA`: contribution `+0.006039`
- `lag_12__CT_place_ARAMP`: contribution `+0.005802`
- `lag_00__CT_kills_last_3s`: contribution `+0.005521`

Top utility-only movements:
- `lag_07__T_A_site_active_infernos`: contribution `+0.002063`
- `lag_00__T5__flash`: contribution `+0.001925`

### tick `228868`, seconds `25.50`, LSTM delta `-0.0737`

Top all feature movements:
- `lag_10__CT_place_LONGDOORS`: contribution `-0.005621`
- `lag_00__kill_diff_last_3s`: contribution `-0.004150`
- `lag_00__CT5__is_walking`: contribution `+0.003087`
- `lag_00__CT_place_LONGDOORS`: contribution `-0.002976`
- `lag_00__T2__duck_amount`: contribution `-0.002794`

Top utility-only movements:
- No utility movement among the top local contributors.
