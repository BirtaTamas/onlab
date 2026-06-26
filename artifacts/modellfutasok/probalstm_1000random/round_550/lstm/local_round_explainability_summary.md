# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-spirit-vs-flyquest-bo3-fQI-qOiPd1cRkmhkz0Xs5h/spirit-vs-flyquest-m1-mirage.csv`
- round_num: `11`

## Largest probability jumps

- tick `84912`, seconds `84.50`, LSTM `0.7863`, delta `+0.2375`
- tick `84752`, seconds `82.00`, LSTM `0.3900`, delta `+0.2240`
- tick `85040`, seconds `86.50`, LSTM `0.8501`, delta `+0.1771`
- tick `84976`, seconds `85.50`, LSTM `0.7066`, delta `-0.1611`
- tick `80592`, seconds `17.00`, LSTM `0.3687`, delta `-0.1028`
- tick `85072`, seconds `87.00`, LSTM `0.9317`, delta `+0.0816`
- tick `84944`, seconds `85.00`, LSTM `0.8677`, delta `+0.0814`
- tick `84784`, seconds `82.50`, LSTM `0.4573`, delta `+0.0672`
- tick `84816`, seconds `83.00`, LSTM `0.5064`, delta `+0.0492`
- tick `80624`, seconds `17.50`, LSTM `0.3212`, delta `-0.0475`

## Top 15 local ridge features

- `lag_00__CT_place_STAIRS`: coefficient `0.003144`, |coef| `0.003144`
- `lag_02__T_place_JUNGLE`: coefficient `0.003053`, |coef| `0.003053`
- `lag_00__T_place_CONNECTOR`: coefficient `-0.002951`, |coef| `0.002951`
- `lag_00__CT_kills_last_3s`: coefficient `0.002889`, |coef| `0.002889`
- `lag_00__kill_diff_last_3s`: coefficient `0.002865`, |coef| `0.002865`
- `lag_00__CT_damage_last_5s`: coefficient `0.002537`, |coef| `0.002537`
- `lag_06__T_place_CONNECTOR`: coefficient `0.002499`, |coef| `0.002499`
- `lag_00__damage_diff_last_5s`: coefficient `0.002458`, |coef| `0.002458`
- `lag_02__CT_place_STAIRS`: coefficient `-0.002451`, |coef| `0.002451`
- `lag_10__T_place_CONNECTOR`: coefficient `0.002408`, |coef| `0.002408`
- `lag_15__T_place_CONNECTOR`: coefficient `0.002388`, |coef| `0.002388`
- `lag_13__CT5__duck_amount`: coefficient `-0.002376`, |coef| `0.002376`
- `lag_11__T_place_CONNECTOR`: coefficient `0.002326`, |coef| `0.002326`
- `lag_00__CT2__is_walking`: coefficient `-0.002249`, |coef| `0.002249`
- `lag_00__CT_place_SCAFFOLDING`: coefficient `-0.002202`, |coef| `0.002202`

## Top 10 utility ridge features

- `lag_06__CT_A_site_active_smokes`: coefficient `-0.001013` (lowers CT win probability)
- `lag_01__T2__flash`: coefficient `0.001010` (raises CT win probability)
- `lag_00__T5__utility_total`: coefficient `-0.000998` (lowers CT win probability)
- `lag_15__CT_A_site_active_smokes`: coefficient `-0.000966` (lowers CT win probability)
- `lag_05__T5__flash_duration`: coefficient `0.000951` (raises CT win probability)
- `lag_13__CT_B_site_active_smokes`: coefficient `-0.000860` (lowers CT win probability)
- `lag_00__T5__molly`: coefficient `-0.000854` (lowers CT win probability)
- `lag_00__T5__smoke`: coefficient `-0.000837` (lowers CT win probability)
- `lag_06__CT_active_smokes`: coefficient `-0.000743` (lowers CT win probability)
- `lag_13__CT_A_site_active_smokes`: coefficient `-0.000740` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_STAIRS`: coefficient `0.003144` (raises CT win probability)
- `lag_02__T_place_JUNGLE`: coefficient `0.003053` (raises CT win probability)
- `lag_00__T_place_CONNECTOR`: coefficient `-0.002951` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002889` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002865` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002537` (raises CT win probability)
- `lag_06__T_place_CONNECTOR`: coefficient `0.002499` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002458` (raises CT win probability)
- `lag_02__CT_place_STAIRS`: coefficient `-0.002451` (lowers CT win probability)
- `lag_10__T_place_CONNECTOR`: coefficient `0.002408` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `84912`, seconds `84.50`, LSTM delta `+0.2375`

Top all feature movements:
- `lag_02__T_place_JUNGLE`: contribution `+0.039548`
- `lag_00__CT_place_STAIRS`: contribution `+0.024466`
- `lag_15__T_place_CONNECTOR`: contribution `+0.011562`
- `lag_11__T_place_CONNECTOR`: contribution `+0.011265`
- `lag_00__CT_kills_last_3s`: contribution `+0.008340`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `84752`, seconds `82.00`, LSTM delta `+0.2240`

Top all feature movements:
- `lag_00__T_place_CONNECTOR`: contribution `+0.014289`
- `lag_06__T_place_CONNECTOR`: contribution `+0.012100`
- `lag_10__T_place_CONNECTOR`: contribution `+0.011662`
- `lag_13__CT5__duck_amount`: contribution `+0.008968`
- `lag_00__CT_kills_last_3s`: contribution `+0.008340`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `85040`, seconds `86.50`, LSTM delta `+0.1771`

Top all feature movements:
- `lag_06__T_place_JUNGLE`: contribution `+0.023320`
- `lag_02__CT_place_STAIRS`: contribution `+0.019078`
- `lag_15__T_place_CONNECTOR`: contribution `+0.011562`
- `lag_00__CT_kills_last_3s`: contribution `+0.008340`
- `lag_00__kill_diff_last_3s`: contribution `+0.006897`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `84976`, seconds `85.50`, LSTM delta `-0.1611`

Top all feature movements:
- `lag_00__CT_place_STAIRS`: contribution `-0.024466`
- `lag_04__T_place_JUNGLE`: contribution `-0.021645`
- `lag_02__CT_place_STAIRS`: contribution `-0.019078`
- `lag_00__CT2__duck_amount`: contribution `+0.007882`
- `lag_00__T4__shots_fired`: contribution `-0.007013`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `80592`, seconds `17.00`, LSTM delta `-0.1028`

Top all feature movements:
- `lag_00__CT_place_SCAFFOLDING`: contribution `-0.045961`
- `lag_04__CT_place_TRUCK`: contribution `-0.007519`
- `lag_10__CT_place_JUNGLE`: contribution `-0.005341`
- `lag_15__T_place_HOUSE`: contribution `-0.004899`
- `lag_10__T_place_UNDERPASS`: contribution `-0.003827`

Top utility-only movements:
- `lag_09__CT3__flash_duration`: contribution `-0.002683`
- `lag_08__T1__flash_duration`: contribution `-0.001902`
