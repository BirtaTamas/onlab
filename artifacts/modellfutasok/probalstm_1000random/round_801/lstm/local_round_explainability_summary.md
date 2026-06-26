# Local Round Explainability

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-falcons-vs-g2-bo3-Xf_UKx9fB2btv0Vy_VBVUC/falcons-vs-g2-m3-mirage.csv`
- round_num: `5`

## Largest probability jumps

- tick `26908`, seconds `82.50`, LSTM `0.4206`, delta `+0.3031`
- tick `26236`, seconds `72.00`, LSTM `0.3124`, delta `-0.2308`
- tick `28060`, seconds `100.50`, LSTM `0.8483`, delta `+0.1950`
- tick `22940`, seconds `20.50`, LSTM `0.8081`, delta `+0.1732`
- tick `22620`, seconds `15.50`, LSTM `0.5893`, delta `+0.1202`
- tick `25020`, seconds `53.00`, LSTM `0.6719`, delta `-0.1186`
- tick `26972`, seconds `83.50`, LSTM `0.5401`, delta `+0.1058`
- tick `27932`, seconds `98.50`, LSTM `0.6122`, delta `+0.0876`
- tick `26268`, seconds `72.50`, LSTM `0.2416`, delta `-0.0708`
- tick `26332`, seconds `73.50`, LSTM `0.1601`, delta `-0.0654`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003916`, |coef| `0.003916`
- `lag_00__CT_kills_last_3s`: coefficient `0.003184`, |coef| `0.003184`
- `lag_00__CT_place_TRUCK`: coefficient `0.003060`, |coef| `0.003060`
- `lag_12__CT_place_SHOP`: coefficient `-0.003014`, |coef| `0.003014`
- `lag_00__damage_diff_last_5s`: coefficient `0.002987`, |coef| `0.002987`
- `lag_14__T_place_JUNGLE`: coefficient `0.002611`, |coef| `0.002611`
- `lag_15__T3__duck_amount`: coefficient `0.002440`, |coef| `0.002440`
- `lag_07__T_place_CATWALK`: coefficient `-0.002313`, |coef| `0.002313`
- `lag_08__T_place_CATWALK`: coefficient `-0.002100`, |coef| `0.002100`
- `lag_02__T_place_CTSPAWN`: coefficient `-0.001998`, |coef| `0.001998`
- `lag_14__T_place_CTSPAWN`: coefficient `-0.001944`, |coef| `0.001944`
- `lag_01__T_place_CONNECTOR`: coefficient `0.001914`, |coef| `0.001914`
- `lag_04__T3__flash_duration`: coefficient `0.001874`, |coef| `0.001874`
- `lag_00__T1__has_bomb`: coefficient `-0.001825`, |coef| `0.001825`
- `lag_04__CT3__flash_duration`: coefficient `0.001797`, |coef| `0.001797`

## Top 10 utility ridge features

- `lag_04__T3__flash_duration`: coefficient `0.001874` (raises CT win probability)
- `lag_04__CT3__flash_duration`: coefficient `0.001797` (raises CT win probability)
- `lag_04__T4__flash_duration`: coefficient `0.001781` (raises CT win probability)
- `lag_09__T1__flash_duration`: coefficient `0.001480` (raises CT win probability)
- `lag_00__T1__molly`: coefficient `-0.001455` (lowers CT win probability)
- `lag_03__T1__flash_duration`: coefficient `-0.001322` (lowers CT win probability)
- `lag_02__T1__flash_duration`: coefficient `-0.001298` (lowers CT win probability)
- `lag_09__T1__molly`: coefficient `-0.001256` (lowers CT win probability)
- `lag_04__T1__flash_duration`: coefficient `-0.001136` (lowers CT win probability)
- `lag_04__T_flash_duration_sum`: coefficient `0.001030` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003916` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003184` (raises CT win probability)
- `lag_00__CT_place_TRUCK`: coefficient `0.003060` (raises CT win probability)
- `lag_12__CT_place_SHOP`: coefficient `-0.003014` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002987` (raises CT win probability)
- `lag_14__T_place_JUNGLE`: coefficient `0.002611` (raises CT win probability)
- `lag_15__T3__duck_amount`: coefficient `0.002440` (raises CT win probability)
- `lag_07__T_place_CATWALK`: coefficient `-0.002313` (lowers CT win probability)
- `lag_08__T_place_CATWALK`: coefficient `-0.002100` (lowers CT win probability)
- `lag_02__T_place_CTSPAWN`: coefficient `-0.001998` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `26908`, seconds `82.50`, LSTM delta `+0.3031`

Top all feature movements:
- `lag_14__T_place_JUNGLE`: contribution `+0.033825`
- `lag_03__T_place_JUNGLE`: contribution `+0.022016`
- `lag_12__CT_place_SHOP`: contribution `+0.015118`
- `lag_00__kill_diff_last_3s`: contribution `+0.009426`
- `lag_14__T_place_CTSPAWN`: contribution `+0.009275`

Top utility-only movements:
- `lag_04__T3__flash_duration`: contribution `+0.004865`
- `lag_00__T1__molly`: contribution `+0.003222`

### tick `26236`, seconds `72.00`, LSTM delta `-0.2308`

Top all feature movements:
- `lag_02__T_place_JUNGLE`: contribution `-0.020494`
- `lag_12__CT_place_SHOP`: contribution `-0.015118`
- `lag_02__T_place_CTSPAWN`: contribution `-0.009532`
- `lag_00__kill_diff_last_3s`: contribution `-0.009426`
- `lag_07__T_place_CATWALK`: contribution `-0.006659`

Top utility-only movements:
- `lag_09__T1__flash_duration`: contribution `-0.006295`
- `lag_01__T3__flash_duration`: contribution `-0.003492`
- `lag_09__T1__molly`: contribution `-0.002781`

### tick `28060`, seconds `100.50`, LSTM delta `+0.1950`

Top all feature movements:
- `lag_04__T4__flash_duration`: contribution `+0.013329`
- `lag_04__T3__flash_duration`: contribution `+0.011626`
- `lag_04__CT3__flash_duration`: contribution `+0.010943`
- `lag_00__kill_diff_last_3s`: contribution `+0.009426`
- `lag_15__T3__duck_amount`: contribution `+0.009199`

Top utility-only movements:
- `lag_04__T4__flash_duration`: contribution `+0.013329`
- `lag_04__T3__flash_duration`: contribution `+0.011626`
- `lag_04__CT3__flash_duration`: contribution `+0.010943`
- `lag_04__T_flash_duration_sum`: contribution `+0.005799`
- `lag_15__T_B_site_active_infernos`: contribution `+0.002616`

### tick `22940`, seconds `20.50`, LSTM delta `+0.1732`

Top all feature movements:
- `lag_01__CT_place_TRUCK`: contribution `+0.010563`
- `lag_00__kill_diff_last_3s`: contribution `+0.009426`
- `lag_00__CT_kills_last_3s`: contribution `+0.009192`
- `lag_11__CT4__is_scoped`: contribution `+0.004067`
- `lag_05__CT_place_TRUCK`: contribution `+0.003909`

Top utility-only movements:
- `lag_06__CT_active_infernos`: contribution `+0.003647`
- `lag_06__CT_B_site_active_infernos`: contribution `+0.002552`

### tick `22620`, seconds `15.50`, LSTM delta `+0.1202`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.009426`
- `lag_00__CT_kills_last_3s`: contribution `+0.009192`
- `lag_15__CT_place_SHOP`: contribution `+0.007598`
- `lag_00__damage_diff_last_5s`: contribution `+0.004987`
- `lag_12__T4__is_walking`: contribution `-0.003673`

Top utility-only movements:
- `lag_00__T2__utility_total`: contribution `+0.002432`
- `lag_05__CT_A_site_active_infernos`: contribution `+0.001971`
