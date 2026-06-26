# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-heroic-vs-aurora-bo3-0XrprgXu_t-aBJHUPpJYb4/heroic-vs-aurora-m1-overpass.csv`
- round_num: `4`

## Largest probability jumps

- tick `36460`, seconds `54.50`, LSTM `0.7257`, delta `+0.1347`
- tick `37356`, seconds `68.50`, LSTM `0.8687`, delta `+0.0945`
- tick `38060`, seconds `79.50`, LSTM `0.9461`, delta `+0.0541`
- tick `36812`, seconds `60.00`, LSTM `0.7624`, delta `+0.0403`
- tick `33228`, seconds `4.00`, LSTM `0.6123`, delta `-0.0374`
- tick `36748`, seconds `59.00`, LSTM `0.7134`, delta `-0.0354`
- tick `33068`, seconds `1.50`, LSTM `0.6240`, delta `+0.0334`
- tick `37292`, seconds `67.50`, LSTM `0.7571`, delta `+0.0328`
- tick `36652`, seconds `57.50`, LSTM `0.7631`, delta `-0.0314`
- tick `36012`, seconds `47.50`, LSTM `0.6081`, delta `-0.0287`

## Top 15 local ridge features

- `lag_00__CT_place_BACKOFA`: coefficient `0.001087`, |coef| `0.001087`
- `lag_10__CT_place_CONSTRUCTION`: coefficient `0.001030`, |coef| `0.001030`
- `lag_00__CT_kills_last_3s`: coefficient `0.000945`, |coef| `0.000945`
- `lag_13__CT_shots_fired_sum`: coefficient `-0.000897`, |coef| `0.000897`
- `lag_13__CT_place_WALKWAY`: coefficient `0.000839`, |coef| `0.000839`
- `lag_00__kill_diff_last_3s`: coefficient `0.000793`, |coef| `0.000793`
- `lag_10__CT_place_WATER`: coefficient `0.000793`, |coef| `0.000793`
- `lag_14__CT_place_WALKWAY`: coefficient `0.000773`, |coef| `0.000773`
- `lag_15__CT_place_WALKWAY`: coefficient `0.000765`, |coef| `0.000765`
- `lag_13__CT3__shots_fired`: coefficient `-0.000756`, |coef| `0.000756`
- `lag_03__CT_place_BACKOFA`: coefficient `0.000746`, |coef| `0.000746`
- `lag_13__T2__is_scoped`: coefficient `-0.000743`, |coef| `0.000743`
- `lag_15__CT_place_WATER`: coefficient `0.000743`, |coef| `0.000743`
- `lag_12__T2__is_scoped`: coefficient `-0.000735`, |coef| `0.000735`
- `lag_15__CT_place_BRIDGE`: coefficient `-0.000676`, |coef| `0.000676`

## Top 10 utility ridge features

- `lag_08__T3__flash_duration`: coefficient `-0.000634` (lowers CT win probability)
- `lag_08__CT_B_site_active_infernos`: coefficient `-0.000492` (lowers CT win probability)
- `lag_07__T3__flash_duration`: coefficient `-0.000402` (lowers CT win probability)
- `lag_12__CT3__smoke`: coefficient `-0.000377` (lowers CT win probability)
- `lag_02__T1__flash_duration`: coefficient `-0.000350` (lowers CT win probability)
- `lag_15__T2__flash_duration`: coefficient `0.000344` (raises CT win probability)
- `lag_00__T1__utility_total`: coefficient `-0.000336` (lowers CT win probability)
- `lag_06__CT1__molly`: coefficient `-0.000324` (lowers CT win probability)
- `lag_01__CT_utility_damage_last_5s`: coefficient `0.000323` (raises CT win probability)
- `lag_05__T2__flash_duration`: coefficient `-0.000320` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_BACKOFA`: coefficient `0.001087` (raises CT win probability)
- `lag_10__CT_place_CONSTRUCTION`: coefficient `0.001030` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000945` (raises CT win probability)
- `lag_13__CT_shots_fired_sum`: coefficient `-0.000897` (lowers CT win probability)
- `lag_13__CT_place_WALKWAY`: coefficient `0.000839` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000793` (raises CT win probability)
- `lag_10__CT_place_WATER`: coefficient `0.000793` (raises CT win probability)
- `lag_14__CT_place_WALKWAY`: coefficient `0.000773` (raises CT win probability)
- `lag_15__CT_place_WALKWAY`: coefficient `0.000765` (raises CT win probability)
- `lag_13__CT3__shots_fired`: coefficient `-0.000756` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `36460`, seconds `54.50`, LSTM delta `+0.1347`

Top all feature movements:
- `lag_10__CT_place_CONSTRUCTION`: contribution `+0.012963`
- `lag_13__CT_shots_fired_sum`: contribution `+0.012464`
- `lag_13__CT3__shots_fired`: contribution `+0.007779`
- `lag_14__CT_place_LOBBY`: contribution `+0.005245`
- `lag_11__CT_place_LOBBY`: contribution `+0.005177`

Top utility-only movements:
- `lag_08__T3__flash_duration`: contribution `+0.003398`

### tick `37356`, seconds `68.50`, LSTM delta `+0.0945`

Top all feature movements:
- `lag_08__T_place_PIPE`: contribution `+0.006862`
- `lag_01__CT_place_BACKOFA`: contribution `+0.005747`
- `lag_03__CT_place_STAIRS`: contribution `+0.005040`
- `lag_01__CT_place_STAIRS`: contribution `+0.004240`
- `lag_13__CT_place_WALKWAY`: contribution `+0.004118`

Top utility-only movements:
- `lag_08__CT_B_site_active_infernos`: contribution `+0.001691`

### tick `38060`, seconds `79.50`, LSTM delta `+0.0541`

Top all feature movements:
- `lag_13__T2__is_scoped`: contribution `+0.006549`
- `lag_15__CT_place_WATER`: contribution `+0.004514`
- `lag_15__CT_place_WALKWAY`: contribution `-0.003755`
- `lag_00__CT_kills_last_3s`: contribution `+0.002730`
- `lag_00__kill_diff_last_3s`: contribution `+0.001908`

Top utility-only movements:
- `lag_15__T2__flash_duration`: contribution `+0.001899`
- `lag_05__T2__flash_duration`: contribution `+0.001765`
- `lag_00__T1__utility_total`: contribution `+0.000790`
- `lag_00__T1__flash`: contribution `+0.000786`

### tick `36812`, seconds `60.00`, LSTM delta `+0.0403`

Top all feature movements:
- `lag_08__T_place_PIPE`: contribution `-0.006862`
- `lag_11__CT_place_CONSTRUCTION`: contribution `+0.006459`
- `lag_06__T_place_PIPE`: contribution `+0.005713`
- `lag_01__CT_place_BRIDGE`: contribution `-0.004350`
- `lag_08__CT_place_LOBBY`: contribution `+0.004105`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `33228`, seconds `4.00`, LSTM delta `-0.0374`

Top all feature movements:
- `lag_00__CT_place_BACKOFA`: contribution `-0.010496`
- `lag_01__CT_place_BACKOFA`: contribution `-0.005747`
- `lag_05__CT_place_BACKOFA`: contribution `-0.004865`
- `lag_04__CT_place_BACKOFA`: contribution `+0.004694`
- `lag_02__CT_place_BACKOFA`: contribution `-0.004325`

Top utility-only movements:
- `lag_08__CT1__molly`: contribution `-0.000392`
- `lag_08__CT_molly_inv`: contribution `-0.000381`
