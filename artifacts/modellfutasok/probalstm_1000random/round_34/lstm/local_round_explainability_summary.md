# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-heroic-vs-aurora-bo3-0XrprgXu_t-aBJHUPpJYb4/heroic-vs-aurora-m1-overpass.csv`
- round_num: `6`

## Largest probability jumps

- tick `56977`, seconds `102.50`, LSTM `0.9364`, delta `+0.2199`
- tick `57617`, seconds `112.50`, LSTM `0.4455`, delta `-0.2072`
- tick `57265`, seconds `107.00`, LSTM `0.7567`, delta `-0.1817`
- tick `56785`, seconds `99.50`, LSTM `0.7837`, delta `-0.1768`
- tick `56561`, seconds `96.00`, LSTM `0.8169`, delta `+0.1016`
- tick `56593`, seconds `96.50`, LSTM `0.8998`, delta `+0.0829`
- tick `56433`, seconds `94.00`, LSTM `0.6739`, delta `+0.0766`
- tick `56817`, seconds `100.00`, LSTM `0.8504`, delta `+0.0667`
- tick `57329`, seconds `108.00`, LSTM `0.7280`, delta `-0.0610`
- tick `50673`, seconds `4.00`, LSTM `0.6160`, delta `-0.0502`

## Top 15 local ridge features

- `lag_08__T_place_CONSTRUCTION`: coefficient `0.002248`, |coef| `0.002248`
- `lag_09__CT_place_STORAGEROOM`: coefficient `-0.001668`, |coef| `0.001668`
- `lag_00__kill_diff_last_3s`: coefficient `0.001637`, |coef| `0.001637`
- `lag_12__T_place_CONSTRUCTION`: coefficient `0.001614`, |coef| `0.001614`
- `lag_13__T_place_CONSTRUCTION`: coefficient `0.001609`, |coef| `0.001609`
- `lag_00__CT_place_WATER`: coefficient `0.001528`, |coef| `0.001528`
- `lag_00__T_kills_last_3s`: coefficient `-0.001505`, |coef| `0.001505`
- `lag_00__T_duck_amount_mean`: coefficient `-0.001294`, |coef| `0.001294`
- `lag_04__T_utility_damage_last_5s`: coefficient `-0.001179`, |coef| `0.001179`
- `lag_01__T_place_CONSTRUCTION`: coefficient `0.001142`, |coef| `0.001142`
- `lag_15__CT4__shots_fired`: coefficient `0.001120`, |coef| `0.001120`
- `lag_02__CT_shots_fired_sum`: coefficient `0.001118`, |coef| `0.001118`
- `lag_10__T_duck_amount_mean`: coefficient `0.001091`, |coef| `0.001091`
- `lag_04__T_place_CONSTRUCTION`: coefficient `0.001071`, |coef| `0.001071`
- `lag_15__CT_place_WATER`: coefficient `-0.001029`, |coef| `0.001029`

## Top 10 utility ridge features

- `lag_04__T_utility_damage_last_5s`: coefficient `-0.001179` (lowers CT win probability)
- `lag_05__T_utility_damage_last_5s`: coefficient `-0.000888` (lowers CT win probability)
- `lag_04__utility_damage_diff_last_5s`: coefficient `0.000823` (raises CT win probability)
- `lag_06__T_utility_damage_last_5s`: coefficient `-0.000784` (lowers CT win probability)
- `lag_15__T_A_site_active_infernos`: coefficient `-0.000766` (lowers CT win probability)
- `lag_12__T2__smoke`: coefficient `-0.000755` (lowers CT win probability)
- `lag_01__T_A_site_active_infernos`: coefficient `0.000740` (raises CT win probability)
- `lag_15__T_B_site_active_infernos`: coefficient `-0.000676` (lowers CT win probability)
- `lag_03__T_utility_damage_last_5s`: coefficient `-0.000669` (lowers CT win probability)
- `lag_01__T_B_site_active_infernos`: coefficient `0.000639` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_08__T_place_CONSTRUCTION`: coefficient `0.002248` (raises CT win probability)
- `lag_09__CT_place_STORAGEROOM`: coefficient `-0.001668` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001637` (raises CT win probability)
- `lag_12__T_place_CONSTRUCTION`: coefficient `0.001614` (raises CT win probability)
- `lag_13__T_place_CONSTRUCTION`: coefficient `0.001609` (raises CT win probability)
- `lag_00__CT_place_WATER`: coefficient `0.001528` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001505` (lowers CT win probability)
- `lag_00__T_duck_amount_mean`: coefficient `-0.001294` (lowers CT win probability)
- `lag_01__T_place_CONSTRUCTION`: coefficient `0.001142` (raises CT win probability)
- `lag_15__CT4__shots_fired`: coefficient `0.001120` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `56977`, seconds `102.50`, LSTM delta `+0.2199`

Top all feature movements:
- `lag_09__CT_place_STORAGEROOM`: contribution `+0.035677`
- `lag_08__T_place_CONSTRUCTION`: contribution `+0.027939`
- `lag_12__T_place_CONSTRUCTION`: contribution `-0.020064`
- `lag_11__CT_place_BRIDGE`: contribution `+0.010538`
- `lag_08__CT_place_BRIDGE`: contribution `+0.010472`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `57617`, seconds `112.50`, LSTM delta `-0.2072`

Top all feature movements:
- `lag_12__T_place_CONSTRUCTION`: contribution `-0.020064`
- `lag_00__CT_place_WATER`: contribution `-0.009288`
- `lag_00__T_duck_amount_mean`: contribution `-0.007451`
- `lag_10__T_duck_amount_mean`: contribution `-0.006343`
- `lag_13__CT_shots_fired_sum`: contribution `-0.006216`

Top utility-only movements:
- `lag_04__T_utility_damage_last_5s`: contribution `-0.004039`
- `lag_15__T_A_site_active_infernos`: contribution `-0.002281`
- `lag_01__T_A_site_active_infernos`: contribution `-0.002204`

### tick `57265`, seconds `107.00`, LSTM delta `-0.1817`

Top all feature movements:
- `lag_01__T_place_CONSTRUCTION`: contribution `-0.014195`
- `lag_00__T_duck_amount_mean`: contribution `-0.007528`
- `lag_02__CT_shots_fired_sum`: contribution `-0.006989`
- `lag_14__T2__is_scoped`: contribution `-0.005589`
- `lag_15__CT_place_STAIRS`: contribution `-0.005345`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `56785`, seconds `99.50`, LSTM delta `-0.1768`

Top all feature movements:
- `lag_08__T_place_CONSTRUCTION`: contribution `-0.027939`
- `lag_12__T_place_CONSTRUCTION`: contribution `+0.020064`
- `lag_11__CT_place_STORAGEROOM`: contribution `-0.016638`
- `lag_01__T_place_CONSTRUCTION`: contribution `-0.014195`
- `lag_03__CT_place_STORAGEROOM`: contribution `-0.013423`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `56561`, seconds `96.00`, LSTM delta `+0.1016`

Top all feature movements:
- `lag_08__T_place_CONSTRUCTION`: contribution `+0.027939`
- `lag_04__CT_place_STORAGEROOM`: contribution `+0.017778`
- `lag_01__T_place_CONSTRUCTION`: contribution `-0.014195`
- `lag_05__T_place_CONSTRUCTION`: contribution `+0.010307`
- `lag_15__CT_place_WATER`: contribution `+0.006255`

Top utility-only movements:
- No utility movement among the top local contributors.
