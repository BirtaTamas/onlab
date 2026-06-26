# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-imperial-vs-legacy-bo3-GRvbnL5Q4zT_JzAd-0AXgo/imperial-vs-legacy-m2-dust2.csv`
- round_num: `11`

## Largest probability jumps

- tick `103828`, seconds `41.50`, LSTM `0.0620`, delta `-0.1693`
- tick `106516`, seconds `83.50`, LSTM `0.1606`, delta `+0.1097`
- tick `106644`, seconds `85.50`, LSTM `0.0208`, delta `-0.0710`
- tick `103796`, seconds `41.00`, LSTM `0.2314`, delta `-0.0439`
- tick `103220`, seconds `32.00`, LSTM `0.2601`, delta `-0.0401`
- tick `101748`, seconds `9.00`, LSTM `0.3093`, delta `-0.0330`
- tick `107764`, seconds `103.00`, LSTM `0.0452`, delta `+0.0321`
- tick `102196`, seconds `16.00`, LSTM `0.3174`, delta `-0.0311`
- tick `103572`, seconds `37.50`, LSTM `0.2713`, delta `-0.0310`
- tick `106548`, seconds `84.00`, LSTM `0.1306`, delta `-0.0299`

## Top 15 local ridge features

- `lag_11__T_place_TUNNELSTAIRS`: coefficient `-0.001318`, |coef| `0.001318`
- `lag_15__CT_place_SHORTSTAIRS`: coefficient `0.001270`, |coef| `0.001270`
- `lag_12__T_place_UNDERA`: coefficient `-0.001258`, |coef| `0.001258`
- `lag_00__kill_diff_last_3s`: coefficient `0.001066`, |coef| `0.001066`
- `lag_15__CT_place_EXTENDEDA`: coefficient `-0.001066`, |coef| `0.001066`
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.001039`, |coef| `0.001039`
- `lag_02__T_flashed_players`: coefficient `-0.000982`, |coef| `0.000982`
- `lag_00__damage_diff_last_5s`: coefficient `0.000941`, |coef| `0.000941`
- `lag_12__CT_place_UNDERA`: coefficient `-0.000923`, |coef| `0.000923`
- `lag_01__T_utility_damage_last_5s`: coefficient `-0.000912`, |coef| `0.000912`
- `lag_00__T_kills_last_3s`: coefficient `-0.000895`, |coef| `0.000895`
- `lag_02__CT_burning_players`: coefficient `-0.000895`, |coef| `0.000895`
- `lag_01__CT_burning_players`: coefficient `-0.000885`, |coef| `0.000885`
- `lag_02__T_place_SHORTSTAIRS`: coefficient `-0.000810`, |coef| `0.000810`
- `lag_00__CT_place_ARAMP`: coefficient `-0.000807`, |coef| `0.000807`

## Top 10 utility ridge features

- `lag_00__T_utility_damage_last_5s`: coefficient `-0.001039` (lowers CT win probability)
- `lag_01__T_utility_damage_last_5s`: coefficient `-0.000912` (lowers CT win probability)
- `lag_00__CT3__flash`: coefficient `0.000606` (raises CT win probability)
- `lag_00__T2__molly`: coefficient `0.000594` (raises CT win probability)
- `lag_02__T_utility_damage_last_5s`: coefficient `-0.000577` (lowers CT win probability)
- `lag_03__T_A_site_active_infernos`: coefficient `-0.000576` (lowers CT win probability)
- `lag_02__T1__flash_duration`: coefficient `-0.000552` (lowers CT win probability)
- `lag_07__T1__molly`: coefficient `0.000549` (raises CT win probability)
- `lag_09__CT3__smoke`: coefficient `0.000518` (raises CT win probability)
- `lag_01__active_infernos_total`: coefficient `0.000509` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_11__T_place_TUNNELSTAIRS`: coefficient `-0.001318` (lowers CT win probability)
- `lag_15__CT_place_SHORTSTAIRS`: coefficient `0.001270` (raises CT win probability)
- `lag_12__T_place_UNDERA`: coefficient `-0.001258` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001066` (raises CT win probability)
- `lag_15__CT_place_EXTENDEDA`: coefficient `-0.001066` (lowers CT win probability)
- `lag_02__T_flashed_players`: coefficient `-0.000982` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000941` (raises CT win probability)
- `lag_12__CT_place_UNDERA`: coefficient `-0.000923` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.000895` (lowers CT win probability)
- `lag_02__CT_burning_players`: coefficient `-0.000895` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `103828`, seconds `41.50`, LSTM delta `-0.1693`

Top all feature movements:
- `lag_11__T_place_TUNNELSTAIRS`: contribution `-0.009201`
- `lag_15__CT_place_SHORTSTAIRS`: contribution `-0.007081`
- `lag_15__CT_place_EXTENDEDA`: contribution `-0.005983`
- `lag_02__T_flashed_players`: contribution `-0.005684`
- `lag_06__T_place_TUNNELSTAIRS`: contribution `-0.005249`

Top utility-only movements:
- `lag_00__T_utility_damage_last_5s`: contribution `-0.002374`

### tick `106516`, seconds `83.50`, LSTM delta `+0.1097`

Top all feature movements:
- `lag_12__T_place_UNDERA`: contribution `+0.019660`
- `lag_07__T_shots_fired_sum`: contribution `+0.004051`
- `lag_00__T_place_LOWERTUNNEL`: contribution `+0.003204`
- `lag_12__T_place_CTSPAWN`: contribution `+0.003041`
- `lag_11__T4__duck_amount`: contribution `+0.002842`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `106644`, seconds `85.50`, LSTM delta `-0.0710`

Top all feature movements:
- `lag_09__T_place_LOWERTUNNEL`: contribution `-0.003034`
- `lag_13__T5__is_scoped`: contribution `-0.002892`
- `lag_00__T_kills_last_3s`: contribution `-0.002837`
- `lag_00__kill_diff_last_3s`: contribution `-0.002566`
- `lag_01__CT_place_HOLE`: contribution `-0.002496`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `103796`, seconds `41.00`, LSTM delta `-0.0439`

Top all feature movements:
- `lag_10__T_place_TUNNELSTAIRS`: contribution `-0.004362`
- `lag_14__CT_place_SHORTSTAIRS`: contribution `-0.004282`
- `lag_05__T_place_TUNNELSTAIRS`: contribution `-0.003303`
- `lag_14__CT_place_EXTENDEDA`: contribution `-0.003237`
- `lag_00__CT_place_EXTENDEDA`: contribution `+0.003045`

Top utility-only movements:
- `lag_00__T_utility_damage_last_5s`: contribution `-0.002225`
- `lag_02__T_A_site_active_infernos`: contribution `-0.001075`

### tick `103220`, seconds `32.00`, LSTM delta `-0.0401`

Top all feature movements:
- `lag_00__CT_place_ARAMP`: contribution `-0.005029`
- `lag_10__T_place_LOWERTUNNEL`: contribution `-0.001985`
- `lag_15__T5__duck_amount`: contribution `-0.001702`
- `lag_09__CT4__is_walking`: contribution `-0.001694`
- `lag_04__T4__duck_amount`: contribution `-0.001601`

Top utility-only movements:
- `lag_00__T2__molly`: contribution `-0.001323`
