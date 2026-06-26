# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-spirit-vs-flyquest-bo3-fQI-qOiPd1cRkmhkz0Xs5h/spirit-vs-flyquest-m1-mirage.csv`
- round_num: `9`

## Largest probability jumps

- tick `63532`, seconds `29.50`, LSTM `0.8397`, delta `+0.1168`
- tick `63628`, seconds `31.00`, LSTM `0.9310`, delta `+0.0757`
- tick `62252`, seconds `9.50`, LSTM `0.7453`, delta `-0.0384`
- tick `63564`, seconds `30.00`, LSTM `0.8754`, delta `+0.0358`
- tick `63500`, seconds `29.00`, LSTM `0.7229`, delta `-0.0352`
- tick `62892`, seconds `19.50`, LSTM `0.7357`, delta `+0.0322`
- tick `64748`, seconds `48.50`, LSTM `0.9643`, delta `+0.0281`
- tick `62284`, seconds `10.00`, LSTM `0.7190`, delta `-0.0263`
- tick `62316`, seconds `10.50`, LSTM `0.6941`, delta `-0.0249`
- tick `61708`, seconds `1.00`, LSTM `0.6787`, delta `-0.0242`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001012`, |coef| `0.001012`
- `lag_12__CT_place_UNDERPASS`: coefficient `0.000906`, |coef| `0.000906`
- `lag_01__T1__duck_amount`: coefficient `-0.000859`, |coef| `0.000859`
- `lag_00__kill_diff_last_3s`: coefficient `0.000844`, |coef| `0.000844`
- `lag_09__T_mollies_last_5s`: coefficient `-0.000786`, |coef| `0.000786`
- `lag_01__T_place_CONNECTOR`: coefficient `0.000772`, |coef| `0.000772`
- `lag_00__CT4__is_scoped`: coefficient `-0.000769`, |coef| `0.000769`
- `lag_00__CT_damage_last_5s`: coefficient `0.000731`, |coef| `0.000731`
- `lag_00__T_place_TRAMP`: coefficient `-0.000729`, |coef| `0.000729`
- `lag_01__CT_place_STAIRS`: coefficient `0.000699`, |coef| `0.000699`
- `lag_00__damage_diff_last_5s`: coefficient `0.000680`, |coef| `0.000680`
- `lag_03__CT_utility_damage_last_5s`: coefficient `-0.000663`, |coef| `0.000663`
- `lag_13__CT_place_JUNGLE`: coefficient `0.000643`, |coef| `0.000643`
- `lag_09__CT1__duck_amount`: coefficient `-0.000634`, |coef| `0.000634`
- `lag_00__T1__is_walking`: coefficient `-0.000611`, |coef| `0.000611`

## Top 10 utility ridge features

- `lag_09__T_mollies_last_5s`: coefficient `-0.000786` (lowers CT win probability)
- `lag_03__CT_utility_damage_last_5s`: coefficient `-0.000663` (lowers CT win probability)
- `lag_13__CT_utility_damage_last_5s`: coefficient `0.000594` (raises CT win probability)
- `lag_04__T_A_site_active_infernos`: coefficient `0.000554` (raises CT win probability)
- `lag_03__utility_damage_diff_last_5s`: coefficient `-0.000518` (lowers CT win probability)
- `lag_13__utility_damage_diff_last_5s`: coefficient `0.000497` (raises CT win probability)
- `lag_00__CT_mollies_last_5s`: coefficient `-0.000468` (lowers CT win probability)
- `lag_05__T3__molly`: coefficient `-0.000441` (lowers CT win probability)
- `lag_08__CT_mollies_last_5s`: coefficient `0.000351` (raises CT win probability)
- `lag_12__T_flashes_last_5s`: coefficient `-0.000350` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001012` (raises CT win probability)
- `lag_12__CT_place_UNDERPASS`: coefficient `0.000906` (raises CT win probability)
- `lag_01__T1__duck_amount`: coefficient `-0.000859` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000844` (raises CT win probability)
- `lag_01__T_place_CONNECTOR`: coefficient `0.000772` (raises CT win probability)
- `lag_00__CT4__is_scoped`: coefficient `-0.000769` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.000731` (raises CT win probability)
- `lag_00__T_place_TRAMP`: coefficient `-0.000729` (lowers CT win probability)
- `lag_01__CT_place_STAIRS`: coefficient `0.000699` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000680` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `63532`, seconds `29.50`, LSTM delta `+0.1168`

Top all feature movements:
- `lag_12__CT_place_UNDERPASS`: contribution `+0.005253`
- `lag_01__T_place_CONNECTOR`: contribution `+0.003736`
- `lag_01__T1__duck_amount`: contribution `+0.003362`
- `lag_00__CT_kills_last_3s`: contribution `+0.002923`
- `lag_03__CT_utility_damage_last_5s`: contribution `+0.002845`

Top utility-only movements:
- `lag_03__CT_utility_damage_last_5s`: contribution `+0.002845`
- `lag_13__CT_utility_damage_last_5s`: contribution `+0.002551`
- `lag_03__utility_damage_diff_last_5s`: contribution `+0.001823`
- `lag_13__utility_damage_diff_last_5s`: contribution `+0.001751`
- `lag_04__T_A_site_active_infernos`: contribution `+0.001649`

### tick `63628`, seconds `31.00`, LSTM delta `+0.0757`

Top all feature movements:
- `lag_15__CT_place_UNDERPASS`: contribution `+0.003289`
- `lag_00__CT_kills_last_3s`: contribution `+0.002923`
- `lag_02__CT_place_SNIPERSNEST`: contribution `+0.002534`
- `lag_04__T_place_CONNECTOR`: contribution `+0.002387`
- `lag_00__kill_diff_last_3s`: contribution `+0.002031`

Top utility-only movements:
- `lag_06__CT_utility_damage_last_5s`: contribution `+0.001458`

### tick `62252`, seconds `9.50`, LSTM delta `-0.0384`

Top all feature movements:
- `lag_09__T_mollies_last_5s`: contribution `-0.016152`
- `lag_07__CT_mollies_last_5s`: contribution `-0.004049`
- `lag_12__T_flashes_last_5s`: contribution `-0.003172`
- `lag_08__CT_place_SHOP`: contribution `-0.002082`
- `lag_01__T_place_PALACEINTERIOR`: contribution `-0.001674`

Top utility-only movements:
- `lag_09__T_mollies_last_5s`: contribution `-0.016152`
- `lag_07__CT_mollies_last_5s`: contribution `-0.004049`
- `lag_12__T_flashes_last_5s`: contribution `-0.003172`
- `lag_07__CT_he_last_5s`: contribution `-0.001207`

### tick `63564`, seconds `30.00`, LSTM delta `+0.0358`

Top all feature movements:
- `lag_13__CT_place_UNDERPASS`: contribution `+0.003057`
- `lag_00__CT_place_SNIPERSNEST`: contribution `+0.002616`
- `lag_02__T1__duck_amount`: contribution `-0.001794`
- `lag_03__CT4__is_scoped`: contribution `+0.001708`
- `lag_01__T_place_PALACEINTERIOR`: contribution `+0.001674`

Top utility-only movements:
- `lag_14__CT_utility_damage_last_5s`: contribution `+0.001029`
- `lag_14__utility_damage_diff_last_5s`: contribution `+0.000955`

### tick `63500`, seconds `29.00`, LSTM delta `-0.0352`

Top all feature movements:
- `lag_01__T1__duck_amount`: contribution `-0.003362`
- `lag_10__CT_place_JUNGLE`: contribution `-0.002646`
- `lag_00__CT4__is_scoped`: contribution `-0.002622`
- `lag_09__CT1__duck_amount`: contribution `-0.002418`
- `lag_00__T_place_CONNECTOR`: contribution `-0.002020`

Top utility-only movements:
- No utility movement among the top local contributors.
