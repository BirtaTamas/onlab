# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-heroic-vs-aurora-bo3-0XrprgXu_t-aBJHUPpJYb4/heroic-vs-aurora-m1-overpass.csv`
- round_num: `7`

## Largest probability jumps

- tick `63840`, seconds `35.00`, LSTM `0.7871`, delta `+0.1091`
- tick `66688`, seconds `79.50`, LSTM `0.9329`, delta `+0.0773`
- tick `66144`, seconds `71.00`, LSTM `0.8409`, delta `+0.0559`
- tick `61856`, seconds `4.00`, LSTM `0.7251`, delta `-0.0341`
- tick `63872`, seconds `35.50`, LSTM `0.8197`, delta `+0.0326`
- tick `64416`, seconds `44.00`, LSTM `0.8503`, delta `+0.0324`
- tick `66752`, seconds `80.50`, LSTM `0.9697`, delta `+0.0314`
- tick `65568`, seconds `62.00`, LSTM `0.7368`, delta `-0.0292`
- tick `63808`, seconds `34.50`, LSTM `0.6779`, delta `-0.0279`
- tick `66080`, seconds `70.00`, LSTM `0.7955`, delta `+0.0278`

## Top 15 local ridge features

- `lag_14__T_place_CONNECTOR`: coefficient `0.001001`, |coef| `0.001001`
- `lag_07__CT_flashed_players`: coefficient `0.000945`, |coef| `0.000945`
- `lag_01__T4__is_walking`: coefficient `0.000882`, |coef| `0.000882`
- `lag_00__CT_kills_last_3s`: coefficient `0.000858`, |coef| `0.000858`
- `lag_00__CT_place_STORAGEROOM`: coefficient `0.000847`, |coef| `0.000847`
- `lag_14__T2__duck_amount`: coefficient `-0.000815`, |coef| `0.000815`
- `lag_09__T5__duck_amount`: coefficient `0.000812`, |coef| `0.000812`
- `lag_00__T4__is_walking`: coefficient `-0.000806`, |coef| `0.000806`
- `lag_01__CT1__duck_amount`: coefficient `-0.000801`, |coef| `0.000801`
- `lag_00__T1__flash`: coefficient `-0.000793`, |coef| `0.000793`
- `lag_03__T5__is_walking`: coefficient `-0.000755`, |coef| `0.000755`
- `lag_00__CT_place_BACKOFA`: coefficient `0.000752`, |coef| `0.000752`
- `lag_00__T_place_CONSTRUCTION`: coefficient `-0.000747`, |coef| `0.000747`
- `lag_00__T3__is_walking`: coefficient `-0.000731`, |coef| `0.000731`
- `lag_00__CT5__is_scoped`: coefficient `-0.000724`, |coef| `0.000724`

## Top 10 utility ridge features

- `lag_00__T1__flash`: coefficient `-0.000793` (lowers CT win probability)
- `lag_07__CT3__flash_duration`: coefficient `0.000714` (raises CT win probability)
- `lag_00__T1__utility_total`: coefficient `-0.000695` (lowers CT win probability)
- `lag_00__T1__molly`: coefficient `-0.000631` (lowers CT win probability)
- `lag_01__CT3__flash_duration`: coefficient `-0.000621` (lowers CT win probability)
- `lag_06__CT_utility_damage_last_5s`: coefficient `0.000588` (raises CT win probability)
- `lag_01__CT2__molly`: coefficient `-0.000561` (lowers CT win probability)
- `lag_06__utility_damage_diff_last_5s`: coefficient `0.000559` (raises CT win probability)
- `lag_04__CT2__smoke`: coefficient `-0.000505` (lowers CT win probability)
- `lag_00__CT4__flash`: coefficient `-0.000494` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_14__T_place_CONNECTOR`: coefficient `0.001001` (raises CT win probability)
- `lag_07__CT_flashed_players`: coefficient `0.000945` (raises CT win probability)
- `lag_01__T4__is_walking`: coefficient `0.000882` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000858` (raises CT win probability)
- `lag_00__CT_place_STORAGEROOM`: coefficient `0.000847` (raises CT win probability)
- `lag_14__T2__duck_amount`: coefficient `-0.000815` (lowers CT win probability)
- `lag_09__T5__duck_amount`: coefficient `0.000812` (raises CT win probability)
- `lag_00__T4__is_walking`: coefficient `-0.000806` (lowers CT win probability)
- `lag_01__CT1__duck_amount`: coefficient `-0.000801` (lowers CT win probability)
- `lag_03__T5__is_walking`: coefficient `-0.000755` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `63840`, seconds `35.00`, LSTM delta `+0.1091`

Top all feature movements:
- `lag_14__T_place_CONNECTOR`: contribution `+0.004847`
- `lag_07__CT_flashed_players`: contribution `+0.004141`
- `lag_14__T2__duck_amount`: contribution `+0.003116`
- `lag_09__T5__duck_amount`: contribution `+0.003084`
- `lag_00__T_place_CONNECTOR`: contribution `+0.003062`

Top utility-only movements:
- `lag_07__CT3__flash_duration`: contribution `+0.002257`
- `lag_00__T1__flash`: contribution `+0.002206`
- `lag_01__CT3__flash_duration`: contribution `+0.001962`
- `lag_00__T1__utility_total`: contribution `+0.001635`

### tick `66688`, seconds `79.50`, LSTM delta `+0.0773`

Top all feature movements:
- `lag_09__CT_place_STORAGEROOM`: contribution `+0.011306`
- `lag_00__T_place_CONSTRUCTION`: contribution `+0.009290`
- `lag_14__T_place_CONSTRUCTION`: contribution `+0.007733`
- `lag_04__T3__flash_duration`: contribution `+0.003055`
- `lag_00__CT5__is_scoped`: contribution `+0.002589`

Top utility-only movements:
- `lag_04__T3__flash_duration`: contribution `+0.003055`
- `lag_00__CT4__flash_duration`: contribution `+0.002001`
- `lag_04__T2__flash_duration`: contribution `+0.001862`
- `lag_04__T_flash_duration_sum`: contribution `+0.001319`

### tick `66144`, seconds `71.00`, LSTM delta `+0.0559`

Top all feature movements:
- `lag_00__CT_place_STORAGEROOM`: contribution `+0.018129`
- `lag_11__T_place_CONSTRUCTION`: contribution `+0.007548`
- `lag_08__CT2__duck_amount`: contribution `-0.002705`
- `lag_05__CT_place_WATER`: contribution `+0.002452`
- `lag_00__CT_place_LOBBY`: contribution `+0.002241`

Top utility-only movements:
- `lag_03__CT_B_site_active_infernos`: contribution `+0.001145`

### tick `61856`, seconds `4.00`, LSTM delta `-0.0341`

Top all feature movements:
- `lag_00__CT_place_BACKOFA`: contribution `-0.007265`
- `lag_00__CT_place_STAIRS`: contribution `-0.004843`
- `lag_02__CT_place_STAIRS`: contribution `-0.004269`
- `lag_02__CT_place_BACKOFA`: contribution `-0.004010`
- `lag_05__CT_place_BACKOFA`: contribution `-0.002431`

Top utility-only movements:
- `lag_08__T1__flash`: contribution `-0.000283`
- `lag_08__T3__flash`: contribution `-0.000272`
- `lag_08__T1__utility_total`: contribution `-0.000260`

### tick `63872`, seconds `35.50`, LSTM delta `+0.0326`

Top all feature movements:
- `lag_01__T_place_CONNECTOR`: contribution `+0.003242`
- `lag_15__T_place_CONNECTOR`: contribution `+0.002248`
- `lag_01__T4__is_walking`: contribution `-0.002035`
- `lag_08__CT_flashed_players`: contribution `+0.002001`
- `lag_09__CT2__duck_amount`: contribution `-0.001737`

Top utility-only movements:
- `lag_08__CT3__flash_duration`: contribution `+0.001095`
