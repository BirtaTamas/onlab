# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-pain-bo3-6mWraId8pA69o5etX6dmBT/falcons-vs-pain-m1-inferno.csv`
- round_num: `17`

## Largest probability jumps

- tick `128364`, seconds `108.00`, LSTM `0.0814`, delta `-0.2476`
- tick `126668`, seconds `81.50`, LSTM `0.6443`, delta `+0.2406`
- tick `127788`, seconds `99.00`, LSTM `0.4277`, delta `-0.1328`
- tick `126732`, seconds `82.50`, LSTM `0.5394`, delta `-0.1278`
- tick `128204`, seconds `105.50`, LSTM `0.2550`, delta `-0.1092`
- tick `127308`, seconds `91.50`, LSTM `0.6562`, delta `+0.0976`
- tick `128332`, seconds `107.50`, LSTM `0.3291`, delta `+0.0781`
- tick `121484`, seconds `0.50`, LSTM `0.2853`, delta `-0.0714`
- tick `125132`, seconds `57.50`, LSTM `0.3651`, delta `+0.0664`
- tick `125100`, seconds `57.00`, LSTM `0.2987`, delta `+0.0629`

## Top 15 local ridge features

- `lag_02__T_place_ARCH`: coefficient `-0.003547`, |coef| `0.003547`
- `lag_00__CT_shots_fired_sum`: coefficient `0.003420`, |coef| `0.003420`
- `lag_00__kill_diff_last_3s`: coefficient `0.002239`, |coef| `0.002239`
- `lag_05__T_bomb_zone_count`: coefficient `-0.002173`, |coef| `0.002173`
- `lag_04__CT5__flash_duration`: coefficient `0.002083`, |coef| `0.002083`
- `lag_04__CT_place_LIBRARY`: coefficient `-0.001792`, |coef| `0.001792`
- `lag_00__T5__shots_fired`: coefficient `0.001766`, |coef| `0.001766`
- `lag_00__T_place_BALCONY`: coefficient `-0.001721`, |coef| `0.001721`
- `lag_00__T_kills_last_3s`: coefficient `-0.001647`, |coef| `0.001647`
- `lag_00__CT_velocity_mean`: coefficient `-0.001539`, |coef| `0.001539`
- `lag_13__CT1__flash_duration`: coefficient `0.001494`, |coef| `0.001494`
- `lag_06__CT_place_PIT`: coefficient `0.001431`, |coef| `0.001431`
- `lag_00__T_bomb_zone_count`: coefficient `-0.001384`, |coef| `0.001384`
- `lag_01__CT4__shots_fired`: coefficient `-0.001337`, |coef| `0.001337`
- `lag_00__CT2__is_walking`: coefficient `-0.001274`, |coef| `0.001274`

## Top 10 utility ridge features

- `lag_04__CT5__flash_duration`: coefficient `0.002083` (raises CT win probability)
- `lag_13__CT1__flash_duration`: coefficient `0.001494` (raises CT win probability)
- `lag_00__T_flashes_last_5s`: coefficient `-0.001266` (lowers CT win probability)
- `lag_03__CT_B_site_active_infernos`: coefficient `0.001160` (raises CT win probability)
- `lag_04__CT_flash_duration_sum`: coefficient `0.001107` (raises CT win probability)
- `lag_11__CT1__flash_duration`: coefficient `0.001070` (raises CT win probability)
- `lag_00__T4__smoke`: coefficient `-0.001068` (lowers CT win probability)
- `lag_03__T_B_site_active_infernos`: coefficient `0.001023` (raises CT win probability)
- `lag_03__active_infernos_total`: coefficient `0.000997` (raises CT win probability)
- `lag_07__T1__smoke`: coefficient `0.000940` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_02__T_place_ARCH`: coefficient `-0.003547` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.003420` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002239` (raises CT win probability)
- `lag_05__T_bomb_zone_count`: coefficient `-0.002173` (lowers CT win probability)
- `lag_04__CT_place_LIBRARY`: coefficient `-0.001792` (lowers CT win probability)
- `lag_00__T5__shots_fired`: coefficient `0.001766` (raises CT win probability)
- `lag_00__T_place_BALCONY`: coefficient `-0.001721` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001647` (lowers CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.001539` (lowers CT win probability)
- `lag_06__CT_place_PIT`: coefficient `0.001431` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `128364`, seconds `108.00`, LSTM delta `-0.2476`

Top all feature movements:
- `lag_02__T_place_ARCH`: contribution `-0.033002`
- `lag_00__CT_shots_fired_sum`: contribution `-0.014254`
- `lag_05__T_bomb_zone_count`: contribution `-0.012652`
- `lag_04__CT_place_LIBRARY`: contribution `-0.011489`
- `lag_00__T5__shots_fired`: contribution `-0.007601`

Top utility-only movements:
- `lag_07__T1__smoke`: contribution `-0.002028`

### tick `126668`, seconds `81.50`, LSTM delta `+0.2406`

Top all feature movements:
- `lag_04__CT5__flash_duration`: contribution `+0.014747`
- `lag_00__CT_shots_fired_sum`: contribution `+0.011879`
- `lag_06__CT_place_PIT`: contribution `+0.006163`
- `lag_00__T5__is_scoped`: contribution `+0.006003`
- `lag_00__kill_diff_last_3s`: contribution `+0.005388`

Top utility-only movements:
- `lag_04__CT5__flash_duration`: contribution `+0.014747`
- `lag_03__CT_B_site_active_infernos`: contribution `+0.003984`
- `lag_04__CT_flash_duration_sum`: contribution `+0.003614`
- `lag_03__T_B_site_active_infernos`: contribution `+0.002892`
- `lag_03__active_infernos_total`: contribution `+0.002865`

### tick `127788`, seconds `99.00`, LSTM delta `-0.1328`

Top all feature movements:
- `lag_13__CT1__flash_duration`: contribution `-0.011539`
- `lag_00__kill_diff_last_3s`: contribution `-0.005388`
- `lag_00__T_kills_last_3s`: contribution `-0.005218`
- `lag_15__CT2__duck_amount`: contribution `-0.004262`
- `lag_00__CT2__is_walking`: contribution `-0.003007`

Top utility-only movements:
- `lag_13__CT1__flash_duration`: contribution `-0.011539`
- `lag_02__CT1__flash_duration`: contribution `-0.002216`
- `lag_13__CT_flash_duration_sum`: contribution `-0.002207`

### tick `126732`, seconds `82.50`, LSTM delta `-0.1278`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.026133`
- `lag_00__CT5__flash_duration`: contribution `-0.006545`
- `lag_00__kill_diff_last_3s`: contribution `-0.005388`
- `lag_00__T_kills_last_3s`: contribution `-0.005218`
- `lag_01__CT_shots_fired_sum`: contribution `-0.003746`

Top utility-only movements:
- `lag_00__CT5__flash_duration`: contribution `-0.006545`
- `lag_06__CT5__flash_duration`: contribution `-0.002204`
- `lag_00__CT_flash_duration_sum`: contribution `-0.001649`

### tick `128204`, seconds `105.50`, LSTM delta `-0.1092`

Top all feature movements:
- `lag_00__T_bomb_zone_count`: contribution `-0.008057`
- `lag_00__CT_velocity_mean`: contribution `-0.004218`
- `lag_12__T5__has_bomb`: contribution `-0.003582`
- `lag_11__CT_place_ARCH`: contribution `-0.003338`
- `lag_14__T2__is_walking`: contribution `-0.002590`

Top utility-only movements:
- `lag_11__T_B_site_active_infernos`: contribution `-0.002133`
- `lag_15__CT1__flash_duration`: contribution `-0.001764`
