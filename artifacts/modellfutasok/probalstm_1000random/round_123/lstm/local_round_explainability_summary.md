# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-liquid-vs-furia-bo3-oYHD2J45okzf6eapD2F9CM/liquid-vs-furia-m1-mirage.csv`
- round_num: `12`

## Largest probability jumps

- tick `98737`, seconds `75.50`, LSTM `0.9088`, delta `+0.1954`
- tick `97233`, seconds `52.00`, LSTM `0.8508`, delta `+0.1651`
- tick `97809`, seconds `61.00`, LSTM `0.7156`, delta `+0.1330`
- tick `97713`, seconds `59.50`, LSTM `0.6278`, delta `-0.1252`
- tick `97073`, seconds `49.50`, LSTM `0.6538`, delta `+0.1047`
- tick `97425`, seconds `55.00`, LSTM `0.7936`, delta `-0.0689`
- tick `98609`, seconds `73.50`, LSTM `0.7148`, delta `-0.0575`
- tick `98001`, seconds `64.00`, LSTM `0.7425`, delta `-0.0367`
- tick `98065`, seconds `65.00`, LSTM `0.8061`, delta `+0.0364`
- tick `97937`, seconds `63.00`, LSTM `0.7675`, delta `+0.0355`

## Top 15 local ridge features

- `lag_00__T5__is_scoped`: coefficient `0.002241`, |coef| `0.002241`
- `lag_00__CT_kills_last_3s`: coefficient `0.002068`, |coef| `0.002068`
- `lag_00__kill_diff_last_3s`: coefficient `0.002002`, |coef| `0.002002`
- `lag_15__CT_place_SHOP`: coefficient `0.001653`, |coef| `0.001653`
- `lag_08__T1__flash_duration`: coefficient `0.001646`, |coef| `0.001646`
- `lag_04__T_bomb_zone_count`: coefficient `0.001459`, |coef| `0.001459`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001444`, |coef| `0.001444`
- `lag_01__T_place_TRUCK`: coefficient `-0.001421`, |coef| `0.001421`
- `lag_00__damage_diff_last_5s`: coefficient `0.001411`, |coef| `0.001411`
- `lag_00__CT_damage_last_5s`: coefficient `0.001348`, |coef| `0.001348`
- `lag_02__CT_shots_fired_sum`: coefficient `0.001243`, |coef| `0.001243`
- `lag_00__T_duck_amount_mean`: coefficient `0.001232`, |coef| `0.001232`
- `lag_11__T5__is_scoped`: coefficient `0.001230`, |coef| `0.001230`
- `lag_01__CT_shots_fired_sum`: coefficient `0.001201`, |coef| `0.001201`
- `lag_00__T_bomb_zone_count`: coefficient `-0.001191`, |coef| `0.001191`

## Top 10 utility ridge features

- `lag_08__T1__flash_duration`: coefficient `0.001646` (raises CT win probability)
- `lag_04__T1__flash_duration`: coefficient `-0.000828` (lowers CT win probability)
- `lag_02__T1__flash_duration`: coefficient `-0.000764` (lowers CT win probability)
- `lag_00__T_B_site_active_infernos`: coefficient `-0.000705` (lowers CT win probability)
- `lag_08__T_flash_duration_sum`: coefficient `0.000648` (raises CT win probability)
- `lag_14__T1__flash_duration`: coefficient `0.000615` (raises CT win probability)
- `lag_00__T2__molly`: coefficient `-0.000606` (lowers CT win probability)
- `lag_00__T5__flash`: coefficient `-0.000592` (lowers CT win probability)
- `lag_12__CT1__smoke`: coefficient `-0.000532` (lowers CT win probability)
- `lag_14__T_B_site_active_infernos`: coefficient `0.000532` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T5__is_scoped`: coefficient `0.002241` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002068` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002002` (raises CT win probability)
- `lag_15__CT_place_SHOP`: coefficient `0.001653` (raises CT win probability)
- `lag_04__T_bomb_zone_count`: coefficient `0.001459` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001444` (raises CT win probability)
- `lag_01__T_place_TRUCK`: coefficient `-0.001421` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001411` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001348` (raises CT win probability)
- `lag_02__CT_shots_fired_sum`: coefficient `0.001243` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `98737`, seconds `75.50`, LSTM delta `+0.1954`

Top all feature movements:
- `lag_08__T1__flash_duration`: contribution `+0.010798`
- `lag_00__T5__is_scoped`: contribution `+0.010689`
- `lag_04__T_bomb_zone_count`: contribution `+0.008496`
- `lag_15__CT_place_SHOP`: contribution `+0.008290`
- `lag_00__CT_shots_fired_sum`: contribution `+0.008026`

Top utility-only movements:
- `lag_08__T1__flash_duration`: contribution `+0.010798`

### tick `97233`, seconds `52.00`, LSTM delta `+0.1651`

Top all feature movements:
- `lag_00__T5__is_scoped`: contribution `+0.010689`
- `lag_00__CT_place_JUNGLE`: contribution `+0.007044`
- `lag_14__CT_place_JUNGLE`: contribution `+0.006359`
- `lag_00__CT_kills_last_3s`: contribution `+0.005971`
- `lag_00__kill_diff_last_3s`: contribution `+0.004820`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `97809`, seconds `61.00`, LSTM delta `+0.1330`

Top all feature movements:
- `lag_00__T_place_TRUCK`: contribution `+0.019293`
- `lag_04__T_place_TRUCK`: contribution `+0.011276`
- `lag_00__CT_kills_last_3s`: contribution `+0.005971`
- `lag_12__T5__is_scoped`: contribution `+0.005657`
- `lag_00__kill_diff_last_3s`: contribution `+0.004820`

Top utility-only movements:
- `lag_00__T3__flash`: contribution `+0.001427`

### tick `97713`, seconds `59.50`, LSTM delta `-0.1252`

Top all feature movements:
- `lag_01__T_place_TRUCK`: contribution `-0.024676`
- `lag_14__CT_shots_fired_sum`: contribution `-0.008371`
- `lag_14__CT3__shots_fired`: contribution `-0.006681`
- `lag_00__kill_diff_last_3s`: contribution `-0.004820`
- `lag_05__CT_place_SHOP`: contribution `-0.004813`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `97073`, seconds `49.50`, LSTM delta `+0.1047`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.005971`
- `lag_00__kill_diff_last_3s`: contribution `+0.004820`
- `lag_00__CT_shots_fired_sum`: contribution `+0.004013`
- `lag_00__T_place_CONNECTOR`: contribution `+0.003544`
- `lag_01__T_place_CONNECTOR`: contribution `+0.003080`

Top utility-only movements:
- No utility movement among the top local contributors.
