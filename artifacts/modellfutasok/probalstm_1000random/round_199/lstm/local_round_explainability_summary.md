# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-gamerlegion-vs-tyloo-bo3-0g9mXt3FIxC8XzjXNUjRL7/gamerlegion-vs-tyloo-m1-ancient-p3.csv`
- round_num: `11`

## Largest probability jumps

- tick `94791`, seconds `75.50`, LSTM `0.8377`, delta `+0.1652`
- tick `94759`, seconds `75.00`, LSTM `0.6725`, delta `+0.1443`
- tick `94855`, seconds `76.50`, LSTM `0.9431`, delta `+0.0853`
- tick `90535`, seconds `9.00`, LSTM `0.4977`, delta `-0.0319`
- tick `90023`, seconds `1.00`, LSTM `0.5432`, delta `-0.0305`
- tick `90951`, seconds `15.50`, LSTM `0.5119`, delta `-0.0248`
- tick `90471`, seconds `8.00`, LSTM `0.5396`, delta `+0.0240`
- tick `90695`, seconds `11.50`, LSTM `0.5228`, delta `+0.0231`
- tick `90247`, seconds `4.50`, LSTM `0.5548`, delta `+0.0204`
- tick `94823`, seconds `76.00`, LSTM `0.8579`, delta `+0.0202`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.002753`, |coef| `0.002753`
- `lag_00__kill_diff_last_3s`: coefficient `0.002295`, |coef| `0.002295`
- `lag_00__damage_diff_last_5s`: coefficient `0.001924`, |coef| `0.001924`
- `lag_00__CT_damage_last_5s`: coefficient `0.001862`, |coef| `0.001862`
- `lag_01__CT_kills_last_3s`: coefficient `0.001617`, |coef| `0.001617`
- `lag_01__T_place_SIDEENTRANCE`: coefficient `-0.001565`, |coef| `0.001565`
- `lag_07__CT_A_site_active_infernos`: coefficient `0.001522`, |coef| `0.001522`
- `lag_00__T_place_SIDEENTRANCE`: coefficient `-0.001511`, |coef| `0.001511`
- `lag_13__bomb_events_last_5s`: coefficient `-0.001483`, |coef| `0.001483`
- `lag_08__CT_A_site_active_infernos`: coefficient `0.001397`, |coef| `0.001397`
- `lag_01__kill_diff_last_3s`: coefficient `0.001360`, |coef| `0.001360`
- `lag_00__T4__alive`: coefficient `-0.001355`, |coef| `0.001355`
- `lag_00__T_burning_players`: coefficient `-0.001350`, |coef| `0.001350`
- `lag_00__T2__hp`: coefficient `-0.001236`, |coef| `0.001236`
- `lag_09__CT1__molly`: coefficient `-0.001234`, |coef| `0.001234`

## Top 10 utility ridge features

- `lag_07__CT_A_site_active_infernos`: coefficient `0.001522` (raises CT win probability)
- `lag_08__CT_A_site_active_infernos`: coefficient `0.001397` (raises CT win probability)
- `lag_09__CT1__molly`: coefficient `-0.001234` (lowers CT win probability)
- `lag_10__CT1__molly`: coefficient `-0.001186` (lowers CT win probability)
- `lag_10__T3__molly`: coefficient `-0.001153` (lowers CT win probability)
- `lag_11__T3__molly`: coefficient `-0.001065` (lowers CT win probability)
- `lag_06__CT_A_site_active_infernos`: coefficient `0.000894` (raises CT win probability)
- `lag_07__T_B_site_active_infernos`: coefficient `0.000879` (raises CT win probability)
- `lag_08__CT1__molly`: coefficient `-0.000839` (lowers CT win probability)
- `lag_07__CT_B_site_active_infernos`: coefficient `0.000811` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.002753` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002295` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001924` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001862` (raises CT win probability)
- `lag_01__CT_kills_last_3s`: coefficient `0.001617` (raises CT win probability)
- `lag_01__T_place_SIDEENTRANCE`: coefficient `-0.001565` (lowers CT win probability)
- `lag_00__T_place_SIDEENTRANCE`: coefficient `-0.001511` (lowers CT win probability)
- `lag_13__bomb_events_last_5s`: coefficient `-0.001483` (lowers CT win probability)
- `lag_01__kill_diff_last_3s`: coefficient `0.001360` (raises CT win probability)
- `lag_00__T4__alive`: coefficient `-0.001355` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `94791`, seconds `75.50`, LSTM delta `+0.1652`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.007949`
- `lag_01__T_place_SIDEENTRANCE`: contribution `+0.007638`
- `lag_00__kill_diff_last_3s`: contribution `+0.005525`
- `lag_08__CT_A_site_active_infernos`: contribution `+0.004930`
- `lag_01__CT_kills_last_3s`: contribution `+0.004668`

Top utility-only movements:
- `lag_08__CT_A_site_active_infernos`: contribution `+0.004930`
- `lag_10__CT1__molly`: contribution `+0.002953`

### tick `94759`, seconds `75.00`, LSTM delta `+0.1443`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.007949`
- `lag_00__T_place_SIDEENTRANCE`: contribution `+0.007373`
- `lag_00__kill_diff_last_3s`: contribution `+0.005525`
- `lag_07__CT_A_site_active_infernos`: contribution `+0.005370`
- `lag_00__damage_diff_last_5s`: contribution `+0.004340`

Top utility-only movements:
- `lag_07__CT_A_site_active_infernos`: contribution `+0.005370`
- `lag_09__CT1__molly`: contribution `+0.003072`
- `lag_07__CT_B_site_active_infernos`: contribution `+0.002786`
- `lag_10__T3__molly`: contribution `+0.002560`
- `lag_07__T_B_site_active_infernos`: contribution `+0.002484`

### tick `94855`, seconds `76.50`, LSTM delta `+0.0853`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.007949`
- `lag_00__kill_diff_last_3s`: contribution `+0.005525`
- `lag_00__damage_diff_last_5s`: contribution `+0.003342`
- `lag_03__CT4__duck_amount`: contribution `-0.003249`
- `lag_00__CT_damage_last_5s`: contribution `+0.003125`

Top utility-only movements:
- `lag_10__CT_A_site_active_infernos`: contribution `+0.001699`
- `lag_00__T5__flash`: contribution `+0.001542`

### tick `90535`, seconds `9.00`, LSTM delta `-0.0319`

Top all feature movements:
- `lag_09__CT_place_HOUSE`: contribution `-0.002611`
- `lag_10__CT_place_HOUSE`: contribution `-0.002311`
- `lag_08__CT_place_HOUSE`: contribution `-0.002146`
- `lag_14__T_place_TUNNEL`: contribution `-0.002108`
- `lag_04__CT_place_HOUSE`: contribution `-0.001880`

Top utility-only movements:
- `lag_00__T2__flash_duration`: contribution `-0.001054`

### tick `90023`, seconds `1.00`, LSTM delta `-0.0305`

Top all feature movements:
- `lag_02__CT_place_SIDEHALL`: contribution `-0.004249`
- `lag_00__CT_place_SIDEHALL`: contribution `+0.003270`
- `lag_02__CT_place_HOUSE`: contribution `+0.001634`
- `lag_02__T_money_sum`: contribution `-0.001576`
- `lag_02__T_start_balance_sum`: contribution `-0.001572`

Top utility-only movements:
- `lag_02__CT1__molly`: contribution `-0.000828`
- `lag_02__CT_molly_inv`: contribution `-0.000760`
- `lag_02__T3__molly`: contribution `-0.000542`
- `lag_02__T_flash_inv`: contribution `-0.000506`
- `lag_02__T_utility_inv`: contribution `-0.000505`
