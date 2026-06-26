# Local Round Explainability

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-the-mongolz-vs-natus-vincere-bo3-jwAddb1WR9PRMQexpSMSG8/the-mongolz-vs-natus-vincere-m2-ancient.csv`
- round_num: `18`

## Largest probability jumps

- tick `125772`, seconds `62.50`, LSTM `0.9503`, delta `+0.0554`
- tick `121804`, seconds `0.50`, LSTM `0.8293`, delta `+0.0378`
- tick `123212`, seconds `22.50`, LSTM `0.7438`, delta `-0.0359`
- tick `123308`, seconds `24.00`, LSTM `0.8082`, delta `+0.0312`
- tick `123276`, seconds `23.50`, LSTM `0.7770`, delta `+0.0295`
- tick `122124`, seconds `5.50`, LSTM `0.8019`, delta `-0.0272`
- tick `123340`, seconds `24.50`, LSTM `0.8340`, delta `+0.0258`
- tick `123596`, seconds `28.50`, LSTM `0.8365`, delta `-0.0237`
- tick `123148`, seconds `21.50`, LSTM `0.7727`, delta `+0.0197`
- tick `125420`, seconds `57.00`, LSTM `0.9014`, delta `+0.0197`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.000822`, |coef| `0.000822`
- `lag_00__T_he_last_5s`: coefficient `0.000814`, |coef| `0.000814`
- `lag_00__CT_walking_count`: coefficient `-0.000688`, |coef| `0.000688`
- `lag_01__CT_place_TSIDELOWER`: coefficient `0.000685`, |coef| `0.000685`
- `lag_10__T2__is_walking`: coefficient `0.000675`, |coef| `0.000675`
- `lag_00__damage_diff_last_5s`: coefficient `0.000662`, |coef| `0.000662`
- `lag_00__kill_diff_last_3s`: coefficient `0.000662`, |coef| `0.000662`
- `lag_00__CT_damage_last_5s`: coefficient `0.000661`, |coef| `0.000661`
- `lag_02__T2__is_walking`: coefficient `-0.000660`, |coef| `0.000660`
- `lag_10__CT1__is_walking`: coefficient `-0.000632`, |coef| `0.000632`
- `lag_00__CT_shots_fired_sum`: coefficient `0.000630`, |coef| `0.000630`
- `lag_00__CT1__is_scoped`: coefficient `-0.000616`, |coef| `0.000616`
- `lag_00__T4__is_walking`: coefficient `-0.000577`, |coef| `0.000577`
- `lag_03__T4__is_walking`: coefficient `-0.000566`, |coef| `0.000566`
- `lag_00__CT5__is_walking`: coefficient `-0.000563`, |coef| `0.000563`

## Top 10 utility ridge features

- `lag_00__T_he_last_5s`: coefficient `0.000814` (raises CT win probability)
- `lag_10__CT2__flash_duration`: coefficient `-0.000561` (lowers CT win probability)
- `lag_11__CT2__flash_duration`: coefficient `-0.000487` (lowers CT win probability)
- `lag_09__CT2__flash_duration`: coefficient `-0.000411` (lowers CT win probability)
- `lag_09__T4__flash_duration`: coefficient `-0.000397` (lowers CT win probability)
- `lag_03__T4__flash_duration`: coefficient `-0.000358` (lowers CT win probability)
- `lag_07__CT_B_site_active_smokes`: coefficient `-0.000327` (lowers CT win probability)
- `lag_04__CT_B_site_active_infernos`: coefficient `-0.000303` (lowers CT win probability)
- `lag_13__CT_A_site_active_infernos`: coefficient `-0.000295` (lowers CT win probability)
- `lag_13__CT_active_infernos`: coefficient `-0.000294` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.000822` (raises CT win probability)
- `lag_00__CT_walking_count`: coefficient `-0.000688` (lowers CT win probability)
- `lag_01__CT_place_TSIDELOWER`: coefficient `0.000685` (raises CT win probability)
- `lag_10__T2__is_walking`: coefficient `0.000675` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000662` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000662` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.000661` (raises CT win probability)
- `lag_02__T2__is_walking`: coefficient `-0.000660` (lowers CT win probability)
- `lag_10__CT1__is_walking`: coefficient `-0.000632` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.000630` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `125772`, seconds `62.50`, LSTM delta `+0.0554`

Top all feature movements:
- `lag_12__T_place_WATER`: contribution `+0.003203`
- `lag_00__CT1__is_scoped`: contribution `+0.002638`
- `lag_00__CT_kills_last_3s`: contribution `+0.002374`
- `lag_12__T_place_TSIDELOWER`: contribution `+0.001880`
- `lag_01__T_place_TSIDELOWER`: contribution `+0.001620`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `121804`, seconds `0.50`, LSTM delta `+0.0378`

Top all feature movements:
- `lag_00__T_he_last_5s`: contribution `+0.010624`
- `lag_01__CT_place_TSIDELOWER`: contribution `+0.009259`
- `lag_01__CT_closest_enemy_dist`: contribution `+0.000853`
- `lag_01__T_place_TSPAWN`: contribution `+0.000768`
- `lag_01__CT_place_HOUSE`: contribution `+0.000723`

Top utility-only movements:
- `lag_00__T_he_last_5s`: contribution `+0.010624`
- `lag_01__molly_inv_diff`: contribution `+0.000328`
- `lag_01__CT_smoke_inv`: contribution `+0.000241`
- `lag_01__T5__flash`: contribution `+0.000240`
- `lag_01__CT_utility_inv`: contribution `+0.000237`

### tick `123212`, seconds `22.50`, LSTM delta `-0.0359`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.004816`
- `lag_00__CT3__shots_fired`: contribution `-0.002623`
- `lag_04__T_place_WATER`: contribution `-0.002486`
- `lag_06__T_place_WATER`: contribution `-0.002356`
- `lag_12__T_place_TUNNEL`: contribution `-0.002348`

Top utility-only movements:
- `lag_15__T2__flash_duration`: contribution `-0.001096`

### tick `123308`, seconds `24.00`, LSTM delta `+0.0312`

Top all feature movements:
- `lag_10__CT2__flash_duration`: contribution `+0.003353`
- `lag_00__CT_shots_fired_sum`: contribution `+0.002189`
- `lag_01__T_place_TSIDELOWER`: contribution `+0.001620`
- `lag_06__CT3__duck_amount`: contribution `+0.001576`
- `lag_10__T2__is_walking`: contribution `+0.001551`

Top utility-only movements:
- `lag_10__CT2__flash_duration`: contribution `+0.003353`
- `lag_10__CT_flash_duration_sum`: contribution `+0.000707`

### tick `123276`, seconds `23.50`, LSTM delta `+0.0295`

Top all feature movements:
- `lag_09__CT2__flash_duration`: contribution `+0.002456`
- `lag_06__T_place_WATER`: contribution `+0.002356`
- `lag_00__CT_shots_fired_sum`: contribution `+0.002189`
- `lag_14__T_place_TUNNEL`: contribution `+0.001904`
- `lag_08__T_place_TUNNEL`: contribution `+0.001753`

Top utility-only movements:
- `lag_09__CT2__flash_duration`: contribution `+0.002456`
- `lag_08__CT_B_site_active_infernos`: contribution `+0.000917`
