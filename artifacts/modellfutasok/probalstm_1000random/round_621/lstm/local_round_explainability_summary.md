# Local Round Explainability

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-flyquest-vs-nomads-bo3-rjDbNQ6hoJ50qwkbItjOHm/flyquest-vs-nomads-m2-mirage.csv`
- round_num: `9`

## Largest probability jumps

- tick `60981`, seconds `60.50`, LSTM `0.2426`, delta `-0.2746`
- tick `60373`, seconds `51.00`, LSTM `0.5273`, delta `-0.1943`
- tick `59381`, seconds `35.50`, LSTM `0.7019`, delta `+0.1053`
- tick `61045`, seconds `61.50`, LSTM `0.0816`, delta `-0.0982`
- tick `58453`, seconds `21.00`, LSTM `0.4619`, delta `-0.0827`
- tick `58421`, seconds `20.50`, LSTM `0.5447`, delta `-0.0764`
- tick `58933`, seconds `28.50`, LSTM `0.5374`, delta `+0.0679`
- tick `63349`, seconds `97.50`, LSTM `0.0155`, delta `-0.0640`
- tick `61013`, seconds `61.00`, LSTM `0.1798`, delta `-0.0628`
- tick `58741`, seconds `25.50`, LSTM `0.5110`, delta `+0.0529`

## Top 15 local ridge features

- `lag_04__T_place_STAIRS`: coefficient `-0.004536`, |coef| `0.004536`
- `lag_00__T_kills_last_3s`: coefficient `-0.002475`, |coef| `0.002475`
- `lag_00__kill_diff_last_3s`: coefficient `0.002400`, |coef| `0.002400`
- `lag_06__T_place_STAIRS`: coefficient `-0.002269`, |coef| `0.002269`
- `lag_12__CT_shots_fired_sum`: coefficient `0.002150`, |coef| `0.002150`
- `lag_00__CT_place_JUNGLE`: coefficient `0.002098`, |coef| `0.002098`
- `lag_00__damage_diff_last_5s`: coefficient `0.002092`, |coef| `0.002092`
- `lag_06__CT_place_JUNGLE`: coefficient `0.002080`, |coef| `0.002080`
- `lag_00__T_damage_last_5s`: coefficient `-0.001906`, |coef| `0.001906`
- `lag_11__CT_A_site_active_infernos`: coefficient `0.001814`, |coef| `0.001814`
- `lag_01__CT_place_TRUCK`: coefficient `-0.001775`, |coef| `0.001775`
- `lag_01__CT_place_JUNGLE`: coefficient `0.001771`, |coef| `0.001771`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001751`, |coef| `0.001751`
- `lag_05__T_place_STAIRS`: coefficient `-0.001651`, |coef| `0.001651`
- `lag_00__CT_place_CONNECTOR`: coefficient `0.001643`, |coef| `0.001643`

## Top 10 utility ridge features

- `lag_11__CT_A_site_active_infernos`: coefficient `0.001814` (raises CT win probability)
- `lag_09__CT_B_site_active_smokes`: coefficient `0.001117` (raises CT win probability)
- `lag_00__CT4__flash`: coefficient `0.001069` (raises CT win probability)
- `lag_09__CT_A_site_active_smokes`: coefficient `0.000998` (raises CT win probability)
- `lag_00__CT3__flash`: coefficient `0.000990` (raises CT win probability)
- `lag_11__T3__smoke`: coefficient `0.000893` (raises CT win probability)
- `lag_11__CT_active_infernos`: coefficient `0.000867` (raises CT win probability)
- `lag_05__CT2__smoke`: coefficient `0.000860` (raises CT win probability)
- `lag_09__CT_active_smokes`: coefficient `0.000833` (raises CT win probability)
- `lag_00__CT_flash_inv`: coefficient `0.000829` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_04__T_place_STAIRS`: coefficient `-0.004536` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002475` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002400` (raises CT win probability)
- `lag_06__T_place_STAIRS`: coefficient `-0.002269` (lowers CT win probability)
- `lag_12__CT_shots_fired_sum`: coefficient `0.002150` (raises CT win probability)
- `lag_00__CT_place_JUNGLE`: coefficient `0.002098` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002092` (raises CT win probability)
- `lag_06__CT_place_JUNGLE`: coefficient `0.002080` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001906` (lowers CT win probability)
- `lag_01__CT_place_TRUCK`: coefficient `-0.001775` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `60981`, seconds `60.50`, LSTM delta `-0.2746`

Top all feature movements:
- `lag_04__T_place_STAIRS`: contribution `-0.086836`
- `lag_12__CT_shots_fired_sum`: contribution `-0.016427`
- `lag_00__CT_place_JUNGLE`: contribution `-0.013463`
- `lag_01__CT_place_TRUCK`: contribution `-0.011448`
- `lag_12__CT3__shots_fired`: contribution `-0.009027`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `60373`, seconds `51.00`, LSTM delta `-0.1943`

Top all feature movements:
- `lag_06__CT_place_JUNGLE`: contribution `-0.013347`
- `lag_01__CT_place_JUNGLE`: contribution `-0.011361`
- `lag_00__T_kills_last_3s`: contribution `-0.007842`
- `lag_11__CT_A_site_active_infernos`: contribution `-0.006402`
- `lag_00__CT_place_CONNECTOR`: contribution `-0.005876`

Top utility-only movements:
- `lag_11__CT_A_site_active_infernos`: contribution `-0.006402`

### tick `59381`, seconds `35.50`, LSTM delta `+0.1053`

Top all feature movements:
- `lag_05__T_place_SCAFFOLDING`: contribution `+0.035593`
- `lag_02__T_place_SCAFFOLDING`: contribution `+0.021938`
- `lag_11__CT_A_site_active_infernos`: contribution `+0.006402`
- `lag_00__kill_diff_last_3s`: contribution `+0.005778`
- `lag_00__T_damage_last_5s`: contribution `-0.003703`

Top utility-only movements:
- `lag_11__CT_A_site_active_infernos`: contribution `+0.006402`
- `lag_11__CT_active_infernos`: contribution `+0.001998`

### tick `61045`, seconds `61.50`, LSTM delta `-0.0982`

Top all feature movements:
- `lag_06__T_place_STAIRS`: contribution `-0.043429`
- `lag_00__T_place_STAIRS`: contribution `-0.023744`
- `lag_02__CT_place_JUNGLE`: contribution `-0.007855`
- `lag_14__CT_shots_fired_sum`: contribution `+0.004699`
- `lag_03__CT_place_TRUCK`: contribution `-0.003564`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `58453`, seconds `21.00`, LSTM delta `-0.0827`

Top all feature movements:
- `lag_12__CT_place_SCAFFOLDING`: contribution `-0.020450`
- `lag_00__T_shots_fired_sum`: contribution `-0.005251`
- `lag_04__CT_place_JUNGLE`: contribution `-0.005019`
- `lag_12__T4__flash_duration`: contribution `-0.004273`
- `lag_05__T3__flash_duration`: contribution `-0.003998`

Top utility-only movements:
- `lag_12__T4__flash_duration`: contribution `-0.004273`
- `lag_05__T3__flash_duration`: contribution `-0.003998`
- `lag_09__T5__flash_duration`: contribution `-0.002025`
- `lag_12__T_flash_duration_sum`: contribution `-0.001521`
