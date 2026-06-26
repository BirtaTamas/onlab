# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-heroic-bo3-VpF2znQtwzecEgVsCr-4Wn/astralis-vs-heroic-m2-inferno.csv`
- round_num: `7`

## Largest probability jumps

- tick `40570`, seconds `33.50`, LSTM `0.3370`, delta `-0.2443`
- tick `41658`, seconds `50.50`, LSTM `0.7971`, delta `+0.2292`
- tick `41114`, seconds `42.00`, LSTM `0.6647`, delta `+0.1938`
- tick `39450`, seconds `16.00`, LSTM `0.6708`, delta `+0.1281`
- tick `40442`, seconds `31.50`, LSTM `0.5198`, delta `-0.1019`
- tick `40538`, seconds `33.00`, LSTM `0.5813`, delta `+0.0932`
- tick `40666`, seconds `35.00`, LSTM `0.3878`, delta `-0.0674`
- tick `40634`, seconds `34.50`, LSTM `0.4552`, delta `+0.0671`
- tick `39706`, seconds `20.00`, LSTM `0.7961`, delta `+0.0595`
- tick `40602`, seconds `34.00`, LSTM `0.3881`, delta `+0.0512`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003345`, |coef| `0.003345`
- `lag_00__T_shots_fired_sum`: coefficient `-0.002891`, |coef| `0.002891`
- `lag_00__CT_kills_last_3s`: coefficient `0.002860`, |coef| `0.002860`
- `lag_00__damage_diff_last_5s`: coefficient `0.002652`, |coef| `0.002652`
- `lag_01__CT1__is_scoped`: coefficient `0.002515`, |coef| `0.002515`
- `lag_00__CT_damage_last_5s`: coefficient `0.002246`, |coef| `0.002246`
- `lag_00__T5__has_bomb`: coefficient `-0.002041`, |coef| `0.002041`
- `lag_12__T_shots_fired_sum`: coefficient `-0.002033`, |coef| `0.002033`
- `lag_13__T_utility_damage_last_5s`: coefficient `0.002031`, |coef| `0.002031`
- `lag_00__CT3__duck_amount`: coefficient `0.001991`, |coef| `0.001991`
- `lag_05__CT_place_ARCH`: coefficient `-0.001938`, |coef| `0.001938`
- `lag_15__CT_place_ARCH`: coefficient `-0.001893`, |coef| `0.001893`
- `lag_15__CT_place_RUINS`: coefficient `0.001771`, |coef| `0.001771`
- `lag_01__T2__duck_amount`: coefficient `0.001712`, |coef| `0.001712`
- `lag_15__CT1__is_scoped`: coefficient `0.001635`, |coef| `0.001635`

## Top 10 utility ridge features

- `lag_13__T_utility_damage_last_5s`: coefficient `0.002031` (raises CT win probability)
- `lag_02__T_B_site_active_infernos`: coefficient `-0.001337` (lowers CT win probability)
- `lag_11__CT2__flash_duration`: coefficient `0.001267` (raises CT win probability)
- `lag_11__CT4__flash_duration`: coefficient `0.001230` (raises CT win probability)
- `lag_09__CT5__smoke`: coefficient `0.001206` (raises CT win probability)
- `lag_12__T_B_site_active_infernos`: coefficient `-0.001184` (lowers CT win probability)
- `lag_11__CT_flash_duration_sum`: coefficient `0.001118` (raises CT win probability)
- `lag_06__T_B_site_active_infernos`: coefficient `0.001115` (raises CT win probability)
- `lag_05__T_B_site_active_smokes`: coefficient `0.001115` (raises CT win probability)
- `lag_13__utility_damage_diff_last_5s`: coefficient `-0.001050` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003345` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.002891` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002860` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002652` (raises CT win probability)
- `lag_01__CT1__is_scoped`: coefficient `0.002515` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002246` (raises CT win probability)
- `lag_00__T5__has_bomb`: coefficient `-0.002041` (lowers CT win probability)
- `lag_12__T_shots_fired_sum`: coefficient `-0.002033` (lowers CT win probability)
- `lag_00__CT3__duck_amount`: coefficient `0.001991` (raises CT win probability)
- `lag_05__CT_place_ARCH`: coefficient `-0.001938` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `40570`, seconds `33.50`, LSTM delta `-0.2443`

Top all feature movements:
- `lag_13__T_utility_damage_last_5s`: contribution `-0.024065`
- `lag_00__T_shots_fired_sum`: contribution `-0.010838`
- `lag_11__CT2__flash_duration`: contribution `-0.008999`
- `lag_11__CT4__flash_duration`: contribution `-0.008695`
- `lag_00__kill_diff_last_3s`: contribution `-0.008052`

Top utility-only movements:
- `lag_13__T_utility_damage_last_5s`: contribution `-0.024065`
- `lag_11__CT2__flash_duration`: contribution `-0.008999`
- `lag_11__CT4__flash_duration`: contribution `-0.008695`
- `lag_13__utility_damage_diff_last_5s`: contribution `-0.007865`
- `lag_11__T3__flash_duration`: contribution `-0.007179`

### tick `41658`, seconds `50.50`, LSTM delta `+0.2292`

Top all feature movements:
- `lag_01__CT1__is_scoped`: contribution `+0.010773`
- `lag_12__T_shots_fired_sum`: contribution `+0.009146`
- `lag_00__CT_kills_last_3s`: contribution `+0.008256`
- `lag_00__kill_diff_last_3s`: contribution `+0.008052`
- `lag_05__CT_place_ARCH`: contribution `+0.007907`

Top utility-only movements:
- `lag_12__T_B_site_active_infernos`: contribution `+0.003348`

### tick `41114`, seconds `42.00`, LSTM delta `+0.1938`

Top all feature movements:
- `lag_01__CT1__is_scoped`: contribution `+0.010773`
- `lag_00__CT_kills_last_3s`: contribution `+0.008256`
- `lag_00__kill_diff_last_3s`: contribution `+0.008052`
- `lag_15__CT_place_ARCH`: contribution `+0.007724`
- `lag_00__CT3__duck_amount`: contribution `+0.007408`

Top utility-only movements:
- `lag_06__T_B_site_active_infernos`: contribution `+0.003154`

### tick `39450`, seconds `16.00`, LSTM delta `+0.1281`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.008256`
- `lag_00__kill_diff_last_3s`: contribution `+0.008052`
- `lag_00__CT1__is_scoped`: contribution `+0.006409`
- `lag_15__CT_place_RUINS`: contribution `-0.006186`
- `lag_00__damage_diff_last_5s`: contribution `+0.005982`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `40442`, seconds `31.50`, LSTM delta `-0.1019`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.010838`
- `lag_09__T_utility_damage_last_5s`: contribution `-0.009407`
- `lag_00__kill_diff_last_3s`: contribution `-0.008052`
- `lag_15__CT_place_ARCH`: contribution `-0.007724`
- `lag_12__utility_damage_diff_last_5s`: contribution `-0.005036`

Top utility-only movements:
- `lag_09__T_utility_damage_last_5s`: contribution `-0.009407`
- `lag_12__utility_damage_diff_last_5s`: contribution `-0.005036`
- `lag_07__T3__flash_duration`: contribution `-0.004661`
- `lag_12__CT_utility_damage_last_5s`: contribution `-0.003588`
- `lag_06__T_B_site_active_infernos`: contribution `-0.003154`
