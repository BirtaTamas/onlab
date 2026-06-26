# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-aurora-vs-faze-bo3-ZgdBOa3Yi0KCkwa_Ap1ef3/aurora-vs-faze-m2-train.csv`
- round_num: `10`

## Largest probability jumps

- tick `60847`, seconds `67.00`, LSTM `0.1284`, delta `-0.3094`
- tick `60911`, seconds `68.00`, LSTM `0.3493`, delta `+0.2160`
- tick `61103`, seconds `71.00`, LSTM `0.1567`, delta `-0.1557`
- tick `60719`, seconds `65.00`, LSTM `0.4241`, delta `-0.1438`
- tick `57295`, seconds `11.50`, LSTM `0.6347`, delta `+0.1412`
- tick `61039`, seconds `70.00`, LSTM `0.4113`, delta `+0.1324`
- tick `61135`, seconds `71.50`, LSTM `0.0259`, delta `-0.1308`
- tick `61071`, seconds `70.50`, LSTM `0.3124`, delta `-0.0990`
- tick `60783`, seconds `66.00`, LSTM `0.4122`, delta `-0.0443`
- tick `60943`, seconds `68.50`, LSTM `0.3065`, delta `-0.0429`

## Top 15 local ridge features

- `lag_06__CT_place_ELECTRICALBOX`: coefficient `-0.003128`, |coef| `0.003128`
- `lag_00__kill_diff_last_3s`: coefficient `0.002492`, |coef| `0.002492`
- `lag_00__T_kills_last_3s`: coefficient `-0.002417`, |coef| `0.002417`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001853`, |coef| `0.001853`
- `lag_12__CT_place_ELECTRICALBOX`: coefficient `0.001662`, |coef| `0.001662`
- `lag_03__CT_place_ENTRANCE`: coefficient `-0.001342`, |coef| `0.001342`
- `lag_04__CT_place_ELECTRICALBOX`: coefficient `0.001317`, |coef| `0.001317`
- `lag_08__T4__flash_duration`: coefficient `0.001317`, |coef| `0.001317`
- `lag_04__T_shots_fired_sum`: coefficient `-0.001313`, |coef| `0.001313`
- `lag_00__T_damage_last_5s`: coefficient `-0.001227`, |coef| `0.001227`
- `lag_04__CT1__flash_duration`: coefficient `0.001204`, |coef| `0.001204`
- `lag_00__CT_place_ENTRANCE`: coefficient `0.001178`, |coef| `0.001178`
- `lag_02__T1__duck_amount`: coefficient `-0.001143`, |coef| `0.001143`
- `lag_00__damage_diff_last_5s`: coefficient `0.001127`, |coef| `0.001127`
- `lag_02__T_place_ENTRANCE`: coefficient `-0.001072`, |coef| `0.001072`

## Top 10 utility ridge features

- `lag_08__T4__flash_duration`: coefficient `0.001317` (raises CT win probability)
- `lag_04__CT1__flash_duration`: coefficient `0.001204` (raises CT win probability)
- `lag_06__T_B_site_active_infernos`: coefficient `0.001046` (raises CT win probability)
- `lag_13__CT3__flash_duration`: coefficient `-0.001001` (lowers CT win probability)
- `lag_06__CT1__flash_duration`: coefficient `-0.000963` (lowers CT win probability)
- `lag_09__T_he_last_5s`: coefficient `-0.000943` (lowers CT win probability)
- `lag_08__CT_utility_damage_last_5s`: coefficient `-0.000877` (lowers CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `0.000865` (raises CT win probability)
- `lag_07__CT_utility_damage_last_5s`: coefficient `-0.000828` (lowers CT win probability)
- `lag_08__T_B_site_active_infernos`: coefficient `-0.000795` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_06__CT_place_ELECTRICALBOX`: coefficient `-0.003128` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002492` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002417` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001853` (lowers CT win probability)
- `lag_12__CT_place_ELECTRICALBOX`: coefficient `0.001662` (raises CT win probability)
- `lag_03__CT_place_ENTRANCE`: coefficient `-0.001342` (lowers CT win probability)
- `lag_04__CT_place_ELECTRICALBOX`: coefficient `0.001317` (raises CT win probability)
- `lag_04__T_shots_fired_sum`: coefficient `-0.001313` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001227` (lowers CT win probability)
- `lag_00__CT_place_ENTRANCE`: coefficient `0.001178` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `60847`, seconds `67.00`, LSTM delta `-0.3094`

Top all feature movements:
- `lag_06__CT_place_ELECTRICALBOX`: contribution `-0.036357`
- `lag_04__CT_place_ELECTRICALBOX`: contribution `-0.015308`
- `lag_03__CT_place_ENTRANCE`: contribution `-0.011909`
- `lag_00__CT_place_ENTRANCE`: contribution `-0.010450`
- `lag_04__CT1__flash_duration`: contribution `-0.008458`

Top utility-only movements:
- `lag_04__CT1__flash_duration`: contribution `-0.008458`
- `lag_08__T4__flash_duration`: contribution `-0.007092`
- `lag_06__T_B_site_active_infernos`: contribution `-0.005914`
- `lag_06__T_active_infernos`: contribution `-0.003211`

### tick `60911`, seconds `68.00`, LSTM delta `+0.2160`

Top all feature movements:
- `lag_06__CT_place_ELECTRICALBOX`: contribution `+0.036357`
- `lag_00__kill_diff_last_3s`: contribution `+0.011998`
- `lag_08__CT_place_ELECTRICALBOX`: contribution `+0.009027`
- `lag_00__T_kills_last_3s`: contribution `+0.007656`
- `lag_05__CT_place_ENTRANCE`: contribution `+0.007256`

Top utility-only movements:
- `lag_06__CT1__flash_duration`: contribution `+0.006767`
- `lag_01__CT_utility_damage_last_5s`: contribution `+0.004546`
- `lag_08__T_B_site_active_infernos`: contribution `+0.004494`
- `lag_10__T4__flash_duration`: contribution `+0.003733`
- `lag_01__utility_damage_diff_last_5s`: contribution `+0.002680`

### tick `61103`, seconds `71.00`, LSTM delta `-0.1557`

Top all feature movements:
- `lag_12__CT_place_ELECTRICALBOX`: contribution `-0.019324`
- `lag_00__kill_diff_last_3s`: contribution `-0.011998`
- `lag_14__CT_place_ELECTRICALBOX`: contribution `-0.009331`
- `lag_00__T_kills_last_3s`: contribution `-0.007656`
- `lag_11__CT_place_ENTRANCE`: contribution `-0.006652`

Top utility-only movements:
- `lag_07__CT_utility_damage_last_5s`: contribution `-0.005468`
- `lag_12__CT1__flash_duration`: contribution `-0.003815`
- `lag_14__T_B_site_active_infernos`: contribution `-0.003812`
- `lag_05__CT3__flash_duration`: contribution `-0.003263`
- `lag_07__utility_damage_diff_last_5s`: contribution `-0.003131`

### tick `60719`, seconds `65.00`, LSTM delta `-0.1438`

Top all feature movements:
- `lag_00__CT_place_ELECTRICALBOX`: contribution `-0.010122`
- `lag_00__T_shots_fired_sum`: contribution `-0.009726`
- `lag_00__T_kills_last_3s`: contribution `-0.007656`
- `lag_13__CT3__flash_duration`: contribution `-0.006286`
- `lag_00__CT1__flash_duration`: contribution `-0.006079`

Top utility-only movements:
- `lag_13__CT3__flash_duration`: contribution `-0.006286`
- `lag_00__CT1__flash_duration`: contribution `-0.006079`
- `lag_13__T4__flash_duration`: contribution `-0.004413`
- `lag_02__T_B_site_active_infernos`: contribution `-0.003165`
- `lag_04__T4__flash_duration`: contribution `-0.003001`

### tick `57295`, seconds `11.50`, LSTM delta `+0.1412`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.016674`
- `lag_09__T_he_last_5s`: contribution `+0.012303`
- `lag_15__CT_place_ENTRANCE`: contribution `+0.006712`
- `lag_00__kill_diff_last_3s`: contribution `+0.005999`
- `lag_07__CT_place_CONNECTOR`: contribution `+0.005830`

Top utility-only movements:
- `lag_09__T_he_last_5s`: contribution `+0.012303`
- `lag_00__T2__flash_duration`: contribution `+0.003532`
- `lag_03__CT2__flash_duration`: contribution `+0.002786`
