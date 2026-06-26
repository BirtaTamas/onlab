# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-ence-bo3-avzZyhQ46OR9GyYE_6NeM7/astralis-vs-ence-m3-overpass.csv`
- round_num: `4`

## Largest probability jumps

- tick `16117`, seconds `60.00`, LSTM `0.1655`, delta `-0.2762`
- tick `13461`, seconds `18.50`, LSTM `0.3409`, delta `-0.2394`
- tick `16917`, seconds `72.50`, LSTM `0.0910`, delta `-0.2251`
- tick `16725`, seconds `69.50`, LSTM `0.0662`, delta `-0.2231`
- tick `16757`, seconds `70.00`, LSTM `0.2818`, delta `+0.2156`
- tick `16661`, seconds `68.50`, LSTM `0.3109`, delta `+0.2037`
- tick `13525`, seconds `19.50`, LSTM `0.4445`, delta `+0.1602`
- tick `16021`, seconds `58.50`, LSTM `0.3463`, delta `-0.0937`
- tick `16085`, seconds `59.50`, LSTM `0.4417`, delta `+0.0834`
- tick `14453`, seconds `34.00`, LSTM `0.4867`, delta `+0.0717`

## Top 15 local ridge features

- `lag_09__T_place_CONSTRUCTION`: coefficient `-0.002739`, |coef| `0.002739`
- `lag_00__T_shots_fired_sum`: coefficient `-0.002540`, |coef| `0.002540`
- `lag_00__CT_place_CANAL`: coefficient `0.002524`, |coef| `0.002524`
- `lag_00__kill_diff_last_3s`: coefficient `0.002108`, |coef| `0.002108`
- `lag_06__T_place_CONSTRUCTION`: coefficient `-0.001930`, |coef| `0.001930`
- `lag_01__T_place_PIPE`: coefficient `0.001844`, |coef| `0.001844`
- `lag_08__CT_place_STAIRS`: coefficient `0.001833`, |coef| `0.001833`
- `lag_00__damage_diff_last_5s`: coefficient `0.001789`, |coef| `0.001789`
- `lag_12__T_place_CONSTRUCTION`: coefficient `-0.001739`, |coef| `0.001739`
- `lag_06__T2__duck_amount`: coefficient `-0.001738`, |coef| `0.001738`
- `lag_03__T_place_WATER`: coefficient `-0.001664`, |coef| `0.001664`
- `lag_02__CT3__is_scoped`: coefficient `-0.001650`, |coef| `0.001650`
- `lag_00__CT_place_BACKOFA`: coefficient `0.001580`, |coef| `0.001580`
- `lag_09__T_place_WATER`: coefficient `0.001570`, |coef| `0.001570`
- `lag_02__CT_place_CANAL`: coefficient `-0.001488`, |coef| `0.001488`

## Top 10 utility ridge features

- `lag_13__T2__flash_duration`: coefficient `0.001184` (raises CT win probability)
- `lag_13__CT5__flash_duration`: coefficient `-0.001155` (lowers CT win probability)
- `lag_04__T2__flash_duration`: coefficient `-0.000981` (lowers CT win probability)
- `lag_08__CT_utility_damage_last_5s`: coefficient `-0.000902` (lowers CT win probability)
- `lag_02__CT_B_site_active_infernos`: coefficient `0.000884` (raises CT win probability)
- `lag_05__T2__flash_duration`: coefficient `-0.000873` (lowers CT win probability)
- `lag_15__T2__flash_duration`: coefficient `-0.000834` (lowers CT win probability)
- `lag_06__T4__flash_duration`: coefficient `0.000787` (raises CT win probability)
- `lag_09__T2__flash_duration`: coefficient `0.000757` (raises CT win probability)
- `lag_11__CT_utility_damage_last_5s`: coefficient `-0.000752` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_09__T_place_CONSTRUCTION`: coefficient `-0.002739` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.002540` (lowers CT win probability)
- `lag_00__CT_place_CANAL`: coefficient `0.002524` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002108` (raises CT win probability)
- `lag_06__T_place_CONSTRUCTION`: coefficient `-0.001930` (lowers CT win probability)
- `lag_01__T_place_PIPE`: coefficient `0.001844` (raises CT win probability)
- `lag_08__CT_place_STAIRS`: coefficient `0.001833` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001789` (raises CT win probability)
- `lag_12__T_place_CONSTRUCTION`: coefficient `-0.001739` (lowers CT win probability)
- `lag_06__T2__duck_amount`: coefficient `-0.001738` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `16117`, seconds `60.00`, LSTM delta `-0.2762`

Top all feature movements:
- `lag_09__T_place_CONSTRUCTION`: contribution `-0.068078`
- `lag_12__T_place_CONSTRUCTION`: contribution `-0.021620`
- `lag_09__T_place_WATER`: contribution `-0.017922`
- `lag_00__CT_place_CANAL`: contribution `-0.015343`
- `lag_00__T_shots_fired_sum`: contribution `-0.009523`

Top utility-only movements:
- `lag_05__CT5__flash_duration`: contribution `-0.004091`
- `lag_01__CT_utility_damage_last_5s`: contribution `-0.003255`

### tick `13461`, seconds `18.50`, LSTM delta `-0.2394`

Top all feature movements:
- `lag_01__T_place_PIPE`: contribution `-0.023559`
- `lag_14__T_place_PIPE`: contribution `-0.015619`
- `lag_00__CT_place_CANAL`: contribution `-0.015343`
- `lag_02__CT_place_CANAL`: contribution `-0.009042`
- `lag_01__CT_place_WATER`: contribution `-0.008162`

Top utility-only movements:
- `lag_06__T4__flash_duration`: contribution `-0.004444`
- `lag_02__CT_B_site_active_infernos`: contribution `-0.003037`

### tick `16917`, seconds `72.50`, LSTM delta `-0.2251`

Top all feature movements:
- `lag_12__T_place_CONSTRUCTION`: contribution `-0.021620`
- `lag_05__T_place_CONSTRUCTION`: contribution `-0.012304`
- `lag_07__T_place_CONSTRUCTION`: contribution `-0.011291`
- `lag_05__T_shots_fired_sum`: contribution `-0.009836`
- `lag_00__T_shots_fired_sum`: contribution `-0.009523`

Top utility-only movements:
- `lag_09__T2__flash_duration`: contribution `-0.004971`

### tick `16725`, seconds `69.50`, LSTM delta `-0.2231`

Top all feature movements:
- `lag_06__T_place_CONSTRUCTION`: contribution `-0.023984`
- `lag_00__T_shots_fired_sum`: contribution `-0.017142`
- `lag_08__CT_place_STAIRS`: contribution `-0.014270`
- `lag_11__CT_place_BACKOFA`: contribution `-0.010107`
- `lag_09__CT_place_BACKOFA`: contribution `-0.009291`

Top utility-only movements:
- `lag_15__T2__flash_duration`: contribution `-0.005475`
- `lag_03__T2__flash_duration`: contribution `-0.004335`

### tick `16757`, seconds `70.00`, LSTM delta `+0.2156`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.024761`
- `lag_02__T_place_CONSTRUCTION`: contribution `+0.017699`
- `lag_07__T_place_CONSTRUCTION`: contribution `+0.011291`
- `lag_11__CT_place_BACKOFA`: contribution `+0.010107`
- `lag_03__T_place_WATER`: contribution `+0.009500`

Top utility-only movements:
- `lag_04__T2__flash_duration`: contribution `+0.006437`
- `lag_11__CT_utility_damage_last_5s`: contribution `+0.003807`
