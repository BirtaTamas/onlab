# Local Round Explainability

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-lynn-vision-vs-tyloo-bo3-tXRL8tbpb2Kb27VbUgWfK1/lynn-vision-vs-tyloo-m2-ancient.csv`
- round_num: `15`

## Largest probability jumps

- tick `154080`, seconds `76.50`, LSTM `0.1232`, delta `-0.3382`
- tick `153536`, seconds `68.00`, LSTM `0.3878`, delta `-0.1368`
- tick `154176`, seconds `78.00`, LSTM `0.0125`, delta `-0.1093`
- tick `153632`, seconds `69.50`, LSTM `0.4283`, delta `+0.0748`
- tick `153568`, seconds `68.50`, LSTM `0.3410`, delta `-0.0468`
- tick `150240`, seconds `16.50`, LSTM `0.6489`, delta `-0.0349`
- tick `154016`, seconds `75.50`, LSTM `0.4518`, delta `-0.0333`
- tick `152800`, seconds `56.50`, LSTM `0.5696`, delta `-0.0320`
- tick `154144`, seconds `77.50`, LSTM `0.1218`, delta `-0.0264`
- tick `154112`, seconds `77.00`, LSTM `0.1481`, delta `+0.0250`

## Top 15 local ridge features

- `lag_14__T_utility_damage_last_5s`: coefficient `0.003941`, |coef| `0.003941`
- `lag_00__T_kills_last_3s`: coefficient `-0.002834`, |coef| `0.002834`
- `lag_14__utility_damage_diff_last_5s`: coefficient `-0.002496`, |coef| `0.002496`
- `lag_01__T_place_RAMP`: coefficient `-0.002183`, |coef| `0.002183`
- `lag_00__T_damage_last_5s`: coefficient `-0.002178`, |coef| `0.002178`
- `lag_03__T3__flash_duration`: coefficient `-0.001953`, |coef| `0.001953`
- `lag_00__kill_diff_last_3s`: coefficient `0.001921`, |coef| `0.001921`
- `lag_10__T_B_site_active_infernos`: coefficient `0.001817`, |coef| `0.001817`
- `lag_00__CT4__alive`: coefficient `0.001781`, |coef| `0.001781`
- `lag_00__CT4__hp`: coefficient `0.001754`, |coef| `0.001754`
- `lag_00__T1__shots_fired`: coefficient `-0.001708`, |coef| `0.001708`
- `lag_08__T2__duck_amount`: coefficient `0.001697`, |coef| `0.001697`
- `lag_00__T3__flash_duration`: coefficient `0.001676`, |coef| `0.001676`
- `lag_01__T1__shots_fired`: coefficient `-0.001658`, |coef| `0.001658`
- `lag_00__CT_place_SIDEENTRANCE`: coefficient `0.001646`, |coef| `0.001646`

## Top 10 utility ridge features

- `lag_14__T_utility_damage_last_5s`: coefficient `0.003941` (raises CT win probability)
- `lag_14__utility_damage_diff_last_5s`: coefficient `-0.002496` (lowers CT win probability)
- `lag_03__T3__flash_duration`: coefficient `-0.001953` (lowers CT win probability)
- `lag_10__T_B_site_active_infernos`: coefficient `0.001817` (raises CT win probability)
- `lag_00__T3__flash_duration`: coefficient `0.001676` (raises CT win probability)
- `lag_03__CT_B_site_active_infernos`: coefficient `0.001491` (raises CT win probability)
- `lag_10__T_active_infernos`: coefficient `0.001374` (raises CT win probability)
- `lag_14__CT_B_site_active_infernos`: coefficient `-0.001372` (lowers CT win probability)
- `lag_00__T1__flash_duration`: coefficient `-0.001202` (lowers CT win probability)
- `lag_12__T_utility_damage_last_5s`: coefficient `0.001196` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.002834` (lowers CT win probability)
- `lag_01__T_place_RAMP`: coefficient `-0.002183` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002178` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001921` (raises CT win probability)
- `lag_00__CT4__alive`: coefficient `0.001781` (raises CT win probability)
- `lag_00__CT4__hp`: coefficient `0.001754` (raises CT win probability)
- `lag_00__T1__shots_fired`: coefficient `-0.001708` (lowers CT win probability)
- `lag_08__T2__duck_amount`: coefficient `0.001697` (raises CT win probability)
- `lag_01__T1__shots_fired`: coefficient `-0.001658` (lowers CT win probability)
- `lag_00__CT_place_SIDEENTRANCE`: coefficient `0.001646` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `154080`, seconds `76.50`, LSTM delta `-0.3382`

Top all feature movements:
- `lag_14__T_utility_damage_last_5s`: contribution `-0.028130`
- `lag_14__utility_damage_diff_last_5s`: contribution `-0.011269`
- `lag_03__T3__flash_duration`: contribution `-0.009064`
- `lag_00__T_kills_last_3s`: contribution `-0.008977`
- `lag_01__T_place_RAMP`: contribution `-0.007722`

Top utility-only movements:
- `lag_14__T_utility_damage_last_5s`: contribution `-0.028130`
- `lag_14__utility_damage_diff_last_5s`: contribution `-0.011269`
- `lag_03__T3__flash_duration`: contribution `-0.009064`
- `lag_10__T_B_site_active_infernos`: contribution `-0.005136`
- `lag_03__CT_B_site_active_infernos`: contribution `-0.005121`

### tick `153536`, seconds `68.00`, LSTM delta `-0.1368`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.008977`
- `lag_01__T_place_RAMP`: contribution `-0.007722`
- `lag_07__T_utility_damage_last_5s`: contribution `-0.007576`
- `lag_00__T1__shots_fired`: contribution `-0.006123`
- `lag_08__T_place_HOUSE`: contribution `-0.006123`

Top utility-only movements:
- `lag_07__T_utility_damage_last_5s`: contribution `-0.007576`
- `lag_07__utility_damage_diff_last_5s`: contribution `-0.003079`

### tick `154176`, seconds `78.00`, LSTM delta `-0.1093`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.017954`
- `lag_00__kill_diff_last_3s`: contribution `-0.009248`
- `lag_00__T_damage_last_5s`: contribution `-0.007937`
- `lag_08__T2__duck_amount`: contribution `-0.006489`
- `lag_00__damage_diff_last_5s`: contribution `-0.005215`

Top utility-only movements:
- `lag_03__T3__flash_duration`: contribution `+0.004768`

### tick `153632`, seconds `69.50`, LSTM delta `+0.0748`

Top all feature movements:
- `lag_10__T_utility_damage_last_5s`: contribution `+0.005884`
- `lag_02__T1__shots_fired`: contribution `+0.005838`
- `lag_02__T_shots_fired_sum`: contribution `+0.005347`
- `lag_00__T_utility_damage_last_5s`: contribution `+0.005285`
- `lag_07__CT4__is_walking`: contribution `+0.003338`

Top utility-only movements:
- `lag_10__T_utility_damage_last_5s`: contribution `+0.005884`
- `lag_00__T_utility_damage_last_5s`: contribution `+0.005285`
- `lag_00__CT_B_site_active_infernos`: contribution `+0.002862`
- `lag_10__utility_damage_diff_last_5s`: contribution `+0.002286`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.002200`

### tick `153568`, seconds `68.50`, LSTM delta `-0.0468`

Top all feature movements:
- `lag_00__T1__shots_fired`: contribution `+0.008164`
- `lag_01__T_place_RAMP`: contribution `+0.007722`
- `lag_00__T_shots_fired_sum`: contribution `+0.007223`
- `lag_01__T1__shots_fired`: contribution `-0.005946`
- `lag_13__T_place_RAMP`: contribution `-0.004320`

Top utility-only movements:
- `lag_08__T_utility_damage_last_5s`: contribution `-0.003910`
- `lag_00__CT3__molly`: contribution `-0.001605`
- `lag_08__utility_damage_diff_last_5s`: contribution `-0.001603`
