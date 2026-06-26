# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-tyloo-vs-nrg-anubis-OygKONihup8TZ7k3ClDb0W/tyloo-vs-nrg-anubis.csv`
- round_num: `8`

## Largest probability jumps

- tick `67466`, seconds `83.50`, LSTM `0.6390`, delta `+0.2753`
- tick `68458`, seconds `99.00`, LSTM `0.8281`, delta `+0.2300`
- tick `67914`, seconds `90.50`, LSTM `0.7568`, delta `-0.1858`
- tick `67818`, seconds `89.00`, LSTM `0.9284`, delta `+0.1511`
- tick `63466`, seconds `21.00`, LSTM `0.6401`, delta `+0.1375`
- tick `68586`, seconds `101.00`, LSTM `0.9508`, delta `+0.1322`
- tick `66890`, seconds `74.50`, LSTM `0.4943`, delta `+0.1047`
- tick `63658`, seconds `24.00`, LSTM `0.5681`, delta `-0.0905`
- tick `66730`, seconds `72.00`, LSTM `0.4227`, delta `-0.0899`
- tick `68010`, seconds `92.00`, LSTM `0.6228`, delta `-0.0877`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.004581`, |coef| `0.004581`
- `lag_00__kill_diff_last_3s`: coefficient `0.004426`, |coef| `0.004426`
- `lag_00__T3__is_scoped`: coefficient `0.002834`, |coef| `0.002834`
- `lag_01__CT_shots_fired_sum`: coefficient `0.002646`, |coef| `0.002646`
- `lag_00__CT_place_TUNNEL`: coefficient `-0.002508`, |coef| `0.002508`
- `lag_03__CT_place_LOWERTUNNEL`: coefficient `-0.002457`, |coef| `0.002457`
- `lag_00__damage_diff_last_5s`: coefficient `0.002457`, |coef| `0.002457`
- `lag_00__CT_damage_last_5s`: coefficient `0.002440`, |coef| `0.002440`
- `lag_14__CT_place_CONNECTOR`: coefficient `-0.002358`, |coef| `0.002358`
- `lag_14__CT_kills_last_3s`: coefficient `-0.002287`, |coef| `0.002287`
- `lag_11__kill_diff_last_3s`: coefficient `0.002278`, |coef| `0.002278`
- `lag_08__CT1__flash_duration`: coefficient `-0.002238`, |coef| `0.002238`
- `lag_03__CT_place_CONNECTOR`: coefficient `-0.002151`, |coef| `0.002151`
- `lag_00__T_utility_damage_last_5s`: coefficient `0.002119`, |coef| `0.002119`
- `lag_04__CT_B_site_active_infernos`: coefficient `0.001962`, |coef| `0.001962`

## Top 10 utility ridge features

- `lag_08__CT1__flash_duration`: coefficient `-0.002238` (lowers CT win probability)
- `lag_00__T_utility_damage_last_5s`: coefficient `0.002119` (raises CT win probability)
- `lag_04__CT_B_site_active_infernos`: coefficient `0.001962` (raises CT win probability)
- `lag_05__T_B_site_active_infernos`: coefficient `-0.001796` (lowers CT win probability)
- `lag_10__T_B_site_active_infernos`: coefficient `0.001469` (raises CT win probability)
- `lag_08__CT5__molly`: coefficient `-0.001466` (lowers CT win probability)
- `lag_12__T3__molly`: coefficient `-0.001407` (lowers CT win probability)
- `lag_11__CT5__smoke`: coefficient `-0.001356` (lowers CT win probability)
- `lag_05__T_active_infernos`: coefficient `-0.001339` (lowers CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `-0.001334` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.004581` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.004426` (raises CT win probability)
- `lag_00__T3__is_scoped`: coefficient `0.002834` (raises CT win probability)
- `lag_01__CT_shots_fired_sum`: coefficient `0.002646` (raises CT win probability)
- `lag_00__CT_place_TUNNEL`: coefficient `-0.002508` (lowers CT win probability)
- `lag_03__CT_place_LOWERTUNNEL`: coefficient `-0.002457` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002457` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002440` (raises CT win probability)
- `lag_14__CT_place_CONNECTOR`: coefficient `-0.002358` (lowers CT win probability)
- `lag_14__CT_kills_last_3s`: coefficient `-0.002287` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `67466`, seconds `83.50`, LSTM delta `+0.2753`

Top all feature movements:
- `lag_00__T3__is_scoped`: contribution `+0.018181`
- `lag_03__CT_place_LOWERTUNNEL`: contribution `+0.018059`
- `lag_08__CT1__flash_duration`: contribution `+0.014258`
- `lag_00__CT_kills_last_3s`: contribution `+0.013226`
- `lag_00__kill_diff_last_3s`: contribution `+0.010653`

Top utility-only movements:
- `lag_08__CT1__flash_duration`: contribution `+0.014258`

### tick `68458`, seconds `99.00`, LSTM delta `+0.2300`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.013226`
- `lag_00__kill_diff_last_3s`: contribution `+0.010653`
- `lag_01__CT_shots_fired_sum`: contribution `+0.009192`
- `lag_14__CT_place_CONNECTOR`: contribution `+0.008432`
- `lag_03__CT_place_CONNECTOR`: contribution `+0.007690`

Top utility-only movements:
- `lag_00__T_utility_damage_last_5s`: contribution `+0.007260`
- `lag_04__CT_B_site_active_infernos`: contribution `+0.006741`
- `lag_05__T_B_site_active_infernos`: contribution `+0.005078`
- `lag_10__T_B_site_active_infernos`: contribution `+0.004154`

### tick `67914`, seconds `90.50`, LSTM delta `-0.1858`

Top all feature movements:
- `lag_00__T3__is_scoped`: contribution `-0.018181`
- `lag_00__kill_diff_last_3s`: contribution `-0.010653`
- `lag_13__CT_place_SNIPERSNEST`: contribution `-0.010451`
- `lag_14__T_place_CONNECTOR`: contribution `-0.008886`
- `lag_14__T3__is_scoped`: contribution `-0.006860`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `67818`, seconds `89.00`, LSTM delta `+0.1511`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.013226`
- `lag_00__kill_diff_last_3s`: contribution `+0.010653`
- `lag_10__CT_place_CANAL`: contribution `+0.009269`
- `lag_13__CT_flashed_players`: contribution `+0.006732`
- `lag_11__kill_diff_last_3s`: contribution `+0.005482`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `63466`, seconds `21.00`, LSTM delta `+0.1375`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.013226`
- `lag_00__kill_diff_last_3s`: contribution `+0.010653`
- `lag_01__CT_shots_fired_sum`: contribution `+0.009192`
- `lag_11__CT_place_BRIDGE`: contribution `+0.007607`
- `lag_05__T_place_BRIDGE`: contribution `+0.003833`

Top utility-only movements:
- `lag_10__CT_B_site_active_infernos`: contribution `+0.003532`
- `lag_10__active_infernos_total`: contribution `+0.001849`
