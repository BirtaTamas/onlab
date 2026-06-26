# Local Round Explainability

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-3dmax-vs-lynn-vision-bo3-0ZNMTlQ0ZfadRgwA0Ax5fN/3dmax-vs-lynn-vision-m2-anubis.csv`
- round_num: `14`

## Largest probability jumps

- tick `129593`, seconds `64.00`, LSTM `0.4298`, delta `-0.3203`
- tick `129721`, seconds `66.00`, LSTM `0.6276`, delta `+0.2777`
- tick `127769`, seconds `35.50`, LSTM `0.3490`, delta `-0.1979`
- tick `129273`, seconds `59.00`, LSTM `0.7727`, delta `+0.1816`
- tick `131129`, seconds `88.00`, LSTM `0.3789`, delta `-0.1453`
- tick `128057`, seconds `40.00`, LSTM `0.5143`, delta `+0.1380`
- tick `127097`, seconds `25.00`, LSTM `0.5246`, delta `-0.1366`
- tick `131097`, seconds `87.50`, LSTM `0.5242`, delta `+0.1149`
- tick `131385`, seconds `92.00`, LSTM `0.4465`, delta `+0.1023`
- tick `127737`, seconds `35.00`, LSTM `0.5469`, delta `-0.1015`

## Top 15 local ridge features

- `lag_00__CT_defusing_count`: coefficient `0.004104`, |coef| `0.004104`
- `lag_00__CT_shots_fired_sum`: coefficient `0.003159`, |coef| `0.003159`
- `lag_13__CT_place_BACKOFB`: coefficient `-0.003148`, |coef| `0.003148`
- `lag_06__CT_place_LOWERTUNNEL`: coefficient `0.003124`, |coef| `0.003124`
- `lag_00__kill_diff_last_3s`: coefficient `0.002883`, |coef| `0.002883`
- `lag_15__CT_place_BACKOFB`: coefficient `-0.002543`, |coef| `0.002543`
- `lag_00__damage_diff_last_5s`: coefficient `0.002535`, |coef| `0.002535`
- `lag_05__CT_place_LOWERTUNNEL`: coefficient `0.002364`, |coef| `0.002364`
- `lag_00__CT_place_TSTAIRS`: coefficient `0.002330`, |coef| `0.002330`
- `lag_08__CT_place_TSTAIRS`: coefficient `-0.002287`, |coef| `0.002287`
- `lag_14__CT_place_BACKOFB`: coefficient `-0.002268`, |coef| `0.002268`
- `lag_00__CT_kills_last_3s`: coefficient `0.002183`, |coef| `0.002183`
- `lag_00__T_shots_fired_sum`: coefficient `-0.002180`, |coef| `0.002180`
- `lag_02__CT_defusing_count`: coefficient `0.002124`, |coef| `0.002124`
- `lag_10__CT_place_MAIN`: coefficient `-0.001894`, |coef| `0.001894`

## Top 10 utility ridge features

- `lag_08__T5__flash_duration`: coefficient `0.000812` (raises CT win probability)
- `lag_00__CT4__utility_total`: coefficient `0.000549` (raises CT win probability)
- `lag_11__T1__flash_duration`: coefficient `0.000507` (raises CT win probability)
- `lag_00__CT4__smoke`: coefficient `0.000480` (raises CT win probability)
- `lag_00__CT4__flash`: coefficient `0.000451` (raises CT win probability)
- `lag_08__T_flash_duration_sum`: coefficient `0.000412` (raises CT win probability)
- `lag_11__CT_active_infernos`: coefficient `0.000349` (raises CT win probability)
- `lag_07__T1__flash_duration`: coefficient `0.000337` (raises CT win probability)
- `lag_04__CT2__molly`: coefficient `0.000329` (raises CT win probability)
- `lag_02__CT_utility_damage_last_5s`: coefficient `-0.000327` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_defusing_count`: coefficient `0.004104` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.003159` (raises CT win probability)
- `lag_13__CT_place_BACKOFB`: coefficient `-0.003148` (lowers CT win probability)
- `lag_06__CT_place_LOWERTUNNEL`: coefficient `0.003124` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002883` (raises CT win probability)
- `lag_15__CT_place_BACKOFB`: coefficient `-0.002543` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002535` (raises CT win probability)
- `lag_05__CT_place_LOWERTUNNEL`: coefficient `0.002364` (raises CT win probability)
- `lag_00__CT_place_TSTAIRS`: coefficient `0.002330` (raises CT win probability)
- `lag_08__CT_place_TSTAIRS`: coefficient `-0.002287` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `129593`, seconds `64.00`, LSTM delta `-0.3203`

Top all feature movements:
- `lag_00__CT_place_TSTAIRS`: contribution `-0.061485`
- `lag_08__CT_place_TSTAIRS`: contribution `-0.060358`
- `lag_04__CT_place_STREET`: contribution `-0.047542`
- `lag_03__CT_place_STREET`: contribution `-0.042410`
- `lag_00__T_shots_fired_sum`: contribution `-0.011440`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `129721`, seconds `66.00`, LSTM delta `+0.2777`

Top all feature movements:
- `lag_08__CT_place_TSTAIRS`: contribution `+0.060358`
- `lag_07__CT_place_STREET`: contribution `+0.048066`
- `lag_12__CT_place_TSTAIRS`: contribution `+0.034471`
- `lag_08__CT_place_STREET`: contribution `+0.026135`
- `lag_07__CT_place_TSTAIRS`: contribution `+0.011769`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `127769`, seconds `35.50`, LSTM delta `-0.1979`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.030726`
- `lag_00__T_shots_fired_sum`: contribution `-0.011440`
- `lag_01__T_shots_fired_sum`: contribution `-0.008197`
- `lag_00__kill_diff_last_3s`: contribution `-0.006939`
- `lag_00__T4__shots_fired`: contribution `-0.005997`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `129273`, seconds `59.00`, LSTM delta `+0.1816`

Top all feature movements:
- `lag_06__CT_place_LOWERTUNNEL`: contribution `+0.022966`
- `lag_00__CT_shots_fired_sum`: contribution `+0.017557`
- `lag_00__T_place_STREET`: contribution `+0.008019`
- `lag_00__kill_diff_last_3s`: contribution `+0.006939`
- `lag_00__CT_kills_last_3s`: contribution `+0.006302`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `131129`, seconds `88.00`, LSTM delta `-0.1453`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `-0.039779`
- `lag_13__CT_place_BACKOFB`: contribution `-0.017972`
- `lag_14__CT_place_BRICKS`: contribution `-0.014186`
- `lag_00__CT_velocity_mean`: contribution `-0.005000`
- `lag_12__T_duck_amount_mean`: contribution `-0.003897`

Top utility-only movements:
- No utility movement among the top local contributors.
