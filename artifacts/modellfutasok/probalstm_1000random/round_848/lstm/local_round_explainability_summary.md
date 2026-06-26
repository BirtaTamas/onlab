# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-mouz-vs-pain-bo3-Ao8EIC0rxvFkpkJ5bGImFu/mouz-vs-pain-m3-dust2.csv`
- round_num: `11`

## Largest probability jumps

- tick `78825`, seconds `30.50`, LSTM `0.1835`, delta `-0.1784`
- tick `77417`, seconds `8.50`, LSTM `0.1428`, delta `-0.1514`
- tick `77673`, seconds `12.50`, LSTM `0.3008`, delta `+0.1426`
- tick `78505`, seconds `25.50`, LSTM `0.2752`, delta `-0.0591`
- tick `77897`, seconds `16.00`, LSTM `0.3952`, delta `+0.0533`
- tick `77385`, seconds `8.00`, LSTM `0.2942`, delta `-0.0496`
- tick `78345`, seconds `23.00`, LSTM `0.3609`, delta `-0.0474`
- tick `77737`, seconds `13.50`, LSTM `0.3896`, delta `+0.0463`
- tick `78857`, seconds `31.00`, LSTM `0.1404`, delta `-0.0430`
- tick `77705`, seconds `13.00`, LSTM `0.3433`, delta `+0.0425`

## Top 15 local ridge features

- `lag_06__CT_flashes_last_5s`: coefficient `-0.002338`, |coef| `0.002338`
- `lag_14__T_shots_fired_sum`: coefficient `0.002013`, |coef| `0.002013`
- `lag_00__T_kills_last_3s`: coefficient `-0.001523`, |coef| `0.001523`
- `lag_13__T_place_LOWERTUNNEL`: coefficient `0.001483`, |coef| `0.001483`
- `lag_07__CT_flashes_last_5s`: coefficient `-0.001451`, |coef| `0.001451`
- `lag_00__kill_diff_last_3s`: coefficient `0.001377`, |coef| `0.001377`
- `lag_03__T_he_last_5s`: coefficient `0.001320`, |coef| `0.001320`
- `lag_09__CT_place_EXTENDEDA`: coefficient `0.001220`, |coef| `0.001220`
- `lag_08__T_place_OUTSIDETUNNEL`: coefficient `-0.001175`, |coef| `0.001175`
- `lag_09__T_place_OUTSIDETUNNEL`: coefficient `-0.001140`, |coef| `0.001140`
- `lag_07__CT1__duck_amount`: coefficient `-0.001127`, |coef| `0.001127`
- `lag_10__T_place_OUTSIDETUNNEL`: coefficient `-0.001069`, |coef| `0.001069`
- `lag_00__CT3__alive`: coefficient `0.001052`, |coef| `0.001052`
- `lag_10__T_place_MIDDOORS`: coefficient `-0.001033`, |coef| `0.001033`
- `lag_14__T_he_last_5s`: coefficient `0.001010`, |coef| `0.001010`

## Top 10 utility ridge features

- `lag_06__CT_flashes_last_5s`: coefficient `-0.002338` (lowers CT win probability)
- `lag_07__CT_flashes_last_5s`: coefficient `-0.001451` (lowers CT win probability)
- `lag_03__T_he_last_5s`: coefficient `0.001320` (raises CT win probability)
- `lag_14__T_he_last_5s`: coefficient `0.001010` (raises CT win probability)
- `lag_12__CT1__molly`: coefficient `0.000925` (raises CT win probability)
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.000918` (lowers CT win probability)
- `lag_01__CT2__flash_duration`: coefficient `-0.000908` (lowers CT win probability)
- `lag_15__T_he_last_5s`: coefficient `0.000843` (raises CT win probability)
- `lag_08__CT_flashes_last_5s`: coefficient `-0.000803` (lowers CT win probability)
- `lag_09__CT3__smoke`: coefficient `0.000783` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_14__T_shots_fired_sum`: coefficient `0.002013` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001523` (lowers CT win probability)
- `lag_13__T_place_LOWERTUNNEL`: coefficient `0.001483` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001377` (raises CT win probability)
- `lag_09__CT_place_EXTENDEDA`: coefficient `0.001220` (raises CT win probability)
- `lag_08__T_place_OUTSIDETUNNEL`: coefficient `-0.001175` (lowers CT win probability)
- `lag_09__T_place_OUTSIDETUNNEL`: coefficient `-0.001140` (lowers CT win probability)
- `lag_07__CT1__duck_amount`: coefficient `-0.001127` (lowers CT win probability)
- `lag_10__T_place_OUTSIDETUNNEL`: coefficient `-0.001069` (lowers CT win probability)
- `lag_00__CT3__alive`: coefficient `0.001052` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `78825`, seconds `30.50`, LSTM delta `-0.1784`

Top all feature movements:
- `lag_06__CT_flashes_last_5s`: contribution `-0.025704`
- `lag_14__T_shots_fired_sum`: contribution `-0.019618`
- `lag_09__CT_place_EXTENDEDA`: contribution `-0.006850`
- `lag_13__T_place_LOWERTUNNEL`: contribution `-0.006412`
- `lag_00__T_kills_last_3s`: contribution `-0.004825`

Top utility-only movements:
- `lag_06__CT_flashes_last_5s`: contribution `-0.025704`
- `lag_00__T_utility_damage_last_5s`: contribution `-0.003145`
- `lag_12__CT1__molly`: contribution `-0.002301`
- `lag_09__CT3__smoke`: contribution `-0.001733`

### tick `77417`, seconds `8.50`, LSTM delta `-0.1514`

Top all feature movements:
- `lag_08__T_place_OUTSIDETUNNEL`: contribution `-0.005874`
- `lag_10__T_place_OUTSIDETUNNEL`: contribution `-0.005343`
- `lag_01__CT2__flash_duration`: contribution `-0.005122`
- `lag_00__T_kills_last_3s`: contribution `-0.004825`
- `lag_04__CT_place_BDOORS`: contribution `-0.004383`

Top utility-only movements:
- `lag_01__CT2__flash_duration`: contribution `-0.005122`
- `lag_01__CT4__flash_duration`: contribution `-0.003992`
- `lag_01__CT_flash_duration_sum`: contribution `-0.003643`
- `lag_00__CT4__flash_duration`: contribution `-0.002314`

### tick `77673`, seconds `12.50`, LSTM delta `+0.1426`

Top all feature movements:
- `lag_03__T_he_last_5s`: contribution `+0.017227`
- `lag_06__CT_place_HOLE`: contribution `+0.009679`
- `lag_07__CT_place_HOLE`: contribution `+0.006262`
- `lag_08__T_place_OUTSIDETUNNEL`: contribution `+0.005874`
- `lag_09__T_place_OUTSIDETUNNEL`: contribution `+0.005696`

Top utility-only movements:
- `lag_03__T_he_last_5s`: contribution `+0.017227`
- `lag_09__CT2__flash_duration`: contribution `+0.003319`
- `lag_08__CT4__flash_duration`: contribution `+0.002821`
- `lag_09__CT4__flash_duration`: contribution `+0.001968`

### tick `78505`, seconds `25.50`, LSTM delta `-0.0591`

Top all feature movements:
- `lag_08__T_place_TUNNELSTAIRS`: contribution `-0.005333`
- `lag_15__T_place_TUNNELSTAIRS`: contribution `-0.004707`
- `lag_04__T_shots_fired_sum`: contribution `-0.004234`
- `lag_05__T_shots_fired_sum`: contribution `-0.003850`
- `lag_15__CT_place_BDOORS`: contribution `-0.003132`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `77897`, seconds `16.00`, LSTM delta `+0.0533`

Top all feature movements:
- `lag_03__T_place_TUNNELSTAIRS`: contribution `+0.006153`
- `lag_14__CT_place_HOLE`: contribution `+0.006081`
- `lag_13__CT_place_HOLE`: contribution `+0.005054`
- `lag_00__T_he_last_5s`: contribution `+0.004663`
- `lag_02__CT1__duck_amount`: contribution `+0.002659`

Top utility-only movements:
- `lag_00__T_he_last_5s`: contribution `+0.004663`
- `lag_10__T_he_last_5s`: contribution `+0.002415`
- `lag_15__CT4__flash_duration`: contribution `+0.001454`
