# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-heroic-vs-nrg-dust2-QDtqFlW1Z9UhZpBNOAavnd/heroic-vs-nrg-dust2.csv`
- round_num: `1`

## Largest probability jumps

- tick `8530`, seconds `28.50`, LSTM `0.5308`, delta `+0.2727`
- tick `7378`, seconds `10.50`, LSTM `0.2523`, delta `-0.2456`
- tick `10962`, seconds `66.50`, LSTM `0.0863`, delta `-0.2164`
- tick `9682`, seconds `46.50`, LSTM `0.1404`, delta `-0.1889`
- tick `8498`, seconds `28.00`, LSTM `0.2581`, delta `+0.1596`
- tick `10706`, seconds `62.50`, LSTM `0.2354`, delta `+0.1587`
- tick `8754`, seconds `32.00`, LSTM `0.4709`, delta `-0.1322`
- tick `8818`, seconds `33.00`, LSTM `0.3393`, delta `-0.1035`
- tick `8850`, seconds `33.50`, LSTM `0.2400`, delta `-0.0992`
- tick `10450`, seconds `58.50`, LSTM `0.1000`, delta `+0.0788`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005211`, |coef| `0.005211`
- `lag_00__damage_diff_last_5s`: coefficient `0.004989`, |coef| `0.004989`
- `lag_00__T_kills_last_3s`: coefficient `-0.003541`, |coef| `0.003541`
- `lag_04__CT_place_HOLE`: coefficient `0.003419`, |coef| `0.003419`
- `lag_15__T_place_ARAMP`: coefficient `-0.003186`, |coef| `0.003186`
- `lag_12__CT_place_ARAMP`: coefficient `-0.003133`, |coef| `0.003133`
- `lag_00__CT_kills_last_3s`: coefficient `0.003024`, |coef| `0.003024`
- `lag_00__T_damage_last_5s`: coefficient `-0.002950`, |coef| `0.002950`
- `lag_14__T_place_TUNNELSTAIRS`: coefficient `0.002625`, |coef| `0.002625`
- `lag_00__CT_place_ARAMP`: coefficient `0.002573`, |coef| `0.002573`
- `lag_06__CT_place_BDOORS`: coefficient `0.002338`, |coef| `0.002338`
- `lag_11__CT_place_LONGA`: coefficient `-0.002194`, |coef| `0.002194`
- `lag_00__CT_damage_last_5s`: coefficient `0.002138`, |coef| `0.002138`
- `lag_09__T_place_CATWALK`: coefficient `0.002075`, |coef| `0.002075`
- `lag_05__CT_place_BDOORS`: coefficient `0.002066`, |coef| `0.002066`

## Top 10 utility ridge features

- `lag_04__CT2__flash_duration`: coefficient `-0.001195` (lowers CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `-0.001150` (lowers CT win probability)
- `lag_05__CT2__flash_duration`: coefficient `-0.000898` (lowers CT win probability)
- `lag_02__T_A_site_active_infernos`: coefficient `-0.000840` (lowers CT win probability)
- `lag_01__T5__smoke`: coefficient `-0.000818` (lowers CT win probability)
- `lag_03__CT2__flash_duration`: coefficient `-0.000716` (lowers CT win probability)
- `lag_01__CT2__flash_duration`: coefficient `-0.000698` (lowers CT win probability)
- `lag_01__CT1__flash`: coefficient `0.000695` (raises CT win probability)
- `lag_02__CT2__flash_duration`: coefficient `-0.000679` (lowers CT win probability)
- `lag_06__CT2__flash_duration`: coefficient `-0.000674` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005211` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.004989` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003541` (lowers CT win probability)
- `lag_04__CT_place_HOLE`: coefficient `0.003419` (raises CT win probability)
- `lag_15__T_place_ARAMP`: coefficient `-0.003186` (lowers CT win probability)
- `lag_12__CT_place_ARAMP`: coefficient `-0.003133` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003024` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002950` (lowers CT win probability)
- `lag_14__T_place_TUNNELSTAIRS`: coefficient `0.002625` (raises CT win probability)
- `lag_00__CT_place_ARAMP`: coefficient `0.002573` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `8530`, seconds `28.50`, LSTM delta `+0.2727`

Top all feature movements:
- `lag_15__T_place_TUNNELSTAIRS`: contribution `+0.012625`
- `lag_00__kill_diff_last_3s`: contribution `+0.012543`
- `lag_00__damage_diff_last_5s`: contribution `+0.011254`
- `lag_07__T_place_TUNNELSTAIRS`: contribution `+0.010915`
- `lag_10__CT_place_EXTENDEDA`: contribution `+0.010441`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `7378`, seconds `10.50`, LSTM delta `-0.2456`

Top all feature movements:
- `lag_04__CT_place_HOLE`: contribution `-0.038166`
- `lag_05__CT_place_HOLE`: contribution `-0.021724`
- `lag_00__kill_diff_last_3s`: contribution `-0.012543`
- `lag_00__T_kills_last_3s`: contribution `-0.011218`
- `lag_05__CT_place_BDOORS`: contribution `-0.009936`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `10962`, seconds `66.50`, LSTM delta `-0.2164`

Top all feature movements:
- `lag_12__CT_place_ARAMP`: contribution `-0.019517`
- `lag_00__CT_place_ARAMP`: contribution `-0.016029`
- `lag_00__kill_diff_last_3s`: contribution `-0.012543`
- `lag_00__damage_diff_last_5s`: contribution `-0.011254`
- `lag_00__T_kills_last_3s`: contribution `-0.011218`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `9682`, seconds `46.50`, LSTM delta `-0.1889`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.012543`
- `lag_09__T_place_SHORTSTAIRS`: contribution `-0.012039`
- `lag_09__T_place_CATWALK`: contribution `-0.011947`
- `lag_06__CT_place_BDOORS`: contribution `-0.011248`
- `lag_00__T_kills_last_3s`: contribution `-0.011218`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `8498`, seconds `28.00`, LSTM delta `+0.1596`

Top all feature movements:
- `lag_14__T_place_TUNNELSTAIRS`: contribution `+0.018324`
- `lag_00__kill_diff_last_3s`: contribution `+0.012543`
- `lag_09__CT_place_EXTENDEDA`: contribution `+0.011576`
- `lag_00__damage_diff_last_5s`: contribution `+0.011254`
- `lag_14__CT_place_EXTENDEDA`: contribution `+0.010090`

Top utility-only movements:
- No utility movement among the top local contributors.
