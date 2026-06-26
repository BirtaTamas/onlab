# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-lynn-vision-bo3-80tf5tBYONxHYQuFp0AoSQ/legacy-vs-lynn-vision-m3-nuke.csv`
- round_num: `15`

## Largest probability jumps

- tick `108670`, seconds `38.50`, LSTM `0.7990`, delta `-0.1415`
- tick `110078`, seconds `60.50`, LSTM `0.9575`, delta `+0.0578`
- tick `110046`, seconds `60.00`, LSTM `0.8997`, delta `+0.0527`
- tick `107774`, seconds `24.50`, LSTM `0.8931`, delta `+0.0436`
- tick `109982`, seconds `59.00`, LSTM `0.8233`, delta `+0.0401`
- tick `108702`, seconds `39.00`, LSTM `0.8377`, delta `+0.0387`
- tick `108990`, seconds `43.50`, LSTM `0.8006`, delta `+0.0381`
- tick `109086`, seconds `45.00`, LSTM `0.7656`, delta `-0.0338`
- tick `108894`, seconds `42.00`, LSTM `0.7626`, delta `-0.0311`
- tick `107934`, seconds `27.00`, LSTM `0.9510`, delta `+0.0295`

## Top 15 local ridge features

- `lag_00__T_place_CONTROL`: coefficient `-0.001150`, |coef| `0.001150`
- `lag_06__T_place_HUT`: coefficient `-0.001145`, |coef| `0.001145`
- `lag_09__CT_place_HUT`: coefficient `0.001045`, |coef| `0.001045`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001043`, |coef| `0.001043`
- `lag_13__CT_place_HEAVEN`: coefficient `-0.001024`, |coef| `0.001024`
- `lag_00__T1__duck_amount`: coefficient `-0.000931`, |coef| `0.000931`
- `lag_10__CT_place_HEAVEN`: coefficient `-0.000929`, |coef| `0.000929`
- `lag_12__CT_place_HEAVEN`: coefficient `-0.000903`, |coef| `0.000903`
- `lag_00__damage_diff_last_5s`: coefficient `0.000898`, |coef| `0.000898`
- `lag_00__kill_diff_last_3s`: coefficient `0.000892`, |coef| `0.000892`
- `lag_11__CT_place_HEAVEN`: coefficient `-0.000883`, |coef| `0.000883`
- `lag_13__CT_place_CATWALK`: coefficient `0.000880`, |coef| `0.000880`
- `lag_05__T_place_HUT`: coefficient `0.000829`, |coef| `0.000829`
- `lag_01__CT_shots_fired_sum`: coefficient `0.000785`, |coef| `0.000785`
- `lag_14__CT_place_HEAVEN`: coefficient `-0.000764`, |coef| `0.000764`

## Top 10 utility ridge features

- `lag_11__CT_B_site_active_smokes`: coefficient `-0.000588` (lowers CT win probability)
- `lag_11__CT_A_site_active_smokes`: coefficient `-0.000569` (lowers CT win probability)
- `lag_01__T_B_site_active_smokes`: coefficient `-0.000523` (lowers CT win probability)
- `lag_01__T_A_site_active_smokes`: coefficient `-0.000493` (lowers CT win probability)
- `lag_00__T_B_site_active_smokes`: coefficient `-0.000489` (lowers CT win probability)
- `lag_00__T_A_site_active_smokes`: coefficient `-0.000460` (lowers CT win probability)
- `lag_11__CT_active_smokes`: coefficient `-0.000412` (lowers CT win probability)
- `lag_09__CT_B_site_active_smokes`: coefficient `-0.000411` (lowers CT win probability)
- `lag_09__CT_A_site_active_smokes`: coefficient `-0.000399` (lowers CT win probability)
- `lag_01__T_active_smokes`: coefficient `-0.000377` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_CONTROL`: coefficient `-0.001150` (lowers CT win probability)
- `lag_06__T_place_HUT`: coefficient `-0.001145` (lowers CT win probability)
- `lag_09__CT_place_HUT`: coefficient `0.001045` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001043` (raises CT win probability)
- `lag_13__CT_place_HEAVEN`: coefficient `-0.001024` (lowers CT win probability)
- `lag_00__T1__duck_amount`: coefficient `-0.000931` (lowers CT win probability)
- `lag_10__CT_place_HEAVEN`: coefficient `-0.000929` (lowers CT win probability)
- `lag_12__CT_place_HEAVEN`: coefficient `-0.000903` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000898` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000892` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `108670`, seconds `38.50`, LSTM delta `-0.1415`

Top all feature movements:
- `lag_06__T_place_HUT`: contribution `-0.010676`
- `lag_09__CT_place_HUT`: contribution `-0.010189`
- `lag_05__T_place_HUT`: contribution `-0.007725`
- `lag_10__CT_place_HEAVEN`: contribution `-0.005014`
- `lag_12__CT_place_ADMIN`: contribution `-0.004727`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `110078`, seconds `60.50`, LSTM delta `+0.0578`

Top all feature movements:
- `lag_00__T_place_CONTROL`: contribution `+0.008169`
- `lag_14__CT_place_HEAVEN`: contribution `+0.004123`
- `lag_00__T1__duck_amount`: contribution `+0.003645`
- `lag_00__CT_shots_fired_sum`: contribution `+0.003622`
- `lag_01__CT_shots_fired_sum`: contribution `+0.002725`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `110046`, seconds `60.00`, LSTM delta `+0.0527`

Top all feature movements:
- `lag_13__CT_place_HEAVEN`: contribution `+0.005528`
- `lag_04__CT_place_HEAVEN`: contribution `+0.003921`
- `lag_00__CT_shots_fired_sum`: contribution `+0.003622`
- `lag_13__CT_place_CATWALK`: contribution `+0.003505`
- `lag_08__CT_place_CATWALK`: contribution `+0.002839`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `107774`, seconds `24.50`, LSTM delta `+0.0436`

Top all feature movements:
- `lag_01__CT_place_LOCKERROOM`: contribution `+0.006312`
- `lag_07__T_place_TROPHY`: contribution `+0.003296`
- `lag_07__T_place_CONTROL`: contribution `+0.003117`
- `lag_05__CT_place_MINI`: contribution `+0.002296`
- `lag_00__CT_kills_last_3s`: contribution `+0.002195`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `109982`, seconds `59.00`, LSTM delta `+0.0401`

Top all feature movements:
- `lag_11__CT_place_HEAVEN`: contribution `+0.004766`
- `lag_00__T1__duck_amount`: contribution `+0.003645`
- `lag_02__CT_place_HEAVEN`: contribution `+0.003630`
- `lag_11__CT_place_CATWALK`: contribution `+0.002549`
- `lag_06__CT_place_CATWALK`: contribution `+0.002316`

Top utility-only movements:
- `lag_08__CT_B_site_active_smokes`: contribution `+0.000596`
- `lag_08__CT_A_site_active_smokes`: contribution `+0.000561`
