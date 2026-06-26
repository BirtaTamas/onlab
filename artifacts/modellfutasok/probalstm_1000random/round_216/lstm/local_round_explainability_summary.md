# Local Round Explainability

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-spirit-vs-furia-bo3-_zQK5XUu10iN1JLmPA8zQ4/spirit-vs-furia-m2-nuke.csv`
- round_num: `19`

## Largest probability jumps

- tick `169933`, seconds `68.50`, LSTM `0.1256`, delta `-0.1797`
- tick `167277`, seconds `27.00`, LSTM `0.3254`, delta `-0.1719`
- tick `168365`, seconds `44.00`, LSTM `0.2639`, delta `+0.1597`
- tick `171021`, seconds `85.50`, LSTM `0.0675`, delta `-0.1092`
- tick `168397`, seconds `44.50`, LSTM `0.3432`, delta `+0.0793`
- tick `167309`, seconds `27.50`, LSTM `0.2574`, delta `-0.0681`
- tick `168429`, seconds `45.00`, LSTM `0.3997`, delta `+0.0565`
- tick `169965`, seconds `69.00`, LSTM `0.0711`, delta `-0.0545`
- tick `170925`, seconds `84.00`, LSTM `0.1382`, delta `+0.0481`
- tick `170893`, seconds `83.50`, LSTM `0.0900`, delta `+0.0459`

## Top 15 local ridge features

- `lag_01__T_place_OBSERVATION`: coefficient `-0.002027`, |coef| `0.002027`
- `lag_14__CT_place_CONTROL`: coefficient `0.001743`, |coef| `0.001743`
- `lag_00__CT_place_DECON`: coefficient `0.001612`, |coef| `0.001612`
- `lag_00__kill_diff_last_3s`: coefficient `0.001501`, |coef| `0.001501`
- `lag_00__T_kills_last_3s`: coefficient `-0.001493`, |coef| `0.001493`
- `lag_11__T_place_RAMP`: coefficient `-0.001486`, |coef| `0.001486`
- `lag_14__CT_place_RAMP`: coefficient `-0.001480`, |coef| `0.001480`
- `lag_13__CT_place_CONTROL`: coefficient `0.001426`, |coef| `0.001426`
- `lag_00__T_place_OBSERVATION`: coefficient `-0.001424`, |coef| `0.001424`
- `lag_00__CT_place_HUT`: coefficient `-0.001420`, |coef| `0.001420`
- `lag_07__T_place_OBSERVATION`: coefficient `0.001407`, |coef| `0.001407`
- `lag_00__T_place_SQUEAKY`: coefficient `-0.001277`, |coef| `0.001277`
- `lag_12__T_place_RAMP`: coefficient `-0.001250`, |coef| `0.001250`
- `lag_02__T_place_OBSERVATION`: coefficient `-0.001221`, |coef| `0.001221`
- `lag_05__CT_place_VENTS`: coefficient `0.001157`, |coef| `0.001157`

## Top 10 utility ridge features

- `lag_15__CT2__flash_duration`: coefficient `0.001146` (raises CT win probability)
- `lag_15__T2__flash_duration`: coefficient `0.000897` (raises CT win probability)
- `lag_07__CT_B_site_active_infernos`: coefficient `0.000831` (raises CT win probability)
- `lag_14__T_utility_damage_last_5s`: coefficient `-0.000811` (lowers CT win probability)
- `lag_15__T_utility_damage_last_5s`: coefficient `-0.000797` (lowers CT win probability)
- `lag_05__T_utility_damage_last_5s`: coefficient `0.000795` (raises CT win probability)
- `lag_04__T_utility_damage_last_5s`: coefficient `0.000766` (raises CT win probability)
- `lag_15__CT_flash_duration_sum`: coefficient `0.000719` (raises CT win probability)
- `lag_06__utility_damage_diff_last_5s`: coefficient `-0.000695` (lowers CT win probability)
- `lag_10__T_A_site_active_infernos`: coefficient `0.000634` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_01__T_place_OBSERVATION`: coefficient `-0.002027` (lowers CT win probability)
- `lag_14__CT_place_CONTROL`: coefficient `0.001743` (raises CT win probability)
- `lag_00__CT_place_DECON`: coefficient `0.001612` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001501` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001493` (lowers CT win probability)
- `lag_11__T_place_RAMP`: coefficient `-0.001486` (lowers CT win probability)
- `lag_14__CT_place_RAMP`: coefficient `-0.001480` (lowers CT win probability)
- `lag_13__CT_place_CONTROL`: coefficient `0.001426` (raises CT win probability)
- `lag_00__T_place_OBSERVATION`: coefficient `-0.001424` (lowers CT win probability)
- `lag_00__CT_place_HUT`: coefficient `-0.001420` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `169933`, seconds `68.50`, LSTM delta `-0.1797`

Top all feature movements:
- `lag_00__CT_place_DECON`: contribution `-0.025640`
- `lag_14__CT_place_CONTROL`: contribution `-0.018093`
- `lag_01__CT_place_DECON`: contribution `-0.014099`
- `lag_05__CT_place_VENTS`: contribution `-0.009707`
- `lag_01__CT_place_ADMIN`: contribution `-0.006906`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `167277`, seconds `27.00`, LSTM delta `-0.1719`

Top all feature movements:
- `lag_01__T_place_TROPHY`: contribution `-0.007302`
- `lag_15__CT2__flash_duration`: contribution `-0.006797`
- `lag_08__CT_place_HELL`: contribution `-0.005796`
- `lag_11__T_place_RAMP`: contribution `-0.005256`
- `lag_11__T_place_CONTROL`: contribution `-0.005232`

Top utility-only movements:
- `lag_15__CT2__flash_duration`: contribution `-0.006797`
- `lag_15__T2__flash_duration`: contribution `-0.004315`
- `lag_07__CT_B_site_active_infernos`: contribution `-0.002856`
- `lag_14__T_utility_damage_last_5s`: contribution `-0.002433`
- `lag_15__CT_flash_duration_sum`: contribution `-0.001937`

### tick `168365`, seconds `44.00`, LSTM delta `+0.1597`

Top all feature movements:
- `lag_01__T_place_OBSERVATION`: contribution `+0.034316`
- `lag_07__T_place_OBSERVATION`: contribution `+0.023820`
- `lag_11__T_place_OBSERVATION`: contribution `+0.017415`
- `lag_00__T_place_SQUEAKY`: contribution `+0.007952`
- `lag_08__CT_place_VENTS`: contribution `+0.006733`

Top utility-only movements:
- `lag_10__T_A_site_active_infernos`: contribution `+0.001886`
- `lag_10__T_B_site_active_infernos`: contribution `+0.001712`

### tick `171021`, seconds `85.50`, LSTM delta `-0.1092`

Top all feature movements:
- `lag_07__T_place_OBSERVATION`: contribution `-0.023820`
- `lag_09__T_place_DECON`: contribution `-0.018221`
- `lag_12__T_place_DECON`: contribution `-0.012564`
- `lag_10__CT_place_ADMIN`: contribution `-0.004740`
- `lag_00__T_kills_last_3s`: contribution `-0.004730`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `168397`, seconds `44.50`, LSTM delta `+0.0793`

Top all feature movements:
- `lag_00__T_place_OBSERVATION`: contribution `+0.024119`
- `lag_02__T_place_OBSERVATION`: contribution `+0.020682`
- `lag_01__T_place_SQUEAKY`: contribution `+0.005501`
- `lag_12__T_place_OBSERVATION`: contribution `+0.003082`
- `lag_14__CT_place_MINI`: contribution `+0.002899`

Top utility-only movements:
- No utility movement among the top local contributors.
