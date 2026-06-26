# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-virtuspro-vs-legacy-ancient-7ivruObh5LTTVaCYe9h-YO/virtus-pro-vs-legacy-ancient.csv`
- round_num: `16`

## Largest probability jumps

- tick `142838`, seconds `96.00`, LSTM `0.1096`, delta `-0.3364`
- tick `142198`, seconds `86.00`, LSTM `0.2055`, delta `-0.3187`
- tick `142806`, seconds `95.50`, LSTM `0.4460`, delta `+0.2958`
- tick `142262`, seconds `87.00`, LSTM `0.3017`, delta `+0.1831`
- tick `143222`, seconds `102.00`, LSTM `0.4454`, delta `+0.1224`
- tick `142582`, seconds `92.00`, LSTM `0.1599`, delta `-0.0982`
- tick `142230`, seconds `86.50`, LSTM `0.1186`, delta `-0.0869`
- tick `139702`, seconds `47.00`, LSTM `0.6023`, delta `+0.0546`
- tick `142294`, seconds `87.50`, LSTM `0.2486`, delta `-0.0531`
- tick `143030`, seconds `99.00`, LSTM `0.1866`, delta `+0.0529`

## Top 15 local ridge features

- `lag_06__T_flashes_last_5s`: coefficient `-0.007642`, |coef| `0.007642`
- `lag_00__CT_shots_fired_sum`: coefficient `0.006120`, |coef| `0.006120`
- `lag_00__CT2__shots_fired`: coefficient `0.005593`, |coef| `0.005593`
- `lag_00__kill_diff_last_3s`: coefficient `0.005051`, |coef| `0.005051`
- `lag_08__T_flashes_last_5s`: coefficient `0.004778`, |coef| `0.004778`
- `lag_00__T_kills_last_3s`: coefficient `-0.004096`, |coef| `0.004096`
- `lag_13__CT_place_SIDEHALL`: coefficient `-0.003822`, |coef| `0.003822`
- `lag_05__T_place_ALLEY`: coefficient `-0.003636`, |coef| `0.003636`
- `lag_06__T_place_ALLEY`: coefficient `-0.003493`, |coef| `0.003493`
- `lag_05__T_place_HOUSE`: coefficient `0.003374`, |coef| `0.003374`
- `lag_00__damage_diff_last_5s`: coefficient `0.003368`, |coef| `0.003368`
- `lag_15__T_flashes_last_5s`: coefficient `-0.003351`, |coef| `0.003351`
- `lag_07__CT_place_TSIDEUPPER`: coefficient `-0.003246`, |coef| `0.003246`
- `lag_07__T_place_SIDEENTRANCE`: coefficient `-0.003012`, |coef| `0.003012`
- `lag_01__T_shots_fired_sum`: coefficient `0.002670`, |coef| `0.002670`

## Top 10 utility ridge features

- `lag_06__T_flashes_last_5s`: coefficient `-0.007642` (lowers CT win probability)
- `lag_08__T_flashes_last_5s`: coefficient `0.004778` (raises CT win probability)
- `lag_15__T_flashes_last_5s`: coefficient `-0.003351` (lowers CT win probability)
- `lag_06__T4__flash`: coefficient `-0.001339` (lowers CT win probability)
- `lag_15__T_A_site_active_smokes`: coefficient `-0.001246` (lowers CT win probability)
- `lag_15__T_B_site_active_smokes`: coefficient `-0.001193` (lowers CT win probability)
- `lag_13__T_flashes_last_5s`: coefficient `-0.000951` (lowers CT win probability)
- `lag_07__T_flashes_last_5s`: coefficient `-0.000860` (lowers CT win probability)
- `lag_15__T_active_smokes`: coefficient `-0.000809` (lowers CT win probability)
- `lag_10__T_flashes_last_5s`: coefficient `0.000778` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.006120` (raises CT win probability)
- `lag_00__CT2__shots_fired`: coefficient `0.005593` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.005051` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.004096` (lowers CT win probability)
- `lag_13__CT_place_SIDEHALL`: coefficient `-0.003822` (lowers CT win probability)
- `lag_05__T_place_ALLEY`: coefficient `-0.003636` (lowers CT win probability)
- `lag_06__T_place_ALLEY`: coefficient `-0.003493` (lowers CT win probability)
- `lag_05__T_place_HOUSE`: coefficient `0.003374` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003368` (raises CT win probability)
- `lag_07__CT_place_TSIDEUPPER`: coefficient `-0.003246` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `142838`, seconds `96.00`, LSTM delta `-0.3364`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.059525`
- `lag_00__CT2__shots_fired`: contribution `-0.038923`
- `lag_08__CT_place_TSIDEUPPER`: contribution `-0.016276`
- `lag_07__T_place_SIDEENTRANCE`: contribution `-0.014697`
- `lag_00__T_kills_last_3s`: contribution `-0.012977`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `142198`, seconds `86.00`, LSTM delta `-0.3187`

Top all feature movements:
- `lag_06__T_flashes_last_5s`: contribution `-0.069241`
- `lag_13__CT_place_SIDEHALL`: contribution `-0.016348`
- `lag_05__T_place_ALLEY`: contribution `-0.015405`
- `lag_05__T_place_HOUSE`: contribution `-0.014836`
- `lag_06__T_place_ALLEY`: contribution `-0.014799`

Top utility-only movements:
- `lag_06__T_flashes_last_5s`: contribution `-0.069241`

### tick `142806`, seconds `95.50`, LSTM delta `+0.2958`

Top all feature movements:
- `lag_15__T_flashes_last_5s`: contribution `+0.030363`
- `lag_00__CT_shots_fired_sum`: contribution `+0.029762`
- `lag_07__CT_place_TSIDEUPPER`: contribution `+0.024400`
- `lag_00__CT2__shots_fired`: contribution `+0.019462`
- `lag_00__kill_diff_last_3s`: contribution `+0.012157`

Top utility-only movements:
- `lag_15__T_flashes_last_5s`: contribution `+0.030363`

### tick `142262`, seconds `87.00`, LSTM delta `+0.1831`

Top all feature movements:
- `lag_08__T_flashes_last_5s`: contribution `+0.043295`
- `lag_00__CT_shots_fired_sum`: contribution `+0.021259`
- `lag_06__T_place_ALLEY`: contribution `-0.014799`
- `lag_00__CT2__shots_fired`: contribution `+0.013901`
- `lag_00__kill_diff_last_3s`: contribution `+0.012157`

Top utility-only movements:
- `lag_08__T_flashes_last_5s`: contribution `+0.043295`

### tick `143222`, seconds `102.00`, LSTM delta `+0.1224`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.021259`
- `lag_07__T_place_SIDEENTRANCE`: contribution `+0.014697`
- `lag_12__CT_shots_fired_sum`: contribution `+0.010766`
- `lag_04__CT_duck_amount_mean`: contribution `+0.010538`
- `lag_04__CT3__duck_amount`: contribution `+0.005954`

Top utility-only movements:
- No utility movement among the top local contributors.
