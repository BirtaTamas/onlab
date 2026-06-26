# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-lynn-vision-bo3-80tf5tBYONxHYQuFp0AoSQ/legacy-vs-lynn-vision-m3-nuke.csv`
- round_num: `13`

## Largest probability jumps

- tick `99087`, seconds `23.50`, LSTM `0.8272`, delta `+0.2080`
- tick `100431`, seconds `44.50`, LSTM `0.6445`, delta `-0.1974`
- tick `100015`, seconds `38.00`, LSTM `0.6367`, delta `-0.1940`
- tick `99183`, seconds `25.00`, LSTM `0.6423`, delta `-0.1235`
- tick `100079`, seconds `39.00`, LSTM `0.7514`, delta `+0.1005`
- tick `99439`, seconds `29.00`, LSTM `0.7424`, delta `+0.0942`
- tick `99023`, seconds `22.50`, LSTM `0.6027`, delta `+0.0867`
- tick `98415`, seconds `13.00`, LSTM `0.4056`, delta `-0.0586`
- tick `98479`, seconds `14.00`, LSTM `0.4475`, delta `+0.0586`
- tick `100271`, seconds `42.00`, LSTM `0.7406`, delta `-0.0535`

## Top 15 local ridge features

- `lag_00__CT_place_SECRET`: coefficient `0.002936`, |coef| `0.002936`
- `lag_00__kill_diff_last_3s`: coefficient `0.002611`, |coef| `0.002611`
- `lag_04__CT_place_SECRET`: coefficient `-0.002388`, |coef| `0.002388`
- `lag_00__T_kills_last_3s`: coefficient `-0.001997`, |coef| `0.001997`
- `lag_06__CT_place_SECRET`: coefficient `0.001868`, |coef| `0.001868`
- `lag_15__CT_place_SECRET`: coefficient `0.001854`, |coef| `0.001854`
- `lag_05__CT_place_DECON`: coefficient `-0.001846`, |coef| `0.001846`
- `lag_13__CT_place_SECRET`: coefficient `0.001733`, |coef| `0.001733`
- `lag_00__damage_diff_last_5s`: coefficient `0.001627`, |coef| `0.001627`
- `lag_02__CT1__duck_amount`: coefficient `-0.001615`, |coef| `0.001615`
- `lag_03__CT_place_RAMP`: coefficient `0.001550`, |coef| `0.001550`
- `lag_13__T2__is_walking`: coefficient `0.001529`, |coef| `0.001529`
- `lag_15__CT_place_TROPHY`: coefficient `0.001458`, |coef| `0.001458`
- `lag_07__CT_place_RAMP`: coefficient `0.001454`, |coef| `0.001454`
- `lag_13__CT_place_OBSERVATION`: coefficient `-0.001444`, |coef| `0.001444`

## Top 10 utility ridge features

- `lag_10__T_B_site_active_smokes`: coefficient `0.000673` (raises CT win probability)
- `lag_10__T_A_site_active_smokes`: coefficient `0.000631` (raises CT win probability)
- `lag_10__T_active_smokes`: coefficient `0.000447` (raises CT win probability)
- `lag_14__T_B_site_active_smokes`: coefficient `0.000352` (raises CT win probability)
- `lag_14__T_A_site_active_smokes`: coefficient `0.000329` (raises CT win probability)
- `lag_12__T_B_site_active_smokes`: coefficient `-0.000327` (lowers CT win probability)
- `lag_12__T_A_site_active_smokes`: coefficient `-0.000308` (lowers CT win probability)
- `lag_10__active_smokes_total`: coefficient `0.000268` (raises CT win probability)
- `lag_06__T_B_site_active_smokes`: coefficient `0.000243` (raises CT win probability)
- `lag_12__T_active_smokes`: coefficient `-0.000241` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_SECRET`: coefficient `0.002936` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002611` (raises CT win probability)
- `lag_04__CT_place_SECRET`: coefficient `-0.002388` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001997` (lowers CT win probability)
- `lag_06__CT_place_SECRET`: coefficient `0.001868` (raises CT win probability)
- `lag_15__CT_place_SECRET`: coefficient `0.001854` (raises CT win probability)
- `lag_05__CT_place_DECON`: coefficient `-0.001846` (lowers CT win probability)
- `lag_13__CT_place_SECRET`: coefficient `0.001733` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001627` (raises CT win probability)
- `lag_02__CT1__duck_amount`: coefficient `-0.001615` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `99087`, seconds `23.50`, LSTM delta `+0.2080`

Top all feature movements:
- `lag_08__CT_place_VENDING`: contribution `+0.021113`
- `lag_01__CT_place_TROPHY`: contribution `+0.017935`
- `lag_04__CT_place_TROPHY`: contribution `+0.016933`
- `lag_07__CT_place_VENDING`: contribution `+0.016570`
- `lag_04__CT_place_VENDING`: contribution `+0.012870`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `100431`, seconds `44.50`, LSTM delta `-0.1974`

Top all feature movements:
- `lag_05__CT_place_DECON`: contribution `-0.029355`
- `lag_13__CT_place_OBSERVATION`: contribution `-0.025142`
- `lag_00__CT_place_DECON`: contribution `-0.022340`
- `lag_11__CT_place_OBSERVATION`: contribution `-0.018260`
- `lag_13__CT_place_SECRET`: contribution `-0.017836`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `100015`, seconds `38.00`, LSTM delta `-0.1940`

Top all feature movements:
- `lag_00__CT_place_SECRET`: contribution `-0.030226`
- `lag_00__CT_place_OBSERVATION`: contribution `-0.024720`
- `lag_04__CT_place_SECRET`: contribution `-0.024582`
- `lag_00__T_kills_last_3s`: contribution `-0.006327`
- `lag_00__kill_diff_last_3s`: contribution `-0.006286`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `99183`, seconds `25.00`, LSTM delta `-0.1235`

Top all feature movements:
- `lag_02__CT_place_OBSERVATION`: contribution `-0.024819`
- `lag_08__CT_place_VENDING`: contribution `-0.021113`
- `lag_06__CT_place_SECRET`: contribution `+0.019229`
- `lag_04__CT_place_TROPHY`: contribution `-0.016933`
- `lag_07__CT_place_VENDING`: contribution `-0.016570`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `100079`, seconds `39.00`, LSTM delta `+0.1005`

Top all feature movements:
- `lag_02__CT_place_OBSERVATION`: contribution `+0.024819`
- `lag_00__CT_place_OBSERVATION`: contribution `-0.024720`
- `lag_06__CT_place_SECRET`: contribution `+0.019229`
- `lag_02__CT_place_SECRET`: contribution `+0.012112`
- `lag_00__kill_diff_last_3s`: contribution `+0.006286`

Top utility-only movements:
- No utility movement among the top local contributors.
