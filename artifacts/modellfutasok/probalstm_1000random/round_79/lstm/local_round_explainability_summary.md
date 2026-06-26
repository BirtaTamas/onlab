# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-nrg-vs-vitality-bo3-7fVRmFXct71SEzAlz8IKvC/nrg-vs-vitality-m2-dust2.csv`
- round_num: `17`

## Largest probability jumps

- tick `132562`, seconds `99.00`, LSTM `0.6319`, delta `+0.3227`
- tick `132690`, seconds `101.00`, LSTM `0.7406`, delta `+0.2631`
- tick `132594`, seconds `99.50`, LSTM `0.4104`, delta `-0.2214`
- tick `132530`, seconds `98.50`, LSTM `0.3092`, delta `+0.2078`
- tick `132754`, seconds `102.00`, LSTM `0.8936`, delta `+0.1231`
- tick `132370`, seconds `96.00`, LSTM `0.2097`, delta `-0.1016`
- tick `132626`, seconds `100.00`, LSTM `0.4871`, delta `+0.0766`
- tick `132434`, seconds `97.00`, LSTM `0.1265`, delta `-0.0630`
- tick `132306`, seconds `95.00`, LSTM `0.3579`, delta `-0.0551`
- tick `132338`, seconds `95.50`, LSTM `0.3112`, delta `-0.0466`

## Top 15 local ridge features

- `lag_00__T_place_HOLE`: coefficient `-0.002936`, |coef| `0.002936`
- `lag_02__T_place_HOLE`: coefficient `0.002833`, |coef| `0.002833`
- `lag_08__T_place_HOLE`: coefficient `0.002000`, |coef| `0.002000`
- `lag_02__T_place_BDOORS`: coefficient `-0.001948`, |coef| `0.001948`
- `lag_00__T_place_BDOORS`: coefficient `-0.001816`, |coef| `0.001816`
- `lag_10__T_place_HOLE`: coefficient `0.001533`, |coef| `0.001533`
- `lag_12__T_place_BDOORS`: coefficient `0.001403`, |coef| `0.001403`
- `lag_03__T_place_BDOORS`: coefficient `-0.001276`, |coef| `0.001276`
- `lag_03__T_place_HOLE`: coefficient `-0.001272`, |coef| `0.001272`
- `lag_06__T_place_HOLE`: coefficient `0.001171`, |coef| `0.001171`
- `lag_01__T_place_HOLE`: coefficient `0.001143`, |coef| `0.001143`
- `lag_05__T_place_BDOORS`: coefficient `0.001093`, |coef| `0.001093`
- `lag_00__CT_shots_fired_sum`: coefficient `0.000956`, |coef| `0.000956`
- `lag_01__T_place_BDOORS`: coefficient `-0.000906`, |coef| `0.000906`
- `lag_13__T_place_BDOORS`: coefficient `0.000905`, |coef| `0.000905`

## Top 10 utility ridge features

- `lag_02__T_active_infernos`: coefficient `-0.000443` (lowers CT win probability)
- `lag_12__T1__flash_duration`: coefficient `0.000433` (raises CT win probability)
- `lag_05__T1__flash_duration`: coefficient `-0.000423` (lowers CT win probability)
- `lag_06__T1__flash_duration`: coefficient `-0.000416` (lowers CT win probability)
- `lag_10__T1__flash_duration`: coefficient `-0.000411` (lowers CT win probability)
- `lag_04__CT4__flash_duration`: coefficient `-0.000409` (lowers CT win probability)
- `lag_04__T2__flash_duration`: coefficient `-0.000385` (lowers CT win probability)
- `lag_07__T1__flash_duration`: coefficient `-0.000380` (lowers CT win probability)
- `lag_00__T1__flash_duration`: coefficient `0.000373` (raises CT win probability)
- `lag_01__CT_active_infernos`: coefficient `0.000345` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_HOLE`: coefficient `-0.002936` (lowers CT win probability)
- `lag_02__T_place_HOLE`: coefficient `0.002833` (raises CT win probability)
- `lag_08__T_place_HOLE`: coefficient `0.002000` (raises CT win probability)
- `lag_02__T_place_BDOORS`: coefficient `-0.001948` (lowers CT win probability)
- `lag_00__T_place_BDOORS`: coefficient `-0.001816` (lowers CT win probability)
- `lag_10__T_place_HOLE`: coefficient `0.001533` (raises CT win probability)
- `lag_12__T_place_BDOORS`: coefficient `0.001403` (raises CT win probability)
- `lag_03__T_place_BDOORS`: coefficient `-0.001276` (lowers CT win probability)
- `lag_03__T_place_HOLE`: coefficient `-0.001272` (lowers CT win probability)
- `lag_06__T_place_HOLE`: coefficient `0.001171` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `132562`, seconds `99.00`, LSTM delta `+0.3227`

Top all feature movements:
- `lag_02__T_place_HOLE`: contribution `+0.146072`
- `lag_00__T_place_HOLE`: contribution `+0.075679`
- `lag_02__T_place_BDOORS`: contribution `+0.048734`
- `lag_01__T_place_HOLE`: contribution `-0.029455`
- `lag_04__T_place_HOLE`: contribution `+0.017627`

Top utility-only movements:
- `lag_13__T1__flash_duration`: contribution `+0.002169`

### tick `132690`, seconds `101.00`, LSTM delta `+0.2631`

Top all feature movements:
- `lag_00__T_place_HOLE`: contribution `+0.075679`
- `lag_06__T_place_HOLE`: contribution `+0.060350`
- `lag_08__T_place_HOLE`: contribution `+0.051568`
- `lag_05__T_place_HOLE`: contribution `+0.023127`
- `lag_04__T_place_HOLE`: contribution `-0.017627`

Top utility-only movements:
- `lag_03__CT4__flash_duration`: contribution `+0.001352`
- `lag_03__T1__flash_duration`: contribution `+0.001342`

### tick `132594`, seconds `99.50`, LSTM delta `-0.2214`

Top all feature movements:
- `lag_02__T_place_HOLE`: contribution `-0.073036`
- `lag_03__T_place_HOLE`: contribution `-0.065582`
- `lag_03__T_place_BDOORS`: contribution `+0.031921`
- `lag_01__T_place_HOLE`: contribution `-0.029455`
- `lag_05__T_place_HOLE`: contribution `-0.023127`

Top utility-only movements:
- `lag_00__T1__flash_duration`: contribution `-0.001681`
- `lag_14__T1__flash_duration`: contribution `-0.001596`

### tick `132530`, seconds `98.50`, LSTM delta `+0.2078`

Top all feature movements:
- `lag_00__T_place_HOLE`: contribution `+0.075679`
- `lag_01__T_place_HOLE`: contribution `+0.058910`
- `lag_03__T_place_HOLE`: contribution `-0.032791`
- `lag_05__T_place_BDOORS`: contribution `+0.027343`
- `lag_01__T_place_BDOORS`: contribution `+0.022671`

Top utility-only movements:
- `lag_12__T1__flash_duration`: contribution `+0.002930`
- `lag_06__CT4__flash_duration`: contribution `+0.001210`

### tick `132754`, seconds `102.00`, LSTM delta `+0.1231`

Top all feature movements:
- `lag_08__T_place_HOLE`: contribution `+0.103135`
- `lag_02__T_place_HOLE`: contribution `-0.073036`
- `lag_10__T_place_HOLE`: contribution `+0.039517`
- `lag_12__T_place_BDOORS`: contribution `+0.035094`
- `lag_06__T_place_HOLE`: contribution `-0.030175`

Top utility-only movements:
- `lag_05__T1__flash_duration`: contribution `+0.001905`
- `lag_13__T4__flash_duration`: contribution `+0.000955`
