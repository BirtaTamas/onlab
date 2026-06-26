# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-vitality-vs-gamerlegion-bo3-HfAhqHTEhpe_HlObeToa76/vitality-vs-gamerlegion-m1-overpass.csv`
- round_num: `2`

## Largest probability jumps

- tick `8089`, seconds `29.00`, LSTM `0.7627`, delta `+0.3332`
- tick `8185`, seconds `30.50`, LSTM `0.8697`, delta `+0.0627`
- tick `8121`, seconds `29.50`, LSTM `0.8108`, delta `+0.0481`
- tick `8057`, seconds `28.50`, LSTM `0.4295`, delta `-0.0434`
- tick `8217`, seconds `31.00`, LSTM `0.9126`, delta `+0.0428`
- tick `7993`, seconds `27.50`, LSTM `0.4894`, delta `-0.0416`
- tick `6937`, seconds `11.00`, LSTM `0.5846`, delta `-0.0381`
- tick `7833`, seconds `25.00`, LSTM `0.5098`, delta `-0.0307`
- tick `6489`, seconds `4.00`, LSTM `0.6179`, delta `-0.0259`
- tick `6329`, seconds `1.50`, LSTM `0.6017`, delta `+0.0240`

## Top 15 local ridge features

- `lag_08__T_place_CONSTRUCTION`: coefficient `0.002741`, |coef| `0.002741`
- `lag_00__T_place_CONSTRUCTION`: coefficient `-0.001486`, |coef| `0.001486`
- `lag_08__T_place_WATER`: coefficient `-0.001484`, |coef| `0.001484`
- `lag_14__CT_place_WATER`: coefficient `0.001271`, |coef| `0.001271`
- `lag_00__CT_place_BACKOFA`: coefficient `0.001263`, |coef| `0.001263`
- `lag_10__T_place_CONSTRUCTION`: coefficient `-0.001183`, |coef| `0.001183`
- `lag_04__T_place_CONSTRUCTION`: coefficient `-0.001129`, |coef| `0.001129`
- `lag_15__T_place_PIPE`: coefficient `-0.001072`, |coef| `0.001072`
- `lag_08__CT_place_WATER`: coefficient `0.001035`, |coef| `0.001035`
- `lag_03__CT_flashed_players`: coefficient `0.001021`, |coef| `0.001021`
- `lag_03__T_place_CONSTRUCTION`: coefficient `-0.000971`, |coef| `0.000971`
- `lag_01__T_place_CONSTRUCTION`: coefficient `-0.000962`, |coef| `0.000962`
- `lag_00__CT_kills_last_3s`: coefficient `0.000953`, |coef| `0.000953`
- `lag_11__T_place_PIPE`: coefficient `-0.000940`, |coef| `0.000940`
- `lag_03__T_place_WATER`: coefficient `-0.000931`, |coef| `0.000931`

## Top 10 utility ridge features

- `lag_03__T2__flash_duration`: coefficient `0.000916` (raises CT win probability)
- `lag_03__T_flash_duration_sum`: coefficient `0.000828` (raises CT win probability)
- `lag_03__T5__flash_duration`: coefficient `0.000805` (raises CT win probability)
- `lag_00__T2__flash_duration`: coefficient `-0.000804` (lowers CT win probability)
- `lag_01__T2__flash_duration`: coefficient `-0.000677` (lowers CT win probability)
- `lag_03__CT4__flash_duration`: coefficient `0.000595` (raises CT win probability)
- `lag_00__T_flash_duration_sum`: coefficient `-0.000588` (lowers CT win probability)
- `lag_01__T_flash_duration_sum`: coefficient `-0.000545` (lowers CT win probability)
- `lag_01__T5__flash_duration`: coefficient `-0.000524` (lowers CT win probability)
- `lag_03__CT_flash_duration_sum`: coefficient `0.000524` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_08__T_place_CONSTRUCTION`: coefficient `0.002741` (raises CT win probability)
- `lag_00__T_place_CONSTRUCTION`: coefficient `-0.001486` (lowers CT win probability)
- `lag_08__T_place_WATER`: coefficient `-0.001484` (lowers CT win probability)
- `lag_14__CT_place_WATER`: coefficient `0.001271` (raises CT win probability)
- `lag_00__CT_place_BACKOFA`: coefficient `0.001263` (raises CT win probability)
- `lag_10__T_place_CONSTRUCTION`: coefficient `-0.001183` (lowers CT win probability)
- `lag_04__T_place_CONSTRUCTION`: coefficient `-0.001129` (lowers CT win probability)
- `lag_15__T_place_PIPE`: coefficient `-0.001072` (lowers CT win probability)
- `lag_08__CT_place_WATER`: coefficient `0.001035` (raises CT win probability)
- `lag_03__CT_flashed_players`: coefficient `0.001021` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `8089`, seconds `29.00`, LSTM delta `+0.3332`

Top all feature movements:
- `lag_08__T_place_CONSTRUCTION`: contribution `+0.068127`
- `lag_00__T_place_CONSTRUCTION`: contribution `+0.018468`
- `lag_08__T_place_WATER`: contribution `+0.016941`
- `lag_15__T_place_PIPE`: contribution `+0.013691`
- `lag_00__CT_place_BACKOFA`: contribution `+0.012194`

Top utility-only movements:
- `lag_03__T2__flash_duration`: contribution `+0.005495`
- `lag_03__T_flash_duration_sum`: contribution `+0.004707`
- `lag_03__T5__flash_duration`: contribution `+0.004371`
- `lag_00__T2__flash_duration`: contribution `+0.003294`

### tick `8185`, seconds `30.50`, LSTM delta `+0.0627`

Top all feature movements:
- `lag_04__T_place_CONSTRUCTION`: contribution `+0.014031`
- `lag_00__CT_place_BACKOFA`: contribution `-0.012194`
- `lag_03__T_place_CONSTRUCTION`: contribution `+0.012069`
- `lag_09__T_place_CONSTRUCTION`: contribution `+0.008380`
- `lag_11__T_place_CONSTRUCTION`: contribution `+0.007608`

Top utility-only movements:
- `lag_03__T2__flash_duration`: contribution `-0.003755`
- `lag_03__T_flash_duration_sum`: contribution `-0.002235`
- `lag_06__T2__flash_duration`: contribution `+0.001769`
- `lag_06__T_flash_duration_sum`: contribution `+0.001542`

### tick `8121`, seconds `29.50`, LSTM delta `+0.0481`

Top all feature movements:
- `lag_09__T_place_CONSTRUCTION`: contribution `+0.016760`
- `lag_04__T_place_CONSTRUCTION`: contribution `+0.014031`
- `lag_01__T_place_CONSTRUCTION`: contribution `+0.011954`
- `lag_02__T_place_CONSTRUCTION`: contribution `-0.011469`
- `lag_07__T_place_CONSTRUCTION`: contribution `-0.008323`

Top utility-only movements:
- `lag_01__T2__flash_duration`: contribution `+0.002774`
- `lag_01__T_flash_duration_sum`: contribution `+0.001472`

### tick `8057`, seconds `28.50`, LSTM delta `-0.0434`

Top all feature movements:
- `lag_00__T_place_CONSTRUCTION`: contribution `+0.018468`
- `lag_07__T_place_CONSTRUCTION`: contribution `-0.016647`
- `lag_02__T_place_CONSTRUCTION`: contribution `-0.011469`
- `lag_05__T_place_CONSTRUCTION`: contribution `-0.009435`
- `lag_12__T_place_PIPE`: contribution `-0.007981`

Top utility-only movements:
- `lag_00__T2__flash_duration`: contribution `+0.001527`
- `lag_02__T5__flash_duration`: contribution `-0.001135`

### tick `8217`, seconds `31.00`, LSTM delta `+0.0428`

Top all feature movements:
- `lag_10__T_place_CONSTRUCTION`: contribution `-0.014706`
- `lag_04__T_place_CONSTRUCTION`: contribution `+0.014031`
- `lag_15__T_place_PIPE`: contribution `+0.013691`
- `lag_05__T_place_CONSTRUCTION`: contribution `+0.009435`
- `lag_07__T_place_CONSTRUCTION`: contribution `+0.008323`

Top utility-only movements:
- No utility movement among the top local contributors.
