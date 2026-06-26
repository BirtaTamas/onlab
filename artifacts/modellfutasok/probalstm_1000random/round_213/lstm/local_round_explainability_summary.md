# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-vitality-vs-the-mongolz-bo3-JVS9HKMAkaZTRHkoiRSMP6/vitality-vs-the-mongolz-m1-mirage.csv`
- round_num: `11`

## Largest probability jumps

- tick `82268`, seconds `72.50`, LSTM `0.1077`, delta `-0.3411`
- tick `81820`, seconds `65.50`, LSTM `0.5422`, delta `-0.2059`
- tick `81788`, seconds `65.00`, LSTM `0.7482`, delta `-0.1533`
- tick `81724`, seconds `64.00`, LSTM `0.8762`, delta `+0.0964`
- tick `81916`, seconds `67.00`, LSTM `0.4381`, delta `-0.0470`
- tick `81852`, seconds `66.00`, LSTM `0.4972`, delta `-0.0450`
- tick `82300`, seconds `73.00`, LSTM `0.0656`, delta `-0.0421`
- tick `81404`, seconds `59.00`, LSTM `0.7461`, delta `-0.0347`
- tick `81980`, seconds `68.00`, LSTM `0.4501`, delta `+0.0338`
- tick `82108`, seconds `70.00`, LSTM `0.4879`, delta `+0.0336`

## Top 15 local ridge features

- `lag_01__T_place_STAIRS`: coefficient `0.002742`, |coef| `0.002742`
- `lag_06__T_place_STAIRS`: coefficient `-0.002094`, |coef| `0.002094`
- `lag_00__T_place_JUNGLE`: coefficient `-0.001876`, |coef| `0.001876`
- `lag_00__T_kills_last_3s`: coefficient `-0.001473`, |coef| `0.001473`
- `lag_00__kill_diff_last_3s`: coefficient `0.001245`, |coef| `0.001245`
- `lag_01__T_shots_fired_sum`: coefficient `-0.001229`, |coef| `0.001229`
- `lag_01__CT2__flash_duration`: coefficient `0.001220`, |coef| `0.001220`
- `lag_01__CT2__shots_fired`: coefficient `-0.001201`, |coef| `0.001201`
- `lag_13__T_shots_fired_sum`: coefficient `0.001108`, |coef| `0.001108`
- `lag_13__T2__flash_duration`: coefficient `0.001081`, |coef| `0.001081`
- `lag_15__CT2__flash_duration`: coefficient `0.001061`, |coef| `0.001061`
- `lag_11__CT_place_JUNGLE`: coefficient `0.001056`, |coef| `0.001056`
- `lag_06__T_place_CONNECTOR`: coefficient `-0.001045`, |coef| `0.001045`
- `lag_00__CT_place_JUNGLE`: coefficient `0.001034`, |coef| `0.001034`
- `lag_00__CT3__flash_duration`: coefficient `0.001034`, |coef| `0.001034`

## Top 10 utility ridge features

- `lag_01__CT2__flash_duration`: coefficient `0.001220` (raises CT win probability)
- `lag_13__T2__flash_duration`: coefficient `0.001081` (raises CT win probability)
- `lag_15__CT2__flash_duration`: coefficient `0.001061` (raises CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `0.001034` (raises CT win probability)
- `lag_09__CT3__flash_duration`: coefficient `-0.001031` (lowers CT win probability)
- `lag_14__CT3__flash_duration`: coefficient `0.001008` (raises CT win probability)
- `lag_06__CT3__flash_duration`: coefficient `0.000914` (raises CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `0.000878` (raises CT win probability)
- `lag_13__T4__flash_duration`: coefficient `0.000864` (raises CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `0.000861` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_01__T_place_STAIRS`: coefficient `0.002742` (raises CT win probability)
- `lag_06__T_place_STAIRS`: coefficient `-0.002094` (lowers CT win probability)
- `lag_00__T_place_JUNGLE`: coefficient `-0.001876` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001473` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001245` (raises CT win probability)
- `lag_01__T_shots_fired_sum`: coefficient `-0.001229` (lowers CT win probability)
- `lag_01__CT2__shots_fired`: coefficient `-0.001201` (lowers CT win probability)
- `lag_13__T_shots_fired_sum`: coefficient `0.001108` (raises CT win probability)
- `lag_11__CT_place_JUNGLE`: coefficient `0.001056` (raises CT win probability)
- `lag_06__T_place_CONNECTOR`: coefficient `-0.001045` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `82268`, seconds `72.50`, LSTM delta `-0.3411`

Top all feature movements:
- `lag_01__T_place_STAIRS`: contribution `-0.052502`
- `lag_06__T_place_STAIRS`: contribution `-0.040080`
- `lag_00__T_place_JUNGLE`: contribution `-0.024300`
- `lag_13__T_shots_fired_sum`: contribution `-0.009135`
- `lag_15__CT2__flash_duration`: contribution `-0.006592`

Top utility-only movements:
- `lag_15__CT2__flash_duration`: contribution `-0.006592`
- `lag_13__T2__flash_duration`: contribution `-0.006015`
- `lag_14__CT3__flash_duration`: contribution `-0.004971`
- `lag_13__T4__flash_duration`: contribution `-0.004920`
- `lag_13__T_flash_duration_sum`: contribution `-0.003632`

### tick `81820`, seconds `65.50`, LSTM delta `-0.2059`

Top all feature movements:
- `lag_01__CT2__flash_duration`: contribution `-0.007585`
- `lag_01__T_shots_fired_sum`: contribution `-0.007371`
- `lag_09__CT3__flash_duration`: contribution `-0.007294`
- `lag_11__CT_place_JUNGLE`: contribution `-0.006774`
- `lag_06__CT_place_STAIRS`: contribution `-0.006172`

Top utility-only movements:
- `lag_01__CT2__flash_duration`: contribution `-0.007585`
- `lag_09__CT3__flash_duration`: contribution `-0.007294`
- `lag_00__CT3__flash_duration`: contribution `-0.005095`
- `lag_09__T4__flash_duration`: contribution `-0.004472`
- `lag_09__CT_flash_duration_sum`: contribution `-0.003979`

### tick `81788`, seconds `65.00`, LSTM delta `-0.1533`

Top all feature movements:
- `lag_00__CT2__flash_duration`: contribution `-0.005455`
- `lag_08__CT3__flash_duration`: contribution `-0.005115`
- `lag_05__CT_place_STAIRS`: contribution `-0.004961`
- `lag_00__T_kills_last_3s`: contribution `-0.004668`
- `lag_05__T_place_CONNECTOR`: contribution `-0.004366`

Top utility-only movements:
- `lag_00__CT2__flash_duration`: contribution `-0.005455`
- `lag_08__CT3__flash_duration`: contribution `-0.005115`
- `lag_08__CT2__flash_duration`: contribution `-0.003663`
- `lag_08__CT_flash_duration_sum`: contribution `-0.003483`
- `lag_08__T4__flash_duration`: contribution `-0.003442`

### tick `81724`, seconds `64.00`, LSTM delta `+0.0964`

Top all feature movements:
- `lag_06__CT3__flash_duration`: contribution `+0.006463`
- `lag_03__CT_place_STAIRS`: contribution `+0.006247`
- `lag_01__T_shots_fired_sum`: contribution `+0.005529`
- `lag_06__T_flashed_players`: contribution `+0.004407`
- `lag_01__CT2__shots_fired`: contribution `+0.004178`

Top utility-only movements:
- `lag_06__CT3__flash_duration`: contribution `+0.006463`
- `lag_06__CT_flash_duration_sum`: contribution `+0.002428`
- `lag_06__T2__flash_duration`: contribution `+0.002227`
- `lag_06__T4__flash_duration`: contribution `+0.002174`
- `lag_06__T_flash_duration_sum`: contribution `+0.001703`

### tick `81916`, seconds `67.00`, LSTM delta `-0.0470`

Top all feature movements:
- `lag_03__CT_place_STAIRS`: contribution `-0.006247`
- `lag_14__CT_place_JUNGLE`: contribution `-0.004109`
- `lag_00__T_place_CONNECTOR`: contribution `-0.003809`
- `lag_07__CT_place_TRUCK`: contribution `-0.003105`
- `lag_00__kill_diff_last_3s`: contribution `-0.002996`

Top utility-only movements:
- `lag_09__CT3__flash_duration`: contribution `+0.002211`
- `lag_12__T2__flash_duration`: contribution `+0.001757`
- `lag_03__CT3__flash_duration`: contribution `+0.001173`
- `lag_02__T4__flash_duration`: contribution `-0.001060`
