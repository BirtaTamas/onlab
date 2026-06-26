# Local Round Explainability

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-aurora-vs-heroic-bo3-Z9VnvF_JkEDX6y_HyMsFXx/aurora-vs-heroic-m3-mirage.csv`
- round_num: `10`

## Largest probability jumps

- tick `57270`, seconds `0.50`, LSTM `0.9527`, delta `+0.0287`
- tick `61814`, seconds `71.50`, LSTM `0.9599`, delta `+0.0215`
- tick `62198`, seconds `77.50`, LSTM `0.9564`, delta `-0.0208`
- tick `60950`, seconds `58.00`, LSTM `0.9687`, delta `+0.0191`
- tick `61558`, seconds `67.50`, LSTM `0.9638`, delta `-0.0174`
- tick `61974`, seconds `74.00`, LSTM `0.9757`, delta `+0.0161`
- tick `62230`, seconds `78.00`, LSTM `0.9675`, delta `+0.0111`
- tick `60694`, seconds `54.00`, LSTM `0.9513`, delta `+0.0096`
- tick `61462`, seconds `66.00`, LSTM `0.9764`, delta `+0.0096`
- tick `60502`, seconds `51.00`, LSTM `0.9450`, delta `+0.0090`

## Top 15 local ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `0.000344`, |coef| `0.000344`
- `lag_01__T_place_TSPAWN`: coefficient `0.000333`, |coef| `0.000333`
- `lag_00__T_velocity_mean`: coefficient `0.000291`, |coef| `0.000291`
- `lag_12__CT_place_UNDERPASS`: coefficient `0.000285`, |coef| `0.000285`
- `lag_00__kill_diff_last_3s`: coefficient `0.000272`, |coef| `0.000272`
- `lag_02__T_place_TRUCK`: coefficient `0.000261`, |coef| `0.000261`
- `lag_01__molly_inv_diff`: coefficient `0.000257`, |coef| `0.000257`
- `lag_00__CT_shots_fired_sum`: coefficient `0.000252`, |coef| `0.000252`
- `lag_03__CT_place_STAIRS`: coefficient `0.000249`, |coef| `0.000249`
- `lag_01__utility_inv_diff`: coefficient `0.000247`, |coef| `0.000247`
- `lag_10__T_place_TRUCK`: coefficient `-0.000236`, |coef| `0.000236`
- `lag_12__T_place_TRUCK`: coefficient `0.000235`, |coef| `0.000235`
- `lag_00__CT_walking_count`: coefficient `-0.000229`, |coef| `0.000229`
- `lag_01__CT_molly_inv`: coefficient `0.000228`, |coef| `0.000228`
- `lag_00__CT_kills_last_3s`: coefficient `0.000228`, |coef| `0.000228`

## Top 10 utility ridge features

- `lag_01__molly_inv_diff`: coefficient `0.000257` (raises CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000247` (raises CT win probability)
- `lag_01__CT_molly_inv`: coefficient `0.000228` (raises CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000213` (raises CT win probability)
- `lag_01__CT4__molly`: coefficient `0.000177` (raises CT win probability)
- `lag_01__CT1__utility_total`: coefficient `0.000176` (raises CT win probability)
- `lag_01__CT1__molly`: coefficient `0.000164` (raises CT win probability)
- `lag_01__CT_utility_inv`: coefficient `0.000163` (raises CT win probability)
- `lag_01__CT1__smoke`: coefficient `0.000159` (raises CT win probability)
- `lag_01__CT2__molly`: coefficient `0.000154` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `0.000344` (raises CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `0.000333` (raises CT win probability)
- `lag_00__T_velocity_mean`: coefficient `0.000291` (raises CT win probability)
- `lag_12__CT_place_UNDERPASS`: coefficient `0.000285` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000272` (raises CT win probability)
- `lag_02__T_place_TRUCK`: coefficient `0.000261` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.000252` (raises CT win probability)
- `lag_03__CT_place_STAIRS`: coefficient `0.000249` (raises CT win probability)
- `lag_10__T_place_TRUCK`: coefficient `-0.000236` (lowers CT win probability)
- `lag_12__T_place_TRUCK`: coefficient `0.000235` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `57270`, seconds `0.50`, LSTM delta `+0.0287`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `+0.001644`
- `lag_01__T_place_TSPAWN`: contribution `+0.001475`
- `lag_00__T_velocity_mean`: contribution `+0.000942`
- `lag_01__molly_inv_diff`: contribution `+0.000843`
- `lag_00__CT_velocity_mean`: contribution `+0.000781`

Top utility-only movements:
- `lag_01__molly_inv_diff`: contribution `+0.000843`
- `lag_01__utility_inv_diff`: contribution `+0.000763`
- `lag_01__CT_molly_inv`: contribution `+0.000682`
- `lag_01__smoke_inv_diff`: contribution `+0.000548`
- `lag_01__CT1__utility_total`: contribution `+0.000346`

### tick `61814`, seconds `71.50`, LSTM delta `+0.0215`

Top all feature movements:
- `lag_10__T_place_TRUCK`: contribution `+0.004093`
- `lag_12__T_place_TRUCK`: contribution `+0.004076`
- `lag_04__CT_place_JUNGLE`: contribution `+0.000951`
- `lag_00__CT_kills_last_3s`: contribution `+0.000658`
- `lag_00__kill_diff_last_3s`: contribution `+0.000654`

Top utility-only movements:
- `lag_08__T3__flash_duration`: contribution `+0.000405`
- `lag_08__T2__flash_duration`: contribution `+0.000399`

### tick `62198`, seconds `77.50`, LSTM delta `-0.0208`

Top all feature movements:
- `lag_01__T_duck_amount_mean`: contribution `-0.001031`
- `lag_07__T_bomb_zone_count`: contribution `-0.000770`
- `lag_00__CT_shots_fired_sum`: contribution `-0.000699`
- `lag_00__kill_diff_last_3s`: contribution `-0.000654`
- `lag_06__CT_place_CATWALK`: contribution `-0.000632`

Top utility-only movements:
- `lag_11__T3__flash_duration`: contribution `-0.000619`
- `lag_11__T2__flash_duration`: contribution `-0.000610`
- `lag_11__T_flash_duration_sum`: contribution `-0.000424`

### tick `60950`, seconds `58.00`, LSTM delta `+0.0191`

Top all feature movements:
- `lag_03__CT_place_STAIRS`: contribution `+0.001938`
- `lag_12__CT_place_UNDERPASS`: contribution `+0.001650`
- `lag_00__CT_place_TRUCK`: contribution `-0.001350`
- `lag_00__CT_shots_fired_sum`: contribution `+0.000874`
- `lag_14__CT_place_TRUCK`: contribution `+0.000782`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `61558`, seconds `67.50`, LSTM delta `-0.0174`

Top all feature movements:
- `lag_02__T_place_TRUCK`: contribution `-0.004538`
- `lag_04__T_place_TRUCK`: contribution `-0.002468`
- `lag_04__CT_place_JUNGLE`: contribution `-0.000951`
- `lag_08__CT_place_JUNGLE`: contribution `-0.000766`
- `lag_00__kill_diff_last_3s`: contribution `-0.000654`

Top utility-only movements:
- `lag_00__T3__flash_duration`: contribution `-0.000478`
- `lag_00__T2__flash_duration`: contribution `-0.000471`
- `lag_00__T_flash_duration_sum`: contribution `-0.000304`
