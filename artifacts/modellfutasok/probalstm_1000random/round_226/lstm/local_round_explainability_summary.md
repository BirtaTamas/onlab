# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-lynn-vision-bo3-80tf5tBYONxHYQuFp0AoSQ/legacy-vs-lynn-vision-m1-dust2.csv`
- round_num: `1`

## Largest probability jumps

- tick `7900`, seconds `65.50`, LSTM `0.0344`, delta `-0.0749`
- tick `7100`, seconds `53.00`, LSTM `0.4313`, delta `-0.0726`
- tick `7676`, seconds `62.00`, LSTM `0.1990`, delta `-0.0547`
- tick `6876`, seconds `49.50`, LSTM `0.4088`, delta `+0.0537`
- tick `6812`, seconds `48.50`, LSTM `0.3429`, delta `-0.0527`
- tick `7548`, seconds `60.00`, LSTM `0.2619`, delta `-0.0509`
- tick `8188`, seconds `70.00`, LSTM `0.0282`, delta `-0.0501`
- tick `6716`, seconds `47.00`, LSTM `0.3642`, delta `-0.0487`
- tick `6940`, seconds `50.50`, LSTM `0.4198`, delta `+0.0474`
- tick `6684`, seconds `46.50`, LSTM `0.4129`, delta `-0.0467`

## Top 15 local ridge features

- `lag_00__T_place_ARAMP`: coefficient `-0.002083`, |coef| `0.002083`
- `lag_00__CT_place_EXTENDEDA`: coefficient `0.001726`, |coef| `0.001726`
- `lag_00__CT_place_SHORTSTAIRS`: coefficient `-0.001342`, |coef| `0.001342`
- `lag_10__T_place_ARAMP`: coefficient `-0.001245`, |coef| `0.001245`
- `lag_06__T_place_ARAMP`: coefficient `-0.001185`, |coef| `0.001185`
- `lag_00__CT_place_BDOORS`: coefficient `-0.001173`, |coef| `0.001173`
- `lag_03__T_place_LONGA`: coefficient `-0.001030`, |coef| `0.001030`
- `lag_01__T_place_ARAMP`: coefficient `-0.000999`, |coef| `0.000999`
- `lag_02__T_place_ARAMP`: coefficient `-0.000980`, |coef| `0.000980`
- `lag_09__CT1__duck_amount`: coefficient `0.000946`, |coef| `0.000946`
- `lag_11__T_place_ARAMP`: coefficient `-0.000925`, |coef| `0.000925`
- `lag_15__CT2__duck_amount`: coefficient `0.000908`, |coef| `0.000908`
- `lag_00__T_place_PIT`: coefficient `-0.000887`, |coef| `0.000887`
- `lag_08__CT1__duck_amount`: coefficient `0.000882`, |coef| `0.000882`
- `lag_08__T_place_ARAMP`: coefficient `-0.000878`, |coef| `0.000878`

## Top 10 utility ridge features

- `lag_06__T3__flash_duration`: coefficient `-0.000457` (lowers CT win probability)
- `lag_06__CT5__flash_duration`: coefficient `-0.000419` (lowers CT win probability)
- `lag_06__T_flash_duration_sum`: coefficient `-0.000379` (lowers CT win probability)
- `lag_05__CT5__flash_duration`: coefficient `-0.000373` (lowers CT win probability)
- `lag_02__CT4__flash_duration`: coefficient `-0.000373` (lowers CT win probability)
- `lag_09__T3__flash_duration`: coefficient `-0.000371` (lowers CT win probability)
- `lag_07__CT5__flash_duration`: coefficient `-0.000361` (lowers CT win probability)
- `lag_09__CT5__flash_duration`: coefficient `-0.000359` (lowers CT win probability)
- `lag_11__CT5__flash_duration`: coefficient `-0.000350` (lowers CT win probability)
- `lag_10__CT5__flash_duration`: coefficient `-0.000348` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_ARAMP`: coefficient `-0.002083` (lowers CT win probability)
- `lag_00__CT_place_EXTENDEDA`: coefficient `0.001726` (raises CT win probability)
- `lag_00__CT_place_SHORTSTAIRS`: coefficient `-0.001342` (lowers CT win probability)
- `lag_10__T_place_ARAMP`: coefficient `-0.001245` (lowers CT win probability)
- `lag_06__T_place_ARAMP`: coefficient `-0.001185` (lowers CT win probability)
- `lag_00__CT_place_BDOORS`: coefficient `-0.001173` (lowers CT win probability)
- `lag_03__T_place_LONGA`: coefficient `-0.001030` (lowers CT win probability)
- `lag_01__T_place_ARAMP`: coefficient `-0.000999` (lowers CT win probability)
- `lag_02__T_place_ARAMP`: coefficient `-0.000980` (lowers CT win probability)
- `lag_09__CT1__duck_amount`: coefficient `0.000946` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `7900`, seconds `65.50`, LSTM delta `-0.0749`

Top all feature movements:
- `lag_00__T_place_ARAMP`: contribution `-0.037688`
- `lag_01__T_place_ARAMP`: contribution `+0.009035`
- `lag_11__T_place_ARAMP`: contribution `-0.008369`
- `lag_07__T_place_ARAMP`: contribution `-0.005680`
- `lag_00__T_place_PIT`: contribution `-0.005595`

Top utility-only movements:
- `lag_03__CT4__flash_duration`: contribution `-0.001286`

### tick `7100`, seconds `53.00`, LSTM delta `-0.0726`

Top all feature movements:
- `lag_00__T_place_SIDE`: contribution `-0.008904`
- `lag_15__CT_place_HOLE`: contribution `-0.007494`
- `lag_02__T_place_LONGA`: contribution `-0.005996`
- `lag_13__CT_place_HOLE`: contribution `-0.005220`
- `lag_02__T_place_PIT`: contribution `-0.004596`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `7676`, seconds `62.00`, LSTM delta `-0.0547`

Top all feature movements:
- `lag_00__T_place_ARAMP`: contribution `-0.018844`
- `lag_04__T_place_ARAMP`: contribution `-0.007371`
- `lag_01__CT_place_SHORTSTAIRS`: contribution `-0.004164`
- `lag_00__CT4__duck_amount`: contribution `-0.003166`
- `lag_09__CT2__duck_amount`: contribution `-0.002874`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `6876`, seconds `49.50`, LSTM delta `+0.0537`

Top all feature movements:
- `lag_15__CT_place_HOLE`: contribution `+0.007494`
- `lag_00__CT_place_BDOORS`: contribution `+0.005644`
- `lag_01__CT_place_HOLE`: contribution `+0.005257`
- `lag_15__CT2__duck_amount`: contribution `+0.003458`
- `lag_05__CT_place_EXTENDEDA`: contribution `+0.003184`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `6812`, seconds `48.50`, LSTM delta `-0.0527`

Top all feature movements:
- `lag_13__CT_place_HOLE`: contribution `-0.005220`
- `lag_03__CT_place_SHORTSTAIRS`: contribution `-0.004043`
- `lag_00__CT4__duck_amount`: contribution `-0.003166`
- `lag_06__CT_place_HOLE`: contribution `-0.002361`
- `lag_13__T_place_LONGA`: contribution `-0.002238`

Top utility-only movements:
- No utility movement among the top local contributors.
