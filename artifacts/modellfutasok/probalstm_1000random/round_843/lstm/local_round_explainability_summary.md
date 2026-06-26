# Local Round Explainability

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-g2-vs-liquid-bo3-w6HylYj4nF7GNnrWujmZUZ/g2-vs-liquid-m2-inferno.csv`
- round_num: `1`

## Largest probability jumps

- tick `14709`, seconds `99.00`, LSTM `0.8116`, delta `+0.2410`
- tick `11765`, seconds `53.00`, LSTM `0.7683`, delta `+0.2084`
- tick `13301`, seconds `77.00`, LSTM `0.7945`, delta `-0.1395`
- tick `14901`, seconds `102.00`, LSTM `0.9272`, delta `+0.0883`
- tick `12117`, seconds `58.50`, LSTM `0.8658`, delta `-0.0867`
- tick `11829`, seconds `54.00`, LSTM `0.8906`, delta `+0.0855`
- tick `11477`, seconds `48.50`, LSTM `0.6344`, delta `+0.0854`
- tick `11893`, seconds `55.00`, LSTM `0.9292`, delta `+0.0696`
- tick `11701`, seconds `52.00`, LSTM `0.5629`, delta `-0.0523`
- tick `14069`, seconds `89.00`, LSTM `0.6154`, delta `-0.0461`

## Top 15 local ridge features

- `lag_10__CT_place_LIBRARY`: coefficient `-0.003572`, |coef| `0.003572`
- `lag_00__CT_defusing_count`: coefficient `0.003154`, |coef| `0.003154`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.002807`, |coef| `0.002807`
- `lag_00__kill_diff_last_3s`: coefficient `0.002397`, |coef| `0.002397`
- `lag_00__CT_place_SECONDMID`: coefficient `0.002279`, |coef| `0.002279`
- `lag_14__CT_defusing_count`: coefficient `-0.002266`, |coef| `0.002266`
- `lag_00__CT_kills_last_3s`: coefficient `0.002003`, |coef| `0.002003`
- `lag_00__T_duck_amount_mean`: coefficient `-0.001959`, |coef| `0.001959`
- `lag_09__T1__flash_duration`: coefficient `0.001910`, |coef| `0.001910`
- `lag_15__CT_defusing_count`: coefficient `-0.001892`, |coef| `0.001892`
- `lag_00__damage_diff_last_5s`: coefficient `0.001817`, |coef| `0.001817`
- `lag_06__T_flash_alpha_mean`: coefficient `-0.001790`, |coef| `0.001790`
- `lag_14__CT4__flash_duration`: coefficient `0.001782`, |coef| `0.001782`
- `lag_00__CT2__is_walking`: coefficient `-0.001727`, |coef| `0.001727`
- `lag_12__CT_place_ARCH`: coefficient `-0.001616`, |coef| `0.001616`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.002807` (lowers CT win probability)
- `lag_09__T1__flash_duration`: coefficient `0.001910` (raises CT win probability)
- `lag_06__T_flash_alpha_mean`: coefficient `-0.001790` (lowers CT win probability)
- `lag_14__CT4__flash_duration`: coefficient `0.001782` (raises CT win probability)
- `lag_09__T4__flash_duration`: coefficient `0.001457` (raises CT win probability)
- `lag_01__CT4__flash_duration`: coefficient `-0.001450` (lowers CT win probability)
- `lag_09__T_flash_duration_sum`: coefficient `0.001382` (raises CT win probability)
- `lag_05__CT4__flash_duration`: coefficient `0.001029` (raises CT win probability)
- `lag_10__T4__flash_duration`: coefficient `0.000902` (raises CT win probability)
- `lag_10__CT_B_site_active_smokes`: coefficient `0.000859` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_10__CT_place_LIBRARY`: coefficient `-0.003572` (lowers CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.003154` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002397` (raises CT win probability)
- `lag_00__CT_place_SECONDMID`: coefficient `0.002279` (raises CT win probability)
- `lag_14__CT_defusing_count`: coefficient `-0.002266` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002003` (raises CT win probability)
- `lag_00__T_duck_amount_mean`: coefficient `-0.001959` (lowers CT win probability)
- `lag_15__CT_defusing_count`: coefficient `-0.001892` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001817` (raises CT win probability)
- `lag_00__CT2__is_walking`: coefficient `-0.001727` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `14709`, seconds `99.00`, LSTM delta `+0.2410`

Top all feature movements:
- `lag_10__CT_place_LIBRARY`: contribution `+0.022901`
- `lag_00__T_flash_alpha_mean`: contribution `+0.017029`
- `lag_14__T_duck_amount_mean`: contribution `+0.007938`
- `lag_12__CT_place_ARCH`: contribution `+0.006595`
- `lag_00__T_duck_amount_mean`: contribution `+0.006103`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.017029`

### tick `11765`, seconds `53.00`, LSTM delta `+0.2084`

Top all feature movements:
- `lag_14__CT4__flash_duration`: contribution `+0.012963`
- `lag_09__T1__flash_duration`: contribution `+0.012071`
- `lag_01__CT4__flash_duration`: contribution `+0.010544`
- `lag_09__T4__flash_duration`: contribution `+0.008542`
- `lag_09__T_flash_duration_sum`: contribution `+0.006930`

Top utility-only movements:
- `lag_14__CT4__flash_duration`: contribution `+0.012963`
- `lag_09__T1__flash_duration`: contribution `+0.012071`
- `lag_01__CT4__flash_duration`: contribution `+0.010544`
- `lag_09__T4__flash_duration`: contribution `+0.008542`
- `lag_09__T_flash_duration_sum`: contribution `+0.006930`

### tick `13301`, seconds `77.00`, LSTM delta `-0.1395`

Top all feature movements:
- `lag_00__CT_place_SECONDMID`: contribution `-0.046732`
- `lag_08__T_duck_amount_mean`: contribution `-0.009149`
- `lag_00__kill_diff_last_3s`: contribution `-0.005770`
- `lag_15__CT5__is_walking`: contribution `-0.003226`
- `lag_00__T_kills_last_3s`: contribution `-0.003034`

Top utility-only movements:
- `lag_10__CT_B_site_active_smokes`: contribution `-0.001427`

### tick `14901`, seconds `102.00`, LSTM delta `+0.0883`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `+0.030573`
- `lag_06__T_flash_alpha_mean`: contribution `+0.010859`
- `lag_00__CT_kills_last_3s`: contribution `-0.005782`
- `lag_00__kill_diff_last_3s`: contribution `-0.005770`
- `lag_06__CT2__duck_amount`: contribution `-0.003235`

Top utility-only movements:
- `lag_06__T_flash_alpha_mean`: contribution `+0.010859`

### tick `12117`, seconds `58.50`, LSTM delta `-0.0867`

Top all feature movements:
- `lag_09__T1__flash_duration`: contribution `-0.012071`
- `lag_01__T_duck_amount_mean`: contribution `+0.006597`
- `lag_00__kill_diff_last_3s`: contribution `-0.005770`
- `lag_00__damage_diff_last_5s`: contribution `-0.005329`
- `lag_10__T4__flash_duration`: contribution `-0.005289`

Top utility-only movements:
- `lag_09__T1__flash_duration`: contribution `-0.012071`
- `lag_10__T4__flash_duration`: contribution `-0.005289`
- `lag_09__T_flash_duration_sum`: contribution `-0.003634`
- `lag_12__CT4__flash_duration`: contribution `+0.002032`
