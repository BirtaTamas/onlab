# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-the-mongolz-vs-m80-bo3-e7FibL-GpwhFRhM0kGS5r4/the-mongolz-vs-m80-m3-inferno.csv`
- round_num: `2`

## Largest probability jumps

- tick `13346`, seconds `31.50`, LSTM `0.8002`, delta `+0.1435`
- tick `14082`, seconds `43.00`, LSTM `0.9129`, delta `+0.1133`
- tick `14050`, seconds `42.50`, LSTM `0.7996`, delta `+0.0624`
- tick `13378`, seconds `32.00`, LSTM `0.8466`, delta `+0.0465`
- tick `13186`, seconds `29.00`, LSTM `0.6007`, delta `+0.0422`
- tick `13154`, seconds `28.50`, LSTM `0.5584`, delta `+0.0419`
- tick `13698`, seconds `37.00`, LSTM `0.7501`, delta `-0.0400`
- tick `14146`, seconds `44.00`, LSTM `0.9602`, delta `+0.0327`
- tick `13218`, seconds `29.50`, LSTM `0.6333`, delta `+0.0327`
- tick `13666`, seconds `36.50`, LSTM `0.7901`, delta `-0.0307`

## Top 15 local ridge features

- `lag_12__T_place_DECK`: coefficient `-0.001864`, |coef| `0.001864`
- `lag_09__T2__duck_amount`: coefficient `0.001858`, |coef| `0.001858`
- `lag_00__CT_damage_last_5s`: coefficient `0.001568`, |coef| `0.001568`
- `lag_00__damage_diff_last_5s`: coefficient `0.001462`, |coef| `0.001462`
- `lag_06__T5__duck_amount`: coefficient `-0.001318`, |coef| `0.001318`
- `lag_09__T_duck_amount_mean`: coefficient `0.001310`, |coef| `0.001310`
- `lag_15__CT_place_TOPOFMID`: coefficient `0.001231`, |coef| `0.001231`
- `lag_15__CT_place_APARTMENTS`: coefficient `-0.001228`, |coef| `0.001228`
- `lag_11__T_place_DECK`: coefficient `-0.001210`, |coef| `0.001210`
- `lag_08__T2__duck_amount`: coefficient `0.001144`, |coef| `0.001144`
- `lag_08__T5__duck_amount`: coefficient `0.001097`, |coef| `0.001097`
- `lag_00__CT_kills_last_3s`: coefficient `0.001059`, |coef| `0.001059`
- `lag_01__CT_damage_last_5s`: coefficient `0.001054`, |coef| `0.001054`
- `lag_09__CT1__is_walking`: coefficient `0.001024`, |coef| `0.001024`
- `lag_01__damage_diff_last_5s`: coefficient `0.001018`, |coef| `0.001018`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.000760` (lowers CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `-0.000584` (lowers CT win probability)
- `lag_00__T3__flash_duration`: coefficient `0.000426` (raises CT win probability)
- `lag_02__T_flash_alpha_mean`: coefficient `-0.000388` (lowers CT win probability)
- `lag_07__T3__flash_duration`: coefficient `0.000381` (raises CT win probability)
- `lag_11__T3__flash_duration`: coefficient `0.000379` (raises CT win probability)
- `lag_15__T3__flash_duration`: coefficient `0.000368` (raises CT win probability)
- `lag_12__T_flash_alpha_mean`: coefficient `-0.000367` (lowers CT win probability)
- `lag_09__T3__flash_duration`: coefficient `0.000356` (raises CT win probability)
- `lag_13__T3__flash_duration`: coefficient `0.000349` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_12__T_place_DECK`: coefficient `-0.001864` (lowers CT win probability)
- `lag_09__T2__duck_amount`: coefficient `0.001858` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001568` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001462` (raises CT win probability)
- `lag_06__T5__duck_amount`: coefficient `-0.001318` (lowers CT win probability)
- `lag_09__T_duck_amount_mean`: coefficient `0.001310` (raises CT win probability)
- `lag_15__CT_place_TOPOFMID`: coefficient `0.001231` (raises CT win probability)
- `lag_15__CT_place_APARTMENTS`: coefficient `-0.001228` (lowers CT win probability)
- `lag_11__T_place_DECK`: coefficient `-0.001210` (lowers CT win probability)
- `lag_08__T2__duck_amount`: coefficient `0.001144` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `13346`, seconds `31.50`, LSTM delta `+0.1435`

Top all feature movements:
- `lag_09__T2__duck_amount`: contribution `+0.006653`
- `lag_06__T5__duck_amount`: contribution `+0.005006`
- `lag_15__CT_place_APARTMENTS`: contribution `+0.004716`
- `lag_15__CT_place_TOPOFMID`: contribution `+0.004467`
- `lag_08__T5__duck_amount`: contribution `+0.004164`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `14082`, seconds `43.00`, LSTM delta `+0.1133`

Top all feature movements:
- `lag_12__T_place_DECK`: contribution `+0.045216`
- `lag_09__T2__duck_amount`: contribution `+0.007105`
- `lag_00__CT_kills_last_3s`: contribution `+0.003057`
- `lag_04__CT_place_ARCH`: contribution `+0.002820`
- `lag_08__CT_place_TOPOFMID`: contribution `+0.002547`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `14050`, seconds `42.50`, LSTM delta `+0.0624`

Top all feature movements:
- `lag_11__T_place_DECK`: contribution `+0.029350`
- `lag_08__T2__duck_amount`: contribution `+0.004376`
- `lag_07__CT_place_TOPOFMID`: contribution `+0.002951`
- `lag_03__CT_place_ARCH`: contribution `+0.002483`
- `lag_09__CT1__is_walking`: contribution `+0.002389`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `13378`, seconds `32.00`, LSTM delta `+0.0465`

Top all feature movements:
- `lag_10__T2__duck_amount`: contribution `+0.003147`
- `lag_09__T5__duck_amount`: contribution `+0.002695`
- `lag_09__T_duck_amount_mean`: contribution `+0.002493`
- `lag_01__CT_damage_last_5s`: contribution `+0.002297`
- `lag_01__damage_diff_last_5s`: contribution `+0.002296`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `13186`, seconds `29.00`, LSTM delta `+0.0422`

Top all feature movements:
- `lag_08__T5__duck_amount`: contribution `+0.004164`
- `lag_10__CT_place_TOPOFMID`: contribution `+0.003324`
- `lag_04__T2__duck_amount`: contribution `+0.002495`
- `lag_01__CT_damage_last_5s`: contribution `+0.002297`
- `lag_01__damage_diff_last_5s`: contribution `+0.002296`

Top utility-only movements:
- No utility movement among the top local contributors.
