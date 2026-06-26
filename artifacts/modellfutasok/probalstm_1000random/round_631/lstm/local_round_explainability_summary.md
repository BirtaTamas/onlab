# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-furia-vs-virtuspro-bo3-E_bOFuD3YUjLJCO2xRj0mq/furia-vs-virtus-pro-m1-mirage.csv`
- round_num: `2`

## Largest probability jumps

- tick `15088`, seconds `70.00`, LSTM `0.9167`, delta `+0.2066`
- tick `13360`, seconds `43.00`, LSTM `0.4241`, delta `+0.2057`
- tick `14704`, seconds `64.00`, LSTM `0.7404`, delta `+0.2037`
- tick `14800`, seconds `65.50`, LSTM `0.8826`, delta `+0.1881`
- tick `14832`, seconds `66.00`, LSTM `0.7277`, delta `-0.1549`
- tick `13424`, seconds `44.00`, LSTM `0.5163`, delta `+0.1084`
- tick `13456`, seconds `44.50`, LSTM `0.6128`, delta `+0.0965`
- tick `13552`, seconds `46.00`, LSTM `0.4562`, delta `-0.0873`
- tick `13488`, seconds `45.00`, LSTM `0.5450`, delta `-0.0678`
- tick `13072`, seconds `38.50`, LSTM `0.1611`, delta `-0.0661`

## Top 15 local ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.005379`, |coef| `0.005379`
- `lag_00__T_place_CONNECTOR`: coefficient `-0.003296`, |coef| `0.003296`
- `lag_00__CT5__shots_fired`: coefficient `0.002834`, |coef| `0.002834`
- `lag_00__CT_kills_last_3s`: coefficient `0.002493`, |coef| `0.002493`
- `lag_00__kill_diff_last_3s`: coefficient `0.002366`, |coef| `0.002366`
- `lag_02__CT_place_JUNGLE`: coefficient `0.001927`, |coef| `0.001927`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001863`, |coef| `0.001863`
- `lag_04__CT_place_JUNGLE`: coefficient `0.001733`, |coef| `0.001733`
- `lag_03__CT_shots_fired_sum`: coefficient `0.001712`, |coef| `0.001712`
- `lag_10__CT_place_JUNGLE`: coefficient `-0.001708`, |coef| `0.001708`
- `lag_15__CT5__is_walking`: coefficient `0.001645`, |coef| `0.001645`
- `lag_00__CT_place_JUNGLE`: coefficient `0.001601`, |coef| `0.001601`
- `lag_08__T_shots_fired_sum`: coefficient `-0.001562`, |coef| `0.001562`
- `lag_00__CT5__duck_amount`: coefficient `0.001504`, |coef| `0.001504`
- `lag_00__T4__shots_fired`: coefficient `-0.001480`, |coef| `0.001480`

## Top 10 utility ridge features

- `lag_11__CT_he_last_5s`: coefficient `-0.001441` (lowers CT win probability)
- `lag_13__T3__flash_duration`: coefficient `0.001249` (raises CT win probability)
- `lag_02__T_A_site_active_infernos`: coefficient `0.001101` (raises CT win probability)
- `lag_02__T3__flash_duration`: coefficient `0.001088` (raises CT win probability)
- `lag_06__T3__flash_duration`: coefficient `0.001043` (raises CT win probability)
- `lag_14__T3__flash_duration`: coefficient `0.000953` (raises CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.000927` (lowers CT win probability)
- `lag_10__T_flash_alpha_mean`: coefficient `-0.000899` (lowers CT win probability)
- `lag_11__T3__flash_duration`: coefficient `0.000898` (raises CT win probability)
- `lag_10__T3__flash_duration`: coefficient `0.000875` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.005379` (raises CT win probability)
- `lag_00__T_place_CONNECTOR`: coefficient `-0.003296` (lowers CT win probability)
- `lag_00__CT5__shots_fired`: coefficient `0.002834` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002493` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002366` (raises CT win probability)
- `lag_02__CT_place_JUNGLE`: coefficient `0.001927` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001863` (lowers CT win probability)
- `lag_04__CT_place_JUNGLE`: coefficient `0.001733` (raises CT win probability)
- `lag_03__CT_shots_fired_sum`: coefficient `0.001712` (raises CT win probability)
- `lag_10__CT_place_JUNGLE`: coefficient `-0.001708` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `15088`, seconds `70.00`, LSTM delta `+0.2066`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.022423`
- `lag_00__T_place_CONNECTOR`: contribution `+0.015961`
- `lag_10__CT_place_JUNGLE`: contribution `+0.010960`
- `lag_07__CT_place_LADDER`: contribution `+0.009671`
- `lag_00__CT5__shots_fired`: contribution `+0.008992`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `13360`, seconds `43.00`, LSTM delta `+0.2057`

Top all feature movements:
- `lag_08__T_shots_fired_sum`: contribution `+0.031613`
- `lag_00__CT_shots_fired_sum`: contribution `+0.029897`
- `lag_08__T4__shots_fired`: contribution `+0.015778`
- `lag_00__CT5__shots_fired`: contribution `+0.011989`
- `lag_10__CT_place_JUNGLE`: contribution `+0.010960`

Top utility-only movements:
- `lag_02__T3__flash_duration`: contribution `+0.005834`
- `lag_02__T4__flash_duration`: contribution `+0.002482`

### tick `14704`, seconds `64.00`, LSTM delta `+0.2037`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.026160`
- `lag_00__T_place_CONNECTOR`: contribution `+0.015961`
- `lag_02__CT_place_JUNGLE`: contribution `+0.012365`
- `lag_04__CT_place_JUNGLE`: contribution `+0.011118`
- `lag_00__CT5__shots_fired`: contribution `+0.010490`

Top utility-only movements:
- `lag_02__T_A_site_active_infernos`: contribution `+0.003279`

### tick `14800`, seconds `65.50`, LSTM delta `+0.1881`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.044846`
- `lag_00__T_place_CONNECTOR`: contribution `+0.015961`
- `lag_03__CT_shots_fired_sum`: contribution `+0.008324`
- `lag_07__CT_place_JUNGLE`: contribution `+0.008283`
- `lag_00__CT5__shots_fired`: contribution `+0.007493`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `14832`, seconds `66.00`, LSTM delta `-0.1549`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.048583`
- `lag_02__CT_place_JUNGLE`: contribution `-0.012365`
- `lag_03__CT_shots_fired_sum`: contribution `-0.010702`
- `lag_00__CT5__shots_fired`: contribution `-0.007493`
- `lag_08__CT_place_JUNGLE`: contribution `-0.006802`

Top utility-only movements:
- No utility movement among the top local contributors.
