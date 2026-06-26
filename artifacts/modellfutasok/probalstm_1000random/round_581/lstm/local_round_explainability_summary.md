# Local Round Explainability

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-spirit-vs-g2-bo3-3aFk7fRwd7iUE0VJycUPHK/spirit-vs-g2-m3-ancient.csv`
- round_num: `12`

## Largest probability jumps

- tick `89433`, seconds `23.00`, LSTM `0.0850`, delta `-0.3068`
- tick `88857`, seconds `14.00`, LSTM `0.4581`, delta `+0.2526`
- tick `89561`, seconds `25.00`, LSTM `0.0543`, delta `-0.2416`
- tick `88537`, seconds `9.00`, LSTM `0.1858`, delta `-0.2402`
- tick `89529`, seconds `24.50`, LSTM `0.2959`, delta `+0.2365`
- tick `90393`, seconds `38.00`, LSTM `0.1318`, delta `+0.1023`
- tick `90041`, seconds `32.50`, LSTM `0.1862`, delta `-0.0763`
- tick `89945`, seconds `31.00`, LSTM `0.2146`, delta `+0.0743`
- tick `90841`, seconds `45.00`, LSTM `0.0396`, delta `-0.0719`
- tick `89785`, seconds `28.50`, LSTM `0.1419`, delta `+0.0564`

## Top 15 local ridge features

- `lag_15__T_he_last_5s`: coefficient `-0.003981`, |coef| `0.003981`
- `lag_00__kill_diff_last_3s`: coefficient `0.003399`, |coef| `0.003399`
- `lag_03__T1__shots_fired`: coefficient `-0.003345`, |coef| `0.003345`
- `lag_00__T1__shots_fired`: coefficient `0.003203`, |coef| `0.003203`
- `lag_02__T_shots_fired_sum`: coefficient `-0.003017`, |coef| `0.003017`
- `lag_00__T_kills_last_3s`: coefficient `-0.002614`, |coef| `0.002614`
- `lag_00__CT_place_UNKNOWN`: coefficient `0.002533`, |coef| `0.002533`
- `lag_00__T_damage_last_5s`: coefficient `-0.002406`, |coef| `0.002406`
- `lag_00__damage_diff_last_5s`: coefficient `0.002290`, |coef| `0.002290`
- `lag_00__CT_place_SIDEENTRANCE`: coefficient `0.001886`, |coef| `0.001886`
- `lag_07__CT_place_TSIDEUPPER`: coefficient `-0.001762`, |coef| `0.001762`
- `lag_05__CT1__duck_amount`: coefficient `-0.001732`, |coef| `0.001732`
- `lag_04__CT_place_TOPOFMID`: coefficient `-0.001712`, |coef| `0.001712`
- `lag_00__CT_kills_last_3s`: coefficient `0.001694`, |coef| `0.001694`
- `lag_08__CT2__flash_duration`: coefficient `-0.001647`, |coef| `0.001647`

## Top 10 utility ridge features

- `lag_15__T_he_last_5s`: coefficient `-0.003981` (lowers CT win probability)
- `lag_08__CT2__flash_duration`: coefficient `-0.001647` (lowers CT win probability)
- `lag_15__CT_B_site_active_infernos`: coefficient `0.001538` (raises CT win probability)
- `lag_11__T_B_site_active_infernos`: coefficient `-0.001517` (lowers CT win probability)
- `lag_05__T_he_last_5s`: coefficient `0.001473` (raises CT win probability)
- `lag_02__CT2__flash_duration`: coefficient `0.001355` (raises CT win probability)
- `lag_00__T5__flash_duration`: coefficient `-0.001238` (lowers CT win probability)
- `lag_04__T_B_site_active_infernos`: coefficient `0.001170` (raises CT win probability)
- `lag_11__CT_B_site_active_infernos`: coefficient `0.001165` (raises CT win probability)
- `lag_11__T_active_infernos`: coefficient `-0.001105` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003399` (raises CT win probability)
- `lag_03__T1__shots_fired`: coefficient `-0.003345` (lowers CT win probability)
- `lag_00__T1__shots_fired`: coefficient `0.003203` (raises CT win probability)
- `lag_02__T_shots_fired_sum`: coefficient `-0.003017` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002614` (lowers CT win probability)
- `lag_00__CT_place_UNKNOWN`: coefficient `0.002533` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002406` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002290` (raises CT win probability)
- `lag_00__CT_place_SIDEENTRANCE`: coefficient `0.001886` (raises CT win probability)
- `lag_07__CT_place_TSIDEUPPER`: coefficient `-0.001762` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `89433`, seconds `23.00`, LSTM delta `-0.3068`

Top all feature movements:
- `lag_00__T1__shots_fired`: contribution `-0.032546`
- `lag_02__T_shots_fired_sum`: contribution `-0.011309`
- `lag_03__T1__shots_fired`: contribution `-0.009997`
- `lag_00__T_kills_last_3s`: contribution `-0.008282`
- `lag_00__kill_diff_last_3s`: contribution `-0.008180`

Top utility-only movements:
- `lag_15__CT_B_site_active_infernos`: contribution `-0.005283`
- `lag_08__CT2__flash_duration`: contribution `-0.004876`
- `lag_11__T_B_site_active_infernos`: contribution `-0.004288`

### tick `88857`, seconds `14.00`, LSTM delta `+0.2526`

Top all feature movements:
- `lag_15__T_he_last_5s`: contribution `+0.051954`
- `lag_08__T_place_TSIDELOWER`: contribution `+0.010406`
- `lag_00__damage_diff_last_5s`: contribution `+0.009352`
- `lag_14__CT_place_TOPOFMID`: contribution `+0.008436`
- `lag_00__kill_diff_last_3s`: contribution `+0.008180`

Top utility-only movements:
- `lag_15__T_he_last_5s`: contribution `+0.051954`
- `lag_10__T5__flash_duration`: contribution `+0.007028`
- `lag_07__CT4__flash_duration`: contribution `+0.004357`

### tick `89561`, seconds `25.00`, LSTM delta `-0.2416`

Top all feature movements:
- `lag_04__T1__shots_fired`: contribution `-0.015601`
- `lag_03__T_shots_fired_sum`: contribution `+0.012742`
- `lag_00__T_kills_last_3s`: contribution `-0.008282`
- `lag_00__kill_diff_last_3s`: contribution `-0.008180`
- `lag_00__CT_place_SIDEENTRANCE`: contribution `-0.007590`

Top utility-only movements:
- `lag_15__CT_B_site_active_infernos`: contribution `-0.005283`

### tick `88537`, seconds `9.00`, LSTM delta `-0.2402`

Top all feature movements:
- `lag_15__T_he_last_5s`: contribution `-0.051954`
- `lag_05__T_he_last_5s`: contribution `-0.019221`
- `lag_04__CT_place_TOPOFMID`: contribution `-0.012424`
- `lag_00__T5__flash_duration`: contribution `-0.008492`
- `lag_02__CT_place_TOPOFMID`: contribution `-0.008460`

Top utility-only movements:
- `lag_15__T_he_last_5s`: contribution `-0.051954`
- `lag_05__T_he_last_5s`: contribution `-0.019221`
- `lag_00__T5__flash_duration`: contribution `-0.008492`
- `lag_03__CT3__flash_duration`: contribution `-0.003172`

### tick `89529`, seconds `24.50`, LSTM delta `+0.2365`

Top all feature movements:
- `lag_03__T1__shots_fired`: contribution `+0.033988`
- `lag_02__T_shots_fired_sum`: contribution `+0.024879`
- `lag_00__kill_diff_last_3s`: contribution `+0.008180`
- `lag_03__T_shots_fired_sum`: contribution `+0.008109`
- `lag_00__T_place_SIDEENTRANCE`: contribution `+0.006787`

Top utility-only movements:
- No utility movement among the top local contributors.
