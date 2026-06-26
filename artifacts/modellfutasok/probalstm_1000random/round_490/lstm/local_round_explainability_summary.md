# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m1-inferno.csv`
- round_num: `5`

## Largest probability jumps

- tick `39400`, seconds `18.50`, LSTM `0.1322`, delta `-0.1478`
- tick `38376`, seconds `2.50`, LSTM `0.2025`, delta `+0.0813`
- tick `39656`, seconds `22.50`, LSTM `0.0284`, delta `-0.0758`
- tick `38696`, seconds `7.50`, LSTM `0.1897`, delta `-0.0614`
- tick `38408`, seconds `3.00`, LSTM `0.2439`, delta `+0.0414`
- tick `38792`, seconds `9.00`, LSTM `0.1819`, delta `+0.0326`
- tick `38728`, seconds `8.00`, LSTM `0.1579`, delta `-0.0318`
- tick `38888`, seconds `10.50`, LSTM `0.2357`, delta `+0.0286`
- tick `39016`, seconds `12.50`, LSTM `0.2596`, delta `+0.0262`
- tick `39176`, seconds `15.00`, LSTM `0.2654`, delta `+0.0238`

## Top 15 local ridge features

- `lag_00__CT_place_BANANA`: coefficient `0.001171`, |coef| `0.001171`
- `lag_00__CT1__duck_amount`: coefficient `0.001126`, |coef| `0.001126`
- `lag_00__CT_he_last_5s`: coefficient `0.001122`, |coef| `0.001122`
- `lag_14__T_place_UNDERPASS`: coefficient `-0.001000`, |coef| `0.001000`
- `lag_01__CT3__duck_amount`: coefficient `0.000981`, |coef| `0.000981`
- `lag_00__T_kills_last_3s`: coefficient `-0.000894`, |coef| `0.000894`
- `lag_12__CT1__duck_amount`: coefficient `-0.000887`, |coef| `0.000887`
- `lag_04__CT_smokes_last_5s`: coefficient `0.000866`, |coef| `0.000866`
- `lag_01__T_A_site_active_infernos`: coefficient `0.000805`, |coef| `0.000805`
- `lag_15__CT_place_TOPOFMID`: coefficient `-0.000785`, |coef| `0.000785`
- `lag_00__damage_diff_last_5s`: coefficient `0.000769`, |coef| `0.000769`
- `lag_14__T_place_SECONDMID`: coefficient `0.000755`, |coef| `0.000755`
- `lag_01__CT_flashes_last_5s`: coefficient `0.000755`, |coef| `0.000755`
- `lag_09__CT_place_BANANA`: coefficient `0.000750`, |coef| `0.000750`
- `lag_00__T_damage_last_5s`: coefficient `-0.000736`, |coef| `0.000736`

## Top 10 utility ridge features

- `lag_00__CT_he_last_5s`: coefficient `0.001122` (raises CT win probability)
- `lag_04__CT_smokes_last_5s`: coefficient `0.000866` (raises CT win probability)
- `lag_01__T_A_site_active_infernos`: coefficient `0.000805` (raises CT win probability)
- `lag_01__CT_flashes_last_5s`: coefficient `0.000755` (raises CT win probability)
- `lag_15__T_A_site_active_infernos`: coefficient `-0.000690` (lowers CT win probability)
- `lag_05__CT4__smoke`: coefficient `0.000663` (raises CT win probability)
- `lag_01__CT_he_last_5s`: coefficient `0.000636` (raises CT win probability)
- `lag_08__T3__smoke`: coefficient `0.000614` (raises CT win probability)
- `lag_00__CT_flashes_last_5s`: coefficient `0.000577` (raises CT win probability)
- `lag_01__T_active_infernos`: coefficient `0.000527` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_BANANA`: coefficient `0.001171` (raises CT win probability)
- `lag_00__CT1__duck_amount`: coefficient `0.001126` (raises CT win probability)
- `lag_14__T_place_UNDERPASS`: coefficient `-0.001000` (lowers CT win probability)
- `lag_01__CT3__duck_amount`: coefficient `0.000981` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.000894` (lowers CT win probability)
- `lag_12__CT1__duck_amount`: coefficient `-0.000887` (lowers CT win probability)
- `lag_15__CT_place_TOPOFMID`: coefficient `-0.000785` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000769` (raises CT win probability)
- `lag_14__T_place_SECONDMID`: coefficient `0.000755` (raises CT win probability)
- `lag_09__CT_place_BANANA`: coefficient `0.000750` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `39400`, seconds `18.50`, LSTM delta `-0.1478`

Top all feature movements:
- `lag_00__CT1__duck_amount`: contribution `-0.004297`
- `lag_14__T_place_UNDERPASS`: contribution `-0.003917`
- `lag_01__CT3__duck_amount`: contribution `-0.003651`
- `lag_00__CT_place_BANANA`: contribution `-0.003468`
- `lag_12__CT1__duck_amount`: contribution `-0.003385`

Top utility-only movements:
- `lag_01__T_A_site_active_infernos`: contribution `-0.002397`
- `lag_15__T_A_site_active_infernos`: contribution `-0.002053`

### tick `38376`, seconds `2.50`, LSTM delta `+0.0813`

Top all feature movements:
- `lag_00__CT_he_last_5s`: contribution `+0.020590`
- `lag_04__CT_smokes_last_5s`: contribution `+0.014969`
- `lag_01__CT_flashes_last_5s`: contribution `+0.008303`
- `lag_00__CT_smokes_last_5s`: contribution `+0.004504`
- `lag_05__CT_place_CTSPAWN`: contribution `+0.001391`

Top utility-only movements:
- `lag_00__CT_he_last_5s`: contribution `+0.020590`
- `lag_04__CT_smokes_last_5s`: contribution `+0.014969`
- `lag_01__CT_flashes_last_5s`: contribution `+0.008303`
- `lag_00__CT_smokes_last_5s`: contribution `+0.004504`
- `lag_05__CT4__smoke`: contribution `+0.001013`

### tick `39656`, seconds `22.50`, LSTM delta `-0.0758`

Top all feature movements:
- `lag_07__CT_place_BALCONY`: contribution `-0.004375`
- `lag_00__CT_place_BANANA`: contribution `-0.003468`
- `lag_04__T5__is_scoped`: contribution `-0.003188`
- `lag_00__T_kills_last_3s`: contribution `-0.002831`
- `lag_00__T_damage_last_5s`: contribution `-0.001765`

Top utility-only movements:
- `lag_09__T_A_site_active_infernos`: contribution `-0.001132`

### tick `38696`, seconds `7.50`, LSTM delta `-0.0614`

Top all feature movements:
- `lag_00__CT_he_last_5s`: contribution `-0.020590`
- `lag_04__CT_smokes_last_5s`: contribution `-0.014969`
- `lag_01__CT_flashes_last_5s`: contribution `-0.008303`
- `lag_10__CT_smokes_last_5s`: contribution `-0.006080`
- `lag_00__CT_smokes_last_5s`: contribution `-0.004504`

Top utility-only movements:
- `lag_00__CT_he_last_5s`: contribution `-0.020590`
- `lag_04__CT_smokes_last_5s`: contribution `-0.014969`
- `lag_01__CT_flashes_last_5s`: contribution `-0.008303`
- `lag_10__CT_smokes_last_5s`: contribution `-0.006080`
- `lag_00__CT_smokes_last_5s`: contribution `-0.004504`

### tick `38408`, seconds `3.00`, LSTM delta `+0.0414`

Top all feature movements:
- `lag_01__CT_he_last_5s`: contribution `+0.011665`
- `lag_05__CT_smokes_last_5s`: contribution `+0.006524`
- `lag_00__CT_flashes_last_5s`: contribution `+0.006347`
- `lag_01__CT_smokes_last_5s`: contribution `+0.003323`
- `lag_02__CT_flashes_last_5s`: contribution `+0.002774`

Top utility-only movements:
- `lag_01__CT_he_last_5s`: contribution `+0.011665`
- `lag_05__CT_smokes_last_5s`: contribution `+0.006524`
- `lag_00__CT_flashes_last_5s`: contribution `+0.006347`
- `lag_01__CT_smokes_last_5s`: contribution `+0.003323`
- `lag_02__CT_flashes_last_5s`: contribution `+0.002774`
