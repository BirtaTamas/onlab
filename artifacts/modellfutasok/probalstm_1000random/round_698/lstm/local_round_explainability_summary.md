# Local Round Explainability

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-flyquest-vs-legacy-bo3-FlEa8e0vdBrf1ft_mNbThh/flyquest-vs-legacy-m2-nuke.csv`
- round_num: `14`

## Largest probability jumps

- tick `140370`, seconds `19.00`, LSTM `0.7255`, delta `+0.1541`
- tick `141970`, seconds `44.00`, LSTM `0.9102`, delta `+0.1069`
- tick `140402`, seconds `19.50`, LSTM `0.8063`, delta `+0.0808`
- tick `142898`, seconds `58.50`, LSTM `0.9277`, delta `+0.0741`
- tick `142866`, seconds `58.00`, LSTM `0.8536`, delta `+0.0728`
- tick `142834`, seconds `57.50`, LSTM `0.7808`, delta `+0.0652`
- tick `140498`, seconds `21.00`, LSTM `0.7789`, delta `-0.0638`
- tick `141938`, seconds `43.50`, LSTM `0.8033`, delta `+0.0575`
- tick `142290`, seconds `49.00`, LSTM `0.7788`, delta `-0.0536`
- tick `142802`, seconds `57.00`, LSTM `0.7156`, delta `-0.0531`

## Top 15 local ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.002341`, |coef| `0.002341`
- `lag_09__CT_place_DECON`: coefficient `-0.001730`, |coef| `0.001730`
- `lag_00__CT_kills_last_3s`: coefficient `0.001657`, |coef| `0.001657`
- `lag_00__CT2__shots_fired`: coefficient `0.001638`, |coef| `0.001638`
- `lag_03__T_place_SQUEAKY`: coefficient `0.001553`, |coef| `0.001553`
- `lag_00__T_place_SQUEAKY`: coefficient `-0.001539`, |coef| `0.001539`
- `lag_01__T_place_ROOF`: coefficient `-0.001500`, |coef| `0.001500`
- `lag_00__damage_diff_last_5s`: coefficient `0.001468`, |coef| `0.001468`
- `lag_00__kill_diff_last_3s`: coefficient `0.001464`, |coef| `0.001464`
- `lag_00__CT_damage_last_5s`: coefficient `0.001364`, |coef| `0.001364`
- `lag_10__CT_place_DECON`: coefficient `-0.001364`, |coef| `0.001364`
- `lag_06__CT_place_RAFTERS`: coefficient `-0.001235`, |coef| `0.001235`
- `lag_09__CT_place_HELL`: coefficient `-0.001137`, |coef| `0.001137`
- `lag_08__CT_place_DECON`: coefficient `-0.001124`, |coef| `0.001124`
- `lag_03__CT5__duck_amount`: coefficient `-0.001109`, |coef| `0.001109`

## Top 10 utility ridge features

- `lag_01__T_A_site_active_infernos`: coefficient `0.000908` (raises CT win probability)
- `lag_00__T3__flash_duration`: coefficient `0.000881` (raises CT win probability)
- `lag_01__T_B_site_active_infernos`: coefficient `0.000858` (raises CT win probability)
- `lag_06__CT_B_site_active_infernos`: coefficient `-0.000817` (lowers CT win probability)
- `lag_12__CT_A_site_active_infernos`: coefficient `-0.000782` (lowers CT win probability)
- `lag_06__T_A_site_active_smokes`: coefficient `0.000774` (raises CT win probability)
- `lag_11__T_A_site_active_infernos`: coefficient `0.000723` (raises CT win probability)
- `lag_11__T_B_site_active_infernos`: coefficient `0.000685` (raises CT win probability)
- `lag_06__CT_A_site_active_infernos`: coefficient `-0.000673` (lowers CT win probability)
- `lag_11__T4__smoke`: coefficient `-0.000623` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.002341` (raises CT win probability)
- `lag_09__CT_place_DECON`: coefficient `-0.001730` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001657` (raises CT win probability)
- `lag_00__CT2__shots_fired`: coefficient `0.001638` (raises CT win probability)
- `lag_03__T_place_SQUEAKY`: coefficient `0.001553` (raises CT win probability)
- `lag_00__T_place_SQUEAKY`: coefficient `-0.001539` (lowers CT win probability)
- `lag_01__T_place_ROOF`: coefficient `-0.001500` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001468` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001464` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001364` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `140370`, seconds `19.00`, LSTM delta `+0.1541`

Top all feature movements:
- `lag_03__T_place_SQUEAKY`: contribution `+0.009670`
- `lag_00__T_place_SQUEAKY`: contribution `+0.009582`
- `lag_01__T_place_ROOF`: contribution `+0.008493`
- `lag_06__CT_place_RAFTERS`: contribution `+0.006601`
- `lag_00__CT_shots_fired_sum`: contribution `+0.006505`

Top utility-only movements:
- `lag_06__CT_B_site_active_infernos`: contribution `+0.002805`
- `lag_12__CT_A_site_active_infernos`: contribution `+0.002758`
- `lag_01__T_A_site_active_infernos`: contribution `+0.002702`
- `lag_01__T_B_site_active_infernos`: contribution `+0.002426`

### tick `141970`, seconds `44.00`, LSTM delta `+0.1069`

Top all feature movements:
- `lag_00__T_place_SECRET`: contribution `+0.005710`
- `lag_00__CT2__shots_fired`: contribution `+0.004886`
- `lag_00__CT_kills_last_3s`: contribution `+0.004783`
- `lag_01__CT_shots_fired_sum`: contribution `+0.004187`
- `lag_06__CT_place_MINI`: contribution `+0.003993`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `140402`, seconds `19.50`, LSTM delta `+0.0808`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.009757`
- `lag_04__T_place_SQUEAKY`: contribution `+0.005928`
- `lag_00__CT2__shots_fired`: contribution `+0.004886`
- `lag_00__T3__flash_duration`: contribution `+0.004254`
- `lag_03__CT5__duck_amount`: contribution `+0.004186`

Top utility-only movements:
- `lag_00__T3__flash_duration`: contribution `+0.004254`
- `lag_07__CT_B_site_active_infernos`: contribution `+0.001383`

### tick `142898`, seconds `58.50`, LSTM delta `+0.0741`

Top all feature movements:
- `lag_10__CT_place_DECON`: contribution `+0.021692`
- `lag_00__CT_shots_fired_sum`: contribution `-0.011384`
- `lag_14__CT_place_DECON`: contribution `+0.006716`
- `lag_00__CT_kills_last_3s`: contribution `+0.004783`
- `lag_00__kill_diff_last_3s`: contribution `+0.003525`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `142866`, seconds `58.00`, LSTM delta `+0.0728`

Top all feature movements:
- `lag_09__CT_place_DECON`: contribution `+0.027508`
- `lag_13__CT_place_DECON`: contribution `+0.013026`
- `lag_00__CT_shots_fired_sum`: contribution `+0.008131`
- `lag_03__CT5__duck_amount`: contribution `+0.004186`
- `lag_00__T1__duck_amount`: contribution `-0.003497`

Top utility-only movements:
- No utility movement among the top local contributors.
