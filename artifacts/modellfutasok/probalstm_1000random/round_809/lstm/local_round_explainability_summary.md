# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m2-dust2.csv`
- round_num: `21`

## Largest probability jumps

- tick `194081`, seconds `63.00`, LSTM `0.2210`, delta `-0.2180`
- tick `194305`, seconds `66.50`, LSTM `0.1548`, delta `-0.2011`
- tick `194241`, seconds `65.50`, LSTM `0.3449`, delta `+0.0965`
- tick `194913`, seconds `76.00`, LSTM `0.0284`, delta `-0.0824`
- tick `190529`, seconds `7.50`, LSTM `0.4323`, delta `-0.0444`
- tick `194401`, seconds `68.00`, LSTM `0.1025`, delta `-0.0411`
- tick `191745`, seconds `26.50`, LSTM `0.4705`, delta `+0.0361`
- tick `192545`, seconds `39.00`, LSTM `0.3748`, delta `-0.0352`
- tick `190561`, seconds `8.00`, LSTM `0.3974`, delta `-0.0349`
- tick `191361`, seconds `20.50`, LSTM `0.4314`, delta `-0.0324`

## Top 15 local ridge features

- `lag_04__CT_place_EXTENDEDA`: coefficient `0.002789`, |coef| `0.002789`
- `lag_00__T_shots_fired_sum`: coefficient `-0.002487`, |coef| `0.002487`
- `lag_06__T_shots_fired_sum`: coefficient `0.002342`, |coef| `0.002342`
- `lag_00__T_kills_last_3s`: coefficient `-0.001698`, |coef| `0.001698`
- `lag_14__T_place_LONGA`: coefficient `-0.001630`, |coef| `0.001630`
- `lag_05__T_place_LOWERTUNNEL`: coefficient `0.001546`, |coef| `0.001546`
- `lag_00__kill_diff_last_3s`: coefficient `0.001528`, |coef| `0.001528`
- `lag_06__T1__shots_fired`: coefficient `0.001524`, |coef| `0.001524`
- `lag_02__CT_place_EXTENDEDA`: coefficient `-0.001522`, |coef| `0.001522`
- `lag_07__T_place_LONGA`: coefficient `-0.001326`, |coef| `0.001326`
- `lag_01__CT3__duck_amount`: coefficient `0.001262`, |coef| `0.001262`
- `lag_04__CT_place_UNDERA`: coefficient `-0.001241`, |coef| `0.001241`
- `lag_00__T1__shots_fired`: coefficient `-0.001222`, |coef| `0.001222`
- `lag_11__T_place_LONGA`: coefficient `-0.001213`, |coef| `0.001213`
- `lag_00__CT3__shots_fired`: coefficient `-0.001190`, |coef| `0.001190`

## Top 10 utility ridge features

- `lag_00__CT3__molly`: coefficient `0.001111` (raises CT win probability)
- `lag_00__CT1__smoke`: coefficient `0.000821` (raises CT win probability)
- `lag_10__T3__molly`: coefficient `0.000812` (raises CT win probability)
- `lag_13__T3__smoke`: coefficient `0.000781` (raises CT win probability)
- `lag_00__CT_molly_inv`: coefficient `0.000737` (raises CT win probability)
- `lag_00__CT3__utility_total`: coefficient `0.000655` (raises CT win probability)
- `lag_00__CT1__utility_total`: coefficient `0.000602` (raises CT win probability)
- `lag_07__T_active_infernos`: coefficient `-0.000579` (lowers CT win probability)
- `lag_00__CT_smoke_inv`: coefficient `0.000571` (raises CT win probability)
- `lag_00__CT_utility_inv`: coefficient `0.000568` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_04__CT_place_EXTENDEDA`: coefficient `0.002789` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.002487` (lowers CT win probability)
- `lag_06__T_shots_fired_sum`: coefficient `0.002342` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001698` (lowers CT win probability)
- `lag_14__T_place_LONGA`: coefficient `-0.001630` (lowers CT win probability)
- `lag_05__T_place_LOWERTUNNEL`: coefficient `0.001546` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001528` (raises CT win probability)
- `lag_06__T1__shots_fired`: coefficient `0.001524` (raises CT win probability)
- `lag_02__CT_place_EXTENDEDA`: coefficient `-0.001522` (lowers CT win probability)
- `lag_07__T_place_LONGA`: coefficient `-0.001326` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `194081`, seconds `63.00`, LSTM delta `-0.2180`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.016783`
- `lag_04__CT_place_EXTENDEDA`: contribution `-0.015659`
- `lag_02__CT_place_EXTENDEDA`: contribution `-0.008545`
- `lag_14__T_place_LONGA`: contribution `-0.006946`
- `lag_05__T_place_LOWERTUNNEL`: contribution `-0.006684`

Top utility-only movements:
- `lag_00__CT3__molly`: contribution `-0.002744`

### tick `194305`, seconds `66.50`, LSTM delta `-0.2011`

Top all feature movements:
- `lag_06__T_shots_fired_sum`: contribution `-0.029849`
- `lag_04__CT_place_EXTENDEDA`: contribution `-0.015659`
- `lag_06__T1__shots_fired`: contribution `-0.011838`
- `lag_14__T_place_LONGA`: contribution `-0.006946`
- `lag_11__CT_place_EXTENDEDA`: contribution `-0.005412`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `194241`, seconds `65.50`, LSTM delta `+0.0965`

Top all feature movements:
- `lag_04__T_shots_fired_sum`: contribution `+0.014273`
- `lag_06__T_shots_fired_sum`: contribution `+0.008779`
- `lag_02__CT_place_EXTENDEDA`: contribution `+0.008545`
- `lag_04__T1__shots_fired`: contribution `+0.005989`
- `lag_06__T1__shots_fired`: contribution `+0.004553`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `194913`, seconds `76.00`, LSTM delta `-0.0824`

Top all feature movements:
- `lag_08__T_place_SIDE`: contribution `-0.007926`
- `lag_08__T_place_ARAMP`: contribution `-0.007404`
- `lag_10__T_place_ARAMP`: contribution `-0.006033`
- `lag_00__T_kills_last_3s`: contribution `-0.005380`
- `lag_00__T_shots_fired_sum`: contribution `-0.003729`

Top utility-only movements:
- `lag_15__T4__flash_duration`: contribution `-0.001403`

### tick `190529`, seconds `7.50`, LSTM delta `-0.0444`

Top all feature movements:
- `lag_00__CT_place_BDOORS`: contribution `-0.005439`
- `lag_01__CT_place_BDOORS`: contribution `-0.004792`
- `lag_00__CT1__flash_duration`: contribution `-0.003036`
- `lag_15__CT_place_CTSPAWN`: contribution `-0.002719`
- `lag_00__CT_place_MIDDOORS`: contribution `+0.002556`

Top utility-only movements:
- `lag_00__CT1__flash_duration`: contribution `-0.003036`
- `lag_00__T5__flash_duration`: contribution `-0.001419`
