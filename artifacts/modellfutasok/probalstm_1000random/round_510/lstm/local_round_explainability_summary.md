# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-3dmax-vs-betboom-anubis-9yOMu3EhAmKzkIxUzvijXH/3dmax-vs-betboom-anubis.csv`
- round_num: `9`

## Largest probability jumps

- tick `72066`, seconds `63.00`, LSTM `0.7774`, delta `+0.2122`
- tick `72514`, seconds `70.00`, LSTM `0.9203`, delta `+0.1371`
- tick `71938`, seconds `61.00`, LSTM `0.5173`, delta `+0.1073`
- tick `69890`, seconds `29.00`, LSTM `0.3712`, delta `-0.0668`
- tick `72258`, seconds `66.00`, LSTM `0.8121`, delta `-0.0462`
- tick `69922`, seconds `29.50`, LSTM `0.4158`, delta `+0.0446`
- tick `73026`, seconds `78.00`, LSTM `0.9392`, delta `+0.0430`
- tick `69858`, seconds `28.50`, LSTM `0.4380`, delta `+0.0399`
- tick `68258`, seconds `3.50`, LSTM `0.3908`, delta `-0.0353`
- tick `69826`, seconds `28.00`, LSTM `0.3980`, delta `+0.0352`

## Top 15 local ridge features

- `lag_00__CT_place_CTSIDEUPPER`: coefficient `-0.002819`, |coef| `0.002819`
- `lag_00__CT_kills_last_3s`: coefficient `0.002388`, |coef| `0.002388`
- `lag_14__CT_place_BACKOFB`: coefficient `-0.002312`, |coef| `0.002312`
- `lag_00__damage_diff_last_5s`: coefficient `0.002266`, |coef| `0.002266`
- `lag_00__CT_damage_last_5s`: coefficient `0.002160`, |coef| `0.002160`
- `lag_00__kill_diff_last_3s`: coefficient `0.002023`, |coef| `0.002023`
- `lag_12__T4__is_scoped`: coefficient `0.001598`, |coef| `0.001598`
- `lag_04__CT4__is_walking`: coefficient `0.001541`, |coef| `0.001541`
- `lag_04__CT_kills_last_3s`: coefficient `0.001484`, |coef| `0.001484`
- `lag_13__CT4__duck_amount`: coefficient `-0.001476`, |coef| `0.001476`
- `lag_00__T5__alive`: coefficient `-0.001451`, |coef| `0.001451`
- `lag_09__T3__is_walking`: coefficient `0.001428`, |coef| `0.001428`
- `lag_00__T5__hp`: coefficient `-0.001426`, |coef| `0.001426`
- `lag_10__CT_place_BACKOFB`: coefficient `-0.001388`, |coef| `0.001388`
- `lag_03__T4__is_scoped`: coefficient `-0.001372`, |coef| `0.001372`

## Top 10 utility ridge features

- `lag_07__T_B_site_active_infernos`: coefficient `0.001262` (raises CT win probability)
- `lag_15__T4__smoke`: coefficient `-0.001126` (lowers CT win probability)
- `lag_09__T3__molly`: coefficient `-0.001110` (lowers CT win probability)
- `lag_09__T_B_site_active_infernos`: coefficient `0.001105` (raises CT win probability)
- `lag_07__T_B_site_active_smokes`: coefficient `0.001038` (raises CT win probability)
- `lag_06__T_B_site_active_smokes`: coefficient `0.000932` (raises CT win probability)
- `lag_13__T_B_site_active_infernos`: coefficient `0.000830` (raises CT win probability)
- `lag_00__T5__flash`: coefficient `-0.000820` (lowers CT win probability)
- `lag_04__CT_B_site_active_smokes`: coefficient `0.000817` (raises CT win probability)
- `lag_07__CT1__smoke`: coefficient `-0.000813` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_CTSIDEUPPER`: coefficient `-0.002819` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002388` (raises CT win probability)
- `lag_14__CT_place_BACKOFB`: coefficient `-0.002312` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002266` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002160` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002023` (raises CT win probability)
- `lag_12__T4__is_scoped`: coefficient `0.001598` (raises CT win probability)
- `lag_04__CT4__is_walking`: coefficient `0.001541` (raises CT win probability)
- `lag_04__CT_kills_last_3s`: coefficient `0.001484` (raises CT win probability)
- `lag_13__CT4__duck_amount`: coefficient `-0.001476` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `72066`, seconds `63.00`, LSTM delta `+0.2122`

Top all feature movements:
- `lag_14__CT_place_BACKOFB`: contribution `+0.013198`
- `lag_12__T4__is_scoped`: contribution `+0.007422`
- `lag_00__CT_kills_last_3s`: contribution `+0.006894`
- `lag_03__T4__is_scoped`: contribution `+0.006374`
- `lag_04__CT3__is_scoped`: contribution `+0.005838`

Top utility-only movements:
- `lag_07__T_B_site_active_infernos`: contribution `+0.003567`

### tick `72514`, seconds `70.00`, LSTM delta `+0.1371`

Top all feature movements:
- `lag_11__CT_place_MAIN`: contribution `+0.007729`
- `lag_00__CT_kills_last_3s`: contribution `+0.006894`
- `lag_01__CT_place_BACKOFB`: contribution `+0.006877`
- `lag_00__damage_diff_last_5s`: contribution `+0.005062`
- `lag_08__CT_place_CANAL`: contribution `+0.004896`

Top utility-only movements:
- `lag_00__T1__flash_duration`: contribution `+0.003576`
- `lag_07__T_B_site_active_infernos`: contribution `-0.003567`
- `lag_01__T1__flash_duration`: contribution `+0.003421`
- `lag_13__T_B_site_active_infernos`: contribution `-0.002347`

### tick `71938`, seconds `61.00`, LSTM delta `+0.1073`

Top all feature movements:
- `lag_10__CT_place_BACKOFB`: contribution `+0.007924`
- `lag_00__CT_kills_last_3s`: contribution `+0.006894`
- `lag_00__damage_diff_last_5s`: contribution `+0.005113`
- `lag_00__kill_diff_last_3s`: contribution `+0.004868`
- `lag_00__CT_damage_last_5s`: contribution `+0.004709`

Top utility-only movements:
- `lag_09__T_B_site_active_infernos`: contribution `+0.003125`
- `lag_03__T_B_site_active_infernos`: contribution `+0.002244`

### tick `69890`, seconds `29.00`, LSTM delta `-0.0668`

Top all feature movements:
- `lag_03__T_shots_fired_sum`: contribution `-0.006449`
- `lag_03__T5__shots_fired`: contribution `-0.004704`
- `lag_06__T5__duck_amount`: contribution `-0.004262`
- `lag_00__CT5__is_scoped`: contribution `+0.003537`
- `lag_01__CT_place_HEAVEN`: contribution `-0.003354`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `72258`, seconds `66.00`, LSTM delta `-0.0462`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `-0.006894`
- `lag_04__CT3__is_scoped`: contribution `-0.005838`
- `lag_13__CT4__duck_amount`: contribution `-0.005421`
- `lag_00__damage_diff_last_5s`: contribution `-0.005113`
- `lag_00__kill_diff_last_3s`: contribution `-0.004868`

Top utility-only movements:
- `lag_13__T_B_site_active_infernos`: contribution `+0.002347`
- `lag_13__T_active_infernos`: contribution `+0.001381`
