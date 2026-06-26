# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-og-vs-nrg-bo3-GH6ZBFOA9sfdeCxgnhHN9f/og-vs-nrg-m2-nuke.csv`
- round_num: `15`

## Largest probability jumps

- tick `121008`, seconds `16.00`, LSTM `0.9383`, delta `+0.0616`
- tick `122800`, seconds `44.00`, LSTM `0.9155`, delta `+0.0604`
- tick `122576`, seconds `40.50`, LSTM `0.9012`, delta `-0.0553`
- tick `122288`, seconds `36.00`, LSTM `0.9426`, delta `+0.0396`
- tick `122608`, seconds `41.00`, LSTM `0.8692`, delta `-0.0320`
- tick `120656`, seconds `10.50`, LSTM `0.8934`, delta `-0.0289`
- tick `121136`, seconds `18.00`, LSTM `0.9213`, delta `-0.0288`
- tick `120016`, seconds `0.50`, LSTM `0.9247`, delta `+0.0234`
- tick `122640`, seconds `41.50`, LSTM `0.8466`, delta `-0.0227`
- tick `122672`, seconds `42.00`, LSTM `0.8245`, delta `-0.0221`

## Top 15 local ridge features

- `lag_14__CT_place_ADMIN`: coefficient `-0.000926`, |coef| `0.000926`
- `lag_15__CT_place_ADMIN`: coefficient `-0.000925`, |coef| `0.000925`
- `lag_00__kill_diff_last_3s`: coefficient `0.000817`, |coef| `0.000817`
- `lag_00__damage_diff_last_5s`: coefficient `0.000741`, |coef| `0.000741`
- `lag_11__CT_place_HELL`: coefficient `-0.000692`, |coef| `0.000692`
- `lag_13__CT_place_ADMIN`: coefficient `-0.000689`, |coef| `0.000689`
- `lag_00__T_place_SQUEAKY`: coefficient `0.000645`, |coef| `0.000645`
- `lag_00__CT_place_HEAVEN`: coefficient `-0.000639`, |coef| `0.000639`
- `lag_00__CT_shots_fired_sum`: coefficient `0.000619`, |coef| `0.000619`
- `lag_01__T_kills_last_3s`: coefficient `-0.000605`, |coef| `0.000605`
- `lag_00__CT1__duck_amount`: coefficient `0.000583`, |coef| `0.000583`
- `lag_00__CT_kills_last_3s`: coefficient `0.000567`, |coef| `0.000567`
- `lag_08__CT_place_ADMIN`: coefficient `0.000556`, |coef| `0.000556`
- `lag_13__CT1__duck_amount`: coefficient `-0.000516`, |coef| `0.000516`
- `lag_09__T1__is_walking`: coefficient `-0.000506`, |coef| `0.000506`

## Top 10 utility ridge features

- `lag_00__CT_A_site_active_infernos`: coefficient `-0.000467` (lowers CT win probability)
- `lag_00__CT_B_site_active_infernos`: coefficient `-0.000454` (lowers CT win probability)
- `lag_00__CT_active_infernos`: coefficient `-0.000318` (lowers CT win probability)
- `lag_03__T5__flash_duration`: coefficient `-0.000311` (lowers CT win probability)
- `lag_12__CT_B_site_active_smokes`: coefficient `-0.000306` (lowers CT win probability)
- `lag_12__CT_A_site_active_smokes`: coefficient `-0.000297` (lowers CT win probability)
- `lag_11__CT_A_site_active_infernos`: coefficient `0.000250` (raises CT win probability)
- `lag_11__CT_B_site_active_infernos`: coefficient `0.000243` (raises CT win probability)
- `lag_06__CT_A_site_active_infernos`: coefficient `0.000222` (raises CT win probability)
- `lag_06__CT_B_site_active_infernos`: coefficient `0.000216` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_14__CT_place_ADMIN`: coefficient `-0.000926` (lowers CT win probability)
- `lag_15__CT_place_ADMIN`: coefficient `-0.000925` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000817` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000741` (raises CT win probability)
- `lag_11__CT_place_HELL`: coefficient `-0.000692` (lowers CT win probability)
- `lag_13__CT_place_ADMIN`: coefficient `-0.000689` (lowers CT win probability)
- `lag_00__T_place_SQUEAKY`: coefficient `0.000645` (raises CT win probability)
- `lag_00__CT_place_HEAVEN`: coefficient `-0.000639` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.000619` (raises CT win probability)
- `lag_01__T_kills_last_3s`: coefficient `-0.000605` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `121008`, seconds `16.00`, LSTM delta `+0.0616`

Top all feature movements:
- `lag_00__CT_place_HEAVEN`: contribution `+0.003448`
- `lag_00__CT_A_site_active_infernos`: contribution `+0.003293`
- `lag_00__CT_B_site_active_infernos`: contribution `+0.003121`
- `lag_00__T_place_CONTROL`: contribution `+0.002763`
- `lag_00__CT_place_HUTROOF`: contribution `+0.002585`

Top utility-only movements:
- `lag_00__CT_A_site_active_infernos`: contribution `+0.003293`
- `lag_00__CT_B_site_active_infernos`: contribution `+0.003121`
- `lag_03__T5__flash_duration`: contribution `+0.001882`
- `lag_11__CT_A_site_active_infernos`: contribution `+0.001765`
- `lag_11__CT_B_site_active_infernos`: contribution `+0.001673`

### tick `122800`, seconds `44.00`, LSTM delta `+0.0604`

Top all feature movements:
- `lag_15__CT_place_ADMIN`: contribution `+0.006429`
- `lag_00__T_place_SQUEAKY`: contribution `+0.004015`
- `lag_12__CT_place_HELL`: contribution `+0.002473`
- `lag_15__CT_place_HELL`: contribution `+0.002010`
- `lag_13__CT1__duck_amount`: contribution `+0.001970`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `122576`, seconds `40.50`, LSTM delta `-0.0553`

Top all feature movements:
- `lag_13__CT_place_ADMIN`: contribution `-0.004788`
- `lag_08__CT_place_ADMIN`: contribution `-0.003864`
- `lag_00__CT1__duck_amount`: contribution `-0.002225`
- `lag_13__CT1__duck_amount`: contribution `-0.001970`
- `lag_00__kill_diff_last_3s`: contribution `-0.001967`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `122288`, seconds `36.00`, LSTM delta `+0.0396`

Top all feature movements:
- `lag_00__damage_diff_last_5s`: contribution `+0.002323`
- `lag_00__CT1__duck_amount`: contribution `+0.002225`
- `lag_00__CT_shots_fired_sum`: contribution `+0.002150`
- `lag_00__kill_diff_last_3s`: contribution `+0.001967`
- `lag_00__CT_kills_last_3s`: contribution `+0.001636`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `122608`, seconds `41.00`, LSTM delta `-0.0320`

Top all feature movements:
- `lag_14__CT_place_ADMIN`: contribution `-0.006431`
- `lag_09__CT_place_ADMIN`: contribution `-0.003271`
- `lag_13__CT1__duck_amount`: contribution `+0.001970`
- `lag_01__T_kills_last_3s`: contribution `-0.001915`
- `lag_06__CT_place_HELL`: contribution `-0.001867`

Top utility-only movements:
- No utility movement among the top local contributors.
