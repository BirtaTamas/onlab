# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-lynn-vision-bo3-6KVULP2-Gxo12lI67V9ZfV/chinggis-warriors-vs-lynn-vision-m3-ancient.csv`
- round_num: `9`

## Largest probability jumps

- tick `70726`, seconds `91.50`, LSTM `0.6474`, delta `+0.2649`
- tick `70694`, seconds `91.00`, LSTM `0.3825`, delta `-0.2382`
- tick `66566`, seconds `26.50`, LSTM `0.7550`, delta `+0.1642`
- tick `66054`, seconds `18.50`, LSTM `0.5879`, delta `-0.1436`
- tick `71206`, seconds `99.00`, LSTM `0.6608`, delta `+0.1431`
- tick `65990`, seconds `17.50`, LSTM `0.7472`, delta `+0.1393`
- tick `69382`, seconds `70.50`, LSTM `0.6928`, delta `-0.1148`
- tick `68774`, seconds `61.00`, LSTM `0.9355`, delta `+0.0941`
- tick `71270`, seconds `100.00`, LSTM `0.5882`, delta `-0.0828`
- tick `68838`, seconds `62.00`, LSTM `0.8709`, delta `-0.0653`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003672`, |coef| `0.003672`
- `lag_00__T_kills_last_3s`: coefficient `-0.003032`, |coef| `0.003032`
- `lag_00__T_shots_fired_sum`: coefficient `-0.002911`, |coef| `0.002911`
- `lag_00__damage_diff_last_5s`: coefficient `0.002752`, |coef| `0.002752`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002075`, |coef| `0.002075`
- `lag_00__T3__duck_amount`: coefficient `0.001876`, |coef| `0.001876`
- `lag_14__T5__duck_amount`: coefficient `0.001725`, |coef| `0.001725`
- `lag_00__CT_kills_last_3s`: coefficient `0.001642`, |coef| `0.001642`
- `lag_04__T_place_CTSPAWN`: coefficient `-0.001621`, |coef| `0.001621`
- `lag_07__CT1__duck_amount`: coefficient `-0.001602`, |coef| `0.001602`
- `lag_00__CT_A_site_active_infernos`: coefficient `0.001552`, |coef| `0.001552`
- `lag_13__T_place_HOUSE`: coefficient `-0.001552`, |coef| `0.001552`
- `lag_12__T5__duck_amount`: coefficient `0.001514`, |coef| `0.001514`
- `lag_00__T_place_HOUSE`: coefficient `-0.001466`, |coef| `0.001466`
- `lag_00__T_damage_last_5s`: coefficient `-0.001400`, |coef| `0.001400`

## Top 10 utility ridge features

- `lag_00__CT_A_site_active_infernos`: coefficient `0.001552` (raises CT win probability)
- `lag_04__T5__flash_duration`: coefficient `0.001394` (raises CT win probability)
- `lag_02__T1__flash_duration`: coefficient `-0.001341` (lowers CT win probability)
- `lag_15__CT_he_last_5s`: coefficient `-0.001208` (lowers CT win probability)
- `lag_08__T5__flash_duration`: coefficient `-0.001193` (lowers CT win probability)
- `lag_00__CT_B_site_active_infernos`: coefficient `0.001158` (raises CT win probability)
- `lag_00__T5__flash_duration`: coefficient `-0.001146` (lowers CT win probability)
- `lag_15__T5__flash_duration`: coefficient `-0.001124` (lowers CT win probability)
- `lag_00__CT1__molly`: coefficient `0.001119` (raises CT win probability)
- `lag_14__CT4__flash_duration`: coefficient `0.001118` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003672` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003032` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.002911` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002752` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002075` (raises CT win probability)
- `lag_00__T3__duck_amount`: coefficient `0.001876` (raises CT win probability)
- `lag_14__T5__duck_amount`: coefficient `0.001725` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001642` (raises CT win probability)
- `lag_04__T_place_CTSPAWN`: coefficient `-0.001621` (lowers CT win probability)
- `lag_07__CT1__duck_amount`: coefficient `-0.001602` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `70726`, seconds `91.50`, LSTM delta `+0.2649`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.010911`
- `lag_00__kill_diff_last_3s`: contribution `+0.008839`
- `lag_08__T_duck_amount_mean`: contribution `+0.006910`
- `lag_00__T_place_HOUSE`: contribution `+0.006448`
- `lag_07__CT1__duck_amount`: contribution `+0.006112`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `70694`, seconds `91.00`, LSTM delta `-0.2382`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.010911`
- `lag_00__T_kills_last_3s`: contribution `-0.009606`
- `lag_00__kill_diff_last_3s`: contribution `-0.008839`
- `lag_07__T_duck_amount_mean`: contribution `-0.006642`
- `lag_14__T5__duck_amount`: contribution `-0.006549`

Top utility-only movements:
- `lag_00__CT1__molly`: contribution `-0.002786`

### tick `66566`, seconds `26.50`, LSTM delta `+0.1642`

Top all feature movements:
- `lag_02__T1__flash_duration`: contribution `+0.008877`
- `lag_00__kill_diff_last_3s`: contribution `+0.008839`
- `lag_15__T_shots_fired_sum`: contribution `+0.007048`
- `lag_14__CT4__flash_duration`: contribution `+0.006953`
- `lag_15__T_place_TUNNEL`: contribution `+0.005852`

Top utility-only movements:
- `lag_02__T1__flash_duration`: contribution `+0.008877`
- `lag_14__CT4__flash_duration`: contribution `+0.006953`
- `lag_14__CT1__flash_duration`: contribution `+0.004796`
- `lag_06__CT1__flash_duration`: contribution `+0.003637`
- `lag_13__T5__flash_duration`: contribution `-0.002690`

### tick `66054`, seconds `18.50`, LSTM delta `-0.1436`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.010911`
- `lag_00__T_kills_last_3s`: contribution `-0.009606`
- `lag_00__kill_diff_last_3s`: contribution `-0.008839`
- `lag_08__T5__flash_duration`: contribution `-0.006551`
- `lag_01__CT_shots_fired_sum`: contribution `-0.006427`

Top utility-only movements:
- `lag_08__T5__flash_duration`: contribution `-0.006551`
- `lag_06__T1__flash_duration`: contribution `-0.005440`
- `lag_00__CT4__flash_duration`: contribution `-0.004839`
- `lag_15__CT5__flash_duration`: contribution `-0.003923`

### tick `71206`, seconds `99.00`, LSTM delta `+0.1431`

Top all feature movements:
- `lag_00__T5__flash_duration`: contribution `+0.008620`
- `lag_13__T5__flash_duration`: contribution `+0.005860`
- `lag_00__CT_A_site_active_infernos`: contribution `+0.005478`
- `lag_15__T_place_HOUSE`: contribution `+0.004527`
- `lag_15__T_shots_fired_sum`: contribution `+0.004405`

Top utility-only movements:
- `lag_00__T5__flash_duration`: contribution `+0.008620`
- `lag_13__T5__flash_duration`: contribution `+0.005860`
- `lag_00__CT_A_site_active_infernos`: contribution `+0.005478`
- `lag_00__CT_B_site_active_infernos`: contribution `+0.003980`
- `lag_09__CT_B_site_active_infernos`: contribution `+0.002233`
