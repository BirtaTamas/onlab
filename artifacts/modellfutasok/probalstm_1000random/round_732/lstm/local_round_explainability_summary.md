# Local Round Explainability

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-vitality-vs-faze-bo3-cXlexQJNK-6GX9ddkYcv53/vitality-vs-faze-m1-mirage.csv`
- round_num: `26`

## Largest probability jumps

- tick `217750`, seconds `77.50`, LSTM `0.5718`, delta `-0.3503`
- tick `217430`, seconds `72.50`, LSTM `0.9118`, delta `+0.1407`
- tick `218230`, seconds `85.00`, LSTM `0.0761`, delta `-0.1379`
- tick `216854`, seconds `63.50`, LSTM `0.7711`, delta `+0.1127`
- tick `216790`, seconds `62.50`, LSTM `0.6348`, delta `-0.0923`
- tick `217878`, seconds `79.50`, LSTM `0.4590`, delta `-0.0923`
- tick `216726`, seconds `61.50`, LSTM `0.7129`, delta `+0.0738`
- tick `213590`, seconds `12.50`, LSTM `0.5438`, delta `-0.0630`
- tick `217910`, seconds `80.00`, LSTM `0.3999`, delta `-0.0591`
- tick `217046`, seconds `66.50`, LSTM `0.7231`, delta `-0.0577`

## Top 15 local ridge features

- `lag_08__CT_place_LADDER`: coefficient `0.003769`, |coef| `0.003769`
- `lag_00__kill_diff_last_3s`: coefficient `0.002888`, |coef| `0.002888`
- `lag_00__T_kills_last_3s`: coefficient `-0.002360`, |coef| `0.002360`
- `lag_00__damage_diff_last_5s`: coefficient `0.002257`, |coef| `0.002257`
- `lag_10__CT_place_SHOP`: coefficient `0.002120`, |coef| `0.002120`
- `lag_14__CT1__flash_duration`: coefficient `-0.001993`, |coef| `0.001993`
- `lag_04__CT_place_SHOP`: coefficient `0.001891`, |coef| `0.001891`
- `lag_01__CT_place_SNIPERSNEST`: coefficient `0.001879`, |coef| `0.001879`
- `lag_04__CT_flash_duration_sum`: coefficient `0.001873`, |coef| `0.001873`
- `lag_04__CT1__flash_duration`: coefficient `0.001842`, |coef| `0.001842`
- `lag_04__CT_flashed_players`: coefficient `0.001821`, |coef| `0.001821`
- `lag_02__CT5__duck_amount`: coefficient `0.001751`, |coef| `0.001751`
- `lag_14__CT5__flash_duration`: coefficient `-0.001617`, |coef| `0.001617`
- `lag_06__T_place_JUNGLE`: coefficient `0.001392`, |coef| `0.001392`
- `lag_04__T_place_JUNGLE`: coefficient `0.001371`, |coef| `0.001371`

## Top 10 utility ridge features

- `lag_14__CT1__flash_duration`: coefficient `-0.001993` (lowers CT win probability)
- `lag_04__CT_flash_duration_sum`: coefficient `0.001873` (raises CT win probability)
- `lag_04__CT1__flash_duration`: coefficient `0.001842` (raises CT win probability)
- `lag_14__CT5__flash_duration`: coefficient `-0.001617` (lowers CT win probability)
- `lag_06__CT5__flash_duration`: coefficient `0.001238` (raises CT win probability)
- `lag_02__CT1__flash_duration`: coefficient `-0.001107` (lowers CT win probability)
- `lag_14__CT_flash_duration_sum`: coefficient `-0.001091` (lowers CT win probability)
- `lag_06__CT_flash_duration_sum`: coefficient `0.001078` (raises CT win probability)
- `lag_10__T1__molly`: coefficient `0.000845` (raises CT win probability)
- `lag_10__T1__smoke`: coefficient `0.000831` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_08__CT_place_LADDER`: coefficient `0.003769` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002888` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002360` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002257` (raises CT win probability)
- `lag_10__CT_place_SHOP`: coefficient `0.002120` (raises CT win probability)
- `lag_04__CT_place_SHOP`: coefficient `0.001891` (raises CT win probability)
- `lag_01__CT_place_SNIPERSNEST`: coefficient `0.001879` (raises CT win probability)
- `lag_04__CT_flashed_players`: coefficient `0.001821` (raises CT win probability)
- `lag_02__CT5__duck_amount`: coefficient `0.001751` (raises CT win probability)
- `lag_06__T_place_JUNGLE`: coefficient `0.001392` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `217750`, seconds `77.50`, LSTM delta `-0.3503`

Top all feature movements:
- `lag_08__CT_place_LADDER`: contribution `-0.039190`
- `lag_00__T_kills_last_3s`: contribution `-0.014955`
- `lag_00__kill_diff_last_3s`: contribution `-0.013902`
- `lag_14__CT1__flash_duration`: contribution `-0.010758`
- `lag_01__CT_place_SNIPERSNEST`: contribution `-0.010066`

Top utility-only movements:
- `lag_14__CT1__flash_duration`: contribution `-0.010758`
- `lag_04__CT1__flash_duration`: contribution `-0.009942`
- `lag_14__CT5__flash_duration`: contribution `-0.007556`
- `lag_06__CT5__flash_duration`: contribution `-0.005783`
- `lag_14__CT_flash_duration_sum`: contribution `-0.004996`

### tick `217430`, seconds `72.50`, LSTM delta `+0.1407`

Top all feature movements:
- `lag_10__CT_place_SHOP`: contribution `+0.010633`
- `lag_04__CT1__flash_duration`: contribution `+0.009942`
- `lag_04__CT_flash_duration_sum`: contribution `+0.008576`
- `lag_04__CT_flashed_players`: contribution `+0.007976`
- `lag_00__kill_diff_last_3s`: contribution `+0.006951`

Top utility-only movements:
- `lag_04__CT1__flash_duration`: contribution `+0.009942`
- `lag_04__CT_flash_duration_sum`: contribution `+0.008576`
- `lag_14__CT_A_site_active_infernos`: contribution `+0.002524`
- `lag_04__CT5__flash_duration`: contribution `+0.002327`

### tick `218230`, seconds `85.00`, LSTM delta `-0.1379`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.007478`
- `lag_00__kill_diff_last_3s`: contribution `-0.006951`
- `lag_15__T_kills_last_3s`: contribution `-0.005586`
- `lag_00__CT_shots_fired_sum`: contribution `-0.005201`
- `lag_15__kill_diff_last_3s`: contribution `-0.004061`

Top utility-only movements:
- `lag_13__CT1__flash_duration`: contribution `-0.003435`
- `lag_13__T5__flash_duration`: contribution `-0.003372`
- `lag_00__T5__flash_duration`: contribution `-0.002109`
- `lag_00__CT2__molly`: contribution `-0.001730`

### tick `216854`, seconds `63.50`, LSTM delta `+0.1127`

Top all feature movements:
- `lag_04__T_place_JUNGLE`: contribution `+0.017755`
- `lag_00__kill_diff_last_3s`: contribution `+0.006951`
- `lag_10__T_place_JUNGLE`: contribution `+0.004859`
- `lag_00__T_place_JUNGLE`: contribution `+0.004407`
- `lag_02__CT_place_JUNGLE`: contribution `+0.003839`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `216790`, seconds `62.50`, LSTM delta `-0.0923`

Top all feature movements:
- `lag_03__T_place_JUNGLE`: contribution `-0.012226`
- `lag_02__T_place_JUNGLE`: contribution `-0.009445`
- `lag_00__T_kills_last_3s`: contribution `-0.007478`
- `lag_00__kill_diff_last_3s`: contribution `-0.006951`
- `lag_08__T_place_JUNGLE`: contribution `-0.004173`

Top utility-only movements:
- No utility movement among the top local contributors.
