# Local Round Explainability

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-faze-vs-g2-bo3-ldI7_iFRuThMOXF8zIbBwX/faze-vs-g2-m1-inferno.csv`
- round_num: `7`

## Largest probability jumps

- tick `79104`, seconds `104.50`, LSTM `0.1537`, delta `-0.2739`
- tick `73568`, seconds `18.00`, LSTM `0.4547`, delta `+0.2395`
- tick `78016`, seconds `87.50`, LSTM `0.3722`, delta `-0.2270`
- tick `77472`, seconds `79.00`, LSTM `0.4631`, delta `-0.1835`
- tick `74240`, seconds `28.50`, LSTM `0.5596`, delta `+0.1637`
- tick `77952`, seconds `86.50`, LSTM `0.6192`, delta `+0.1619`
- tick `74080`, seconds `26.00`, LSTM `0.4201`, delta `-0.1367`
- tick `78208`, seconds `90.50`, LSTM `0.4447`, delta `+0.0987`
- tick `74112`, seconds `26.50`, LSTM `0.3516`, delta `-0.0685`
- tick `78144`, seconds `89.50`, LSTM `0.3199`, delta `-0.0682`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.006103`, |coef| `0.006103`
- `lag_00__T_kills_last_3s`: coefficient `-0.005362`, |coef| `0.005362`
- `lag_00__damage_diff_last_5s`: coefficient `0.003509`, |coef| `0.003509`
- `lag_00__T_damage_last_5s`: coefficient `-0.003451`, |coef| `0.003451`
- `lag_00__T3__is_scoped`: coefficient `0.003184`, |coef| `0.003184`
- `lag_00__CT_place_MIDDLE`: coefficient `0.002893`, |coef| `0.002893`
- `lag_13__CT_place_BANANA`: coefficient `0.002831`, |coef| `0.002831`
- `lag_00__CT2__alive`: coefficient `0.002701`, |coef| `0.002701`
- `lag_00__CT2__hp`: coefficient `0.002670`, |coef| `0.002670`
- `lag_00__CT2__armor`: coefficient `0.002528`, |coef| `0.002528`
- `lag_00__CT_kills_last_3s`: coefficient `0.002434`, |coef| `0.002434`
- `lag_13__CT_place_MIDDLE`: coefficient `-0.002431`, |coef| `0.002431`
- `lag_02__CT_place_ARCH`: coefficient `-0.002405`, |coef| `0.002405`
- `lag_06__bomb_events_last_5s`: coefficient `0.002307`, |coef| `0.002307`
- `lag_04__CT1__is_scoped`: coefficient `-0.002306`, |coef| `0.002306`

## Top 10 utility ridge features

- `lag_01__CT2__flash_duration`: coefficient `-0.001731` (lowers CT win probability)
- `lag_13__CT2__flash_duration`: coefficient `0.001639` (raises CT win probability)
- `lag_05__T3__flash_duration`: coefficient `-0.001485` (lowers CT win probability)
- `lag_12__T3__flash_duration`: coefficient `0.001106` (raises CT win probability)
- `lag_01__CT2__molly`: coefficient `-0.001005` (lowers CT win probability)
- `lag_08__T_B_site_active_infernos`: coefficient `0.000983` (raises CT win probability)
- `lag_13__CT_flash_duration_sum`: coefficient `0.000907` (raises CT win probability)
- `lag_00__T_A_site_active_smokes`: coefficient `0.000867` (raises CT win probability)
- `lag_14__T3__flash_duration`: coefficient `0.000858` (raises CT win probability)
- `lag_15__T2__flash`: coefficient `-0.000848` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.006103` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.005362` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003509` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.003451` (lowers CT win probability)
- `lag_00__T3__is_scoped`: coefficient `0.003184` (raises CT win probability)
- `lag_00__CT_place_MIDDLE`: coefficient `0.002893` (raises CT win probability)
- `lag_13__CT_place_BANANA`: coefficient `0.002831` (raises CT win probability)
- `lag_00__CT2__alive`: coefficient `0.002701` (raises CT win probability)
- `lag_00__CT2__hp`: coefficient `0.002670` (raises CT win probability)
- `lag_00__CT2__armor`: coefficient `0.002528` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `79104`, seconds `104.50`, LSTM delta `-0.2739`

Top all feature movements:
- `lag_00__T3__is_scoped`: contribution `-0.020423`
- `lag_00__T_kills_last_3s`: contribution `-0.016988`
- `lag_00__kill_diff_last_3s`: contribution `-0.014689`
- `lag_06__T_duck_amount_mean`: contribution `-0.013029`
- `lag_04__CT1__is_scoped`: contribution `-0.009876`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `73568`, seconds `18.00`, LSTM delta `+0.2395`

Top all feature movements:
- `lag_01__T_place_BALCONY`: contribution `+0.029420`
- `lag_00__kill_diff_last_3s`: contribution `+0.014689`
- `lag_01__CT2__flash_duration`: contribution `+0.012292`
- `lag_13__CT2__flash_duration`: contribution `+0.011640`
- `lag_05__T3__flash_duration`: contribution `+0.008138`

Top utility-only movements:
- `lag_01__CT2__flash_duration`: contribution `+0.012292`
- `lag_13__CT2__flash_duration`: contribution `+0.011640`
- `lag_05__T3__flash_duration`: contribution `+0.008138`
- `lag_12__T3__flash_duration`: contribution `+0.007708`
- `lag_13__CT_flash_duration_sum`: contribution `+0.003855`

### tick `78016`, seconds `87.50`, LSTM delta `-0.2270`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.016988`
- `lag_00__kill_diff_last_3s`: contribution `-0.014689`
- `lag_01__T_bomb_zone_count`: contribution `-0.009569`
- `lag_09__T3__is_scoped`: contribution `-0.009264`
- `lag_00__T_damage_last_5s`: contribution `-0.008109`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `77472`, seconds `79.00`, LSTM delta `-0.1835`

Top all feature movements:
- `lag_00__T3__is_scoped`: contribution `-0.020423`
- `lag_00__T_kills_last_3s`: contribution `-0.016988`
- `lag_00__kill_diff_last_3s`: contribution `-0.014689`
- `lag_00__CT1__duck_amount`: contribution `-0.007345`
- `lag_15__T4__duck_amount`: contribution `-0.007223`

Top utility-only movements:
- `lag_08__T_B_site_active_infernos`: contribution `-0.002779`

### tick `74240`, seconds `28.50`, LSTM delta `+0.1637`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.014689`
- `lag_05__CT_place_QUAD`: contribution `+0.012586`
- `lag_00__CT_shots_fired_sum`: contribution `+0.007295`
- `lag_14__T_flashed_players`: contribution `+0.007213`
- `lag_04__T3__duck_amount`: contribution `+0.007160`

Top utility-only movements:
- `lag_07__CT_B_site_active_infernos`: contribution `+0.002409`
