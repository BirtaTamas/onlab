# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-eternal-fire-vs-fluxo-bo3-Kqy3ohBVu1ANumI6Qdn26R/eternal-fire-vs-fluxo-m2-dust2.csv`
- round_num: `10`

## Largest probability jumps

- tick `74537`, seconds `100.00`, LSTM `0.3350`, delta `+0.2761`
- tick `69449`, seconds `20.50`, LSTM `0.1795`, delta `-0.2090`
- tick `74601`, seconds `101.00`, LSTM `0.0876`, delta `-0.1933`
- tick `72681`, seconds `71.00`, LSTM `0.0653`, delta `-0.1137`
- tick `73449`, seconds `83.00`, LSTM `0.2480`, delta `+0.0816`
- tick `73129`, seconds `78.00`, LSTM `0.0441`, delta `-0.0741`
- tick `73225`, seconds `79.50`, LSTM `0.0958`, delta `+0.0731`
- tick `73545`, seconds `84.50`, LSTM `0.2434`, delta `-0.0718`
- tick `73929`, seconds `90.50`, LSTM `0.2090`, delta `-0.0660`
- tick `71945`, seconds `59.50`, LSTM `0.2111`, delta `+0.0650`

## Top 15 local ridge features

- `lag_08__CT_flashes_last_5s`: coefficient `-0.003500`, |coef| `0.003500`
- `lag_00__T_place_ARAMP`: coefficient `-0.003401`, |coef| `0.003401`
- `lag_00__kill_diff_last_3s`: coefficient `0.002408`, |coef| `0.002408`
- `lag_05__CT5__duck_amount`: coefficient `0.002005`, |coef| `0.002005`
- `lag_00__T_kills_last_3s`: coefficient `-0.001846`, |coef| `0.001846`
- `lag_05__CT_duck_amount_mean`: coefficient `0.001742`, |coef| `0.001742`
- `lag_07__T_place_ARAMP`: coefficient `0.001716`, |coef| `0.001716`
- `lag_02__CT_place_SHORTSTAIRS`: coefficient `0.001678`, |coef| `0.001678`
- `lag_09__T_place_ARAMP`: coefficient `-0.001657`, |coef| `0.001657`
- `lag_05__T1__duck_amount`: coefficient `0.001652`, |coef| `0.001652`
- `lag_05__T_duck_amount_mean`: coefficient `0.001609`, |coef| `0.001609`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001532`, |coef| `0.001532`
- `lag_00__CT5__duck_amount`: coefficient `-0.001509`, |coef| `0.001509`
- `lag_10__CT4__flash_duration`: coefficient `-0.001493`, |coef| `0.001493`
- `lag_00__damage_diff_last_5s`: coefficient `0.001414`, |coef| `0.001414`

## Top 10 utility ridge features

- `lag_08__CT_flashes_last_5s`: coefficient `-0.003500` (lowers CT win probability)
- `lag_10__CT4__flash_duration`: coefficient `-0.001493` (lowers CT win probability)
- `lag_03__CT4__flash_duration`: coefficient `0.001388` (raises CT win probability)
- `lag_10__CT_flashes_last_5s`: coefficient `0.000993` (raises CT win probability)
- `lag_12__CT_flashes_last_5s`: coefficient `-0.000890` (lowers CT win probability)
- `lag_00__CT4__molly`: coefficient `0.000848` (raises CT win probability)
- `lag_10__CT_flash_duration_sum`: coefficient `-0.000829` (lowers CT win probability)
- `lag_11__CT_flashes_last_5s`: coefficient `-0.000796` (lowers CT win probability)
- `lag_03__CT_flash_duration_sum`: coefficient `0.000767` (raises CT win probability)
- `lag_00__CT4__smoke`: coefficient `0.000751` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_ARAMP`: coefficient `-0.003401` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002408` (raises CT win probability)
- `lag_05__CT5__duck_amount`: coefficient `0.002005` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001846` (lowers CT win probability)
- `lag_05__CT_duck_amount_mean`: coefficient `0.001742` (raises CT win probability)
- `lag_07__T_place_ARAMP`: coefficient `0.001716` (raises CT win probability)
- `lag_02__CT_place_SHORTSTAIRS`: coefficient `0.001678` (raises CT win probability)
- `lag_09__T_place_ARAMP`: coefficient `-0.001657` (lowers CT win probability)
- `lag_05__T1__duck_amount`: coefficient `0.001652` (raises CT win probability)
- `lag_05__T_duck_amount_mean`: coefficient `0.001609` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `74537`, seconds `100.00`, LSTM delta `+0.2761`

Top all feature movements:
- `lag_08__CT_flashes_last_5s`: contribution `+0.038483`
- `lag_00__T_place_ARAMP`: contribution `+0.030776`
- `lag_07__T_place_ARAMP`: contribution `+0.015527`
- `lag_05__T_duck_amount_mean`: contribution `+0.008811`
- `lag_05__CT5__duck_amount`: contribution `+0.007570`

Top utility-only movements:
- `lag_08__CT_flashes_last_5s`: contribution `+0.038483`

### tick `69449`, seconds `20.50`, LSTM delta `-0.2090`

Top all feature movements:
- `lag_10__CT4__flash_duration`: contribution `-0.009532`
- `lag_02__CT_place_SHORTSTAIRS`: contribution `-0.009355`
- `lag_03__CT4__flash_duration`: contribution `-0.007513`
- `lag_05__CT_place_SHORTSTAIRS`: contribution `-0.006655`
- `lag_05__CT_place_EXTENDEDA`: contribution `-0.006126`

Top utility-only movements:
- `lag_10__CT4__flash_duration`: contribution `-0.009532`
- `lag_03__CT4__flash_duration`: contribution `-0.007513`

### tick `74601`, seconds `101.00`, LSTM delta `-0.1933`

Top all feature movements:
- `lag_09__T_place_ARAMP`: contribution `-0.014989`
- `lag_10__CT_flashes_last_5s`: contribution `-0.010922`
- `lag_05__CT5__duck_amount`: contribution `-0.007570`
- `lag_00__T_kills_last_3s`: contribution `-0.005848`
- `lag_00__kill_diff_last_3s`: contribution `-0.005797`

Top utility-only movements:
- `lag_10__CT_flashes_last_5s`: contribution `-0.010922`

### tick `72681`, seconds `71.00`, LSTM delta `-0.1137`

Top all feature movements:
- `lag_10__T_place_SIDE`: contribution `-0.016362`
- `lag_09__T_place_ARAMP`: contribution `-0.014989`
- `lag_00__T_kills_last_3s`: contribution `-0.005848`
- `lag_00__kill_diff_last_3s`: contribution `-0.005797`
- `lag_07__CT_place_EXTENDEDA`: contribution `-0.005754`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `73449`, seconds `83.00`, LSTM delta `+0.0816`

Top all feature movements:
- `lag_00__T_place_ARAMP`: contribution `+0.030776`
- `lag_02__T_place_ARAMP`: contribution `-0.005776`
- `lag_00__CT1__duck_amount`: contribution `+0.005267`
- `lag_05__CT_duck_amount_mean`: contribution `+0.005215`
- `lag_06__CT5__duck_amount`: contribution `+0.003740`

Top utility-only movements:
- `lag_11__T_utility_damage_last_5s`: contribution `+0.001705`
