# Local Round Explainability

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-vitality-vs-g2-bo5-ieXHvClzA7f_aJ_85fPFqK/vitality-vs-g2-m5-train.csv`
- round_num: `15`

## Largest probability jumps

- tick `132373`, seconds `60.50`, LSTM `0.0571`, delta `-0.1791`
- tick `132309`, seconds `59.50`, LSTM `0.2415`, delta `-0.1416`
- tick `131285`, seconds `43.50`, LSTM `0.2789`, delta `+0.1358`
- tick `131605`, seconds `48.50`, LSTM `0.3610`, delta `-0.0685`
- tick `131573`, seconds `48.00`, LSTM `0.4295`, delta `+0.0543`
- tick `128533`, seconds `0.50`, LSTM `0.0287`, delta `-0.0492`
- tick `132597`, seconds `64.00`, LSTM `0.0300`, delta `-0.0492`
- tick `131317`, seconds `44.00`, LSTM `0.3276`, delta `+0.0487`
- tick `132085`, seconds `56.00`, LSTM `0.4294`, delta `+0.0481`
- tick `132053`, seconds `55.50`, LSTM `0.3813`, delta `+0.0478`

## Top 15 local ridge features

- `lag_15__CT_place_ELECTRICALBOX`: coefficient `0.002767`, |coef| `0.002767`
- `lag_01__CT3__flash_duration`: coefficient `-0.001776`, |coef| `0.001776`
- `lag_15__CT_place_BACKOFB`: coefficient `-0.001602`, |coef| `0.001602`
- `lag_03__CT3__flash_duration`: coefficient `-0.001544`, |coef| `0.001544`
- `lag_14__CT_place_ELECTRICALBOX`: coefficient `0.001528`, |coef| `0.001528`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001483`, |coef| `0.001483`
- `lag_13__CT_place_BACKOFB`: coefficient `-0.001454`, |coef| `0.001454`
- `lag_00__damage_diff_last_5s`: coefficient `0.001401`, |coef| `0.001401`
- `lag_00__kill_diff_last_3s`: coefficient `0.001395`, |coef| `0.001395`
- `lag_07__T5__flash_duration`: coefficient `0.001360`, |coef| `0.001360`
- `lag_09__T5__flash_duration`: coefficient `0.001349`, |coef| `0.001349`
- `lag_03__CT_burning_players`: coefficient `-0.001337`, |coef| `0.001337`
- `lag_02__T_place_TMAIN`: coefficient `0.001313`, |coef| `0.001313`
- `lag_02__CT3__duck_amount`: coefficient `-0.001234`, |coef| `0.001234`
- `lag_00__CT4__is_walking`: coefficient `-0.001228`, |coef| `0.001228`

## Top 10 utility ridge features

- `lag_01__CT3__flash_duration`: coefficient `-0.001776` (lowers CT win probability)
- `lag_03__CT3__flash_duration`: coefficient `-0.001544` (lowers CT win probability)
- `lag_07__T5__flash_duration`: coefficient `0.001360` (raises CT win probability)
- `lag_09__T5__flash_duration`: coefficient `0.001349` (raises CT win probability)
- `lag_03__CT_flash_duration_sum`: coefficient `-0.001095` (lowers CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `0.001068` (raises CT win probability)
- `lag_14__T_A_site_active_infernos`: coefficient `0.001065` (raises CT win probability)
- `lag_01__CT_flash_duration_sum`: coefficient `-0.001041` (lowers CT win probability)
- `lag_11__CT1__flash_duration`: coefficient `0.001039` (raises CT win probability)
- `lag_13__CT1__flash_duration`: coefficient `0.000987` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_15__CT_place_ELECTRICALBOX`: coefficient `0.002767` (raises CT win probability)
- `lag_15__CT_place_BACKOFB`: coefficient `-0.001602` (lowers CT win probability)
- `lag_14__CT_place_ELECTRICALBOX`: coefficient `0.001528` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001483` (lowers CT win probability)
- `lag_13__CT_place_BACKOFB`: coefficient `-0.001454` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001401` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001395` (raises CT win probability)
- `lag_03__CT_burning_players`: coefficient `-0.001337` (lowers CT win probability)
- `lag_02__T_place_TMAIN`: coefficient `0.001313` (raises CT win probability)
- `lag_02__CT3__duck_amount`: coefficient `-0.001234` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `132373`, seconds `60.50`, LSTM delta `-0.1791`

Top all feature movements:
- `lag_03__CT3__flash_duration`: contribution `-0.010370`
- `lag_15__CT_place_BACKOFB`: contribution `-0.009144`
- `lag_09__T5__flash_duration`: contribution `-0.009006`
- `lag_00__T_shots_fired_sum`: contribution `-0.005558`
- `lag_02__T_place_TMAIN`: contribution `-0.005093`

Top utility-only movements:
- `lag_03__CT3__flash_duration`: contribution `-0.010370`
- `lag_09__T5__flash_duration`: contribution `-0.009006`
- `lag_13__CT1__flash_duration`: contribution `-0.004854`
- `lag_03__CT_flash_duration_sum`: contribution `-0.004633`
- `lag_14__T_A_site_active_infernos`: contribution `-0.003169`

### tick `132309`, seconds `59.50`, LSTM delta `-0.1416`

Top all feature movements:
- `lag_01__CT3__flash_duration`: contribution `-0.011933`
- `lag_07__T5__flash_duration`: contribution `-0.009080`
- `lag_13__CT_place_BACKOFB`: contribution `-0.008300`
- `lag_01__CT_flashed_players`: contribution `-0.005109`
- `lag_11__CT1__flash_duration`: contribution `-0.005109`

Top utility-only movements:
- `lag_01__CT3__flash_duration`: contribution `-0.011933`
- `lag_07__T5__flash_duration`: contribution `-0.009080`
- `lag_11__CT1__flash_duration`: contribution `-0.005109`
- `lag_01__CT_flash_duration_sum`: contribution `-0.004405`
- `lag_04__T_A_site_active_infernos`: contribution `-0.002412`

### tick `131285`, seconds `43.50`, LSTM delta `+0.1358`

Top all feature movements:
- `lag_15__CT_place_ELECTRICALBOX`: contribution `+0.032168`
- `lag_03__T_place_LONGDOG`: contribution `+0.003612`
- `lag_05__bomb_events_last_5s`: contribution `+0.003524`
- `lag_11__CT4__duck_amount`: contribution `+0.003407`
- `lag_00__kill_diff_last_3s`: contribution `+0.003357`

Top utility-only movements:
- `lag_15__T_A_site_active_infernos`: contribution `+0.002112`

### tick `131605`, seconds `48.50`, LSTM delta `-0.0685`

Top all feature movements:
- `lag_09__CT_place_ELECTRICALBOX`: contribution `-0.009578`
- `lag_01__CT3__flash_duration`: contribution `-0.009130`
- `lag_01__CT_flashed_players`: contribution `-0.007663`
- `lag_01__CT_flash_duration_sum`: contribution `-0.005473`
- `lag_10__T_place_LONGDOG`: contribution `-0.005048`

Top utility-only movements:
- `lag_01__CT3__flash_duration`: contribution `-0.009130`
- `lag_01__CT_flash_duration_sum`: contribution `-0.005473`
- `lag_04__T_A_site_active_infernos`: contribution `-0.002412`
- `lag_01__CT1__flash_duration`: contribution `-0.002322`
- `lag_11__T_A_site_active_infernos`: contribution `-0.001545`

### tick `131573`, seconds `48.00`, LSTM delta `+0.0543`

Top all feature movements:
- `lag_00__CT_flashed_players`: contribution `+0.006325`
- `lag_00__CT1__flash_duration`: contribution `+0.005788`
- `lag_08__CT_place_ELECTRICALBOX`: contribution `+0.004239`
- `lag_11__CT4__duck_amount`: contribution `+0.004025`
- `lag_12__T_place_LONGDOG`: contribution `+0.003635`

Top utility-only movements:
- `lag_00__CT1__flash_duration`: contribution `+0.005788`
- `lag_00__CT_flash_duration_sum`: contribution `+0.003244`
- `lag_00__CT3__flash_duration`: contribution `+0.001237`
