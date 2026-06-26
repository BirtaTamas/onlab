# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m1-inferno.csv`
- round_num: `18`

## Largest probability jumps

- tick `185413`, seconds `87.50`, LSTM `0.0795`, delta `-0.1927`
- tick `181125`, seconds `20.50`, LSTM `0.3047`, delta `-0.1798`
- tick `185445`, seconds `88.00`, LSTM `0.2435`, delta `+0.1640`
- tick `183141`, seconds `52.00`, LSTM `0.0734`, delta `-0.1526`
- tick `185317`, seconds `86.00`, LSTM `0.2549`, delta `+0.1037`
- tick `185509`, seconds `89.00`, LSTM `0.0943`, delta `-0.1003`
- tick `182053`, seconds `35.00`, LSTM `0.2847`, delta `+0.0770`
- tick `181221`, seconds `22.00`, LSTM `0.2053`, delta `-0.0688`
- tick `185221`, seconds `84.50`, LSTM `0.0937`, delta `+0.0681`
- tick `185349`, seconds `86.50`, LSTM `0.3174`, delta `+0.0625`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002111`, |coef| `0.002111`
- `lag_00__T_kills_last_3s`: coefficient `-0.001891`, |coef| `0.001891`
- `lag_00__damage_diff_last_5s`: coefficient `0.001851`, |coef| `0.001851`
- `lag_02__CT_shots_fired_sum`: coefficient `-0.001723`, |coef| `0.001723`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001585`, |coef| `0.001585`
- `lag_00__T_damage_last_5s`: coefficient `-0.001578`, |coef| `0.001578`
- `lag_00__T_place_BALCONY`: coefficient `-0.001576`, |coef| `0.001576`
- `lag_14__T_shots_fired_sum`: coefficient `0.001543`, |coef| `0.001543`
- `lag_14__T_place_SECONDMID`: coefficient `0.001448`, |coef| `0.001448`
- `lag_01__T4__is_scoped`: coefficient `-0.001430`, |coef| `0.001430`
- `lag_00__CT5__alive`: coefficient `0.001368`, |coef| `0.001368`
- `lag_00__CT5__hp`: coefficient `0.001280`, |coef| `0.001280`
- `lag_00__CT5__armor`: coefficient `0.001274`, |coef| `0.001274`
- `lag_13__T_place_SECONDMID`: coefficient `0.001269`, |coef| `0.001269`
- `lag_08__CT1__flash_duration`: coefficient `0.001173`, |coef| `0.001173`

## Top 10 utility ridge features

- `lag_08__CT1__flash_duration`: coefficient `0.001173` (raises CT win probability)
- `lag_08__CT3__flash_duration`: coefficient `0.001038` (raises CT win probability)
- `lag_10__T4__molly`: coefficient `0.001033` (raises CT win probability)
- `lag_08__T2__flash_duration`: coefficient `-0.001011` (lowers CT win probability)
- `lag_00__CT5__flash`: coefficient `0.000999` (raises CT win probability)
- `lag_07__T2__flash_duration`: coefficient `-0.000995` (lowers CT win probability)
- `lag_07__CT1__flash_duration`: coefficient `0.000859` (raises CT win probability)
- `lag_01__CT_B_site_active_smokes`: coefficient `0.000827` (raises CT win probability)
- `lag_08__CT_flash_duration_sum`: coefficient `0.000820` (raises CT win probability)
- `lag_02__T_flash_duration_sum`: coefficient `0.000818` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002111` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001891` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001851` (raises CT win probability)
- `lag_02__CT_shots_fired_sum`: coefficient `-0.001723` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001585` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001578` (lowers CT win probability)
- `lag_00__T_place_BALCONY`: coefficient `-0.001576` (lowers CT win probability)
- `lag_14__T_shots_fired_sum`: coefficient `0.001543` (raises CT win probability)
- `lag_14__T_place_SECONDMID`: coefficient `0.001448` (raises CT win probability)
- `lag_01__T4__is_scoped`: coefficient `-0.001430` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `185413`, seconds `87.50`, LSTM delta `-0.1927`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.010164`
- `lag_01__CT_shots_fired_sum`: contribution `-0.009602`
- `lag_02__CT_shots_fired_sum`: contribution `-0.008377`
- `lag_01__CT1__shots_fired`: contribution `-0.006869`
- `lag_01__T4__is_scoped`: contribution `-0.006640`

Top utility-only movements:
- `lag_08__T5__flash_duration`: contribution `-0.005892`
- `lag_05__T2__flash_duration`: contribution `-0.005669`
- `lag_08__CT1__flash_duration`: contribution `-0.004135`
- `lag_12__CT1__flash_duration`: contribution `-0.003792`
- `lag_08__CT3__flash_duration`: contribution `-0.003676`

### tick `181125`, seconds `20.50`, LSTM delta `-0.1798`

Top all feature movements:
- `lag_14__T_shots_fired_sum`: contribution `-0.055541`
- `lag_14__T5__shots_fired`: contribution `-0.011787`
- `lag_14__T2__shots_fired`: contribution `-0.007764`
- `lag_00__T_kills_last_3s`: contribution `-0.005990`
- `lag_00__kill_diff_last_3s`: contribution `-0.005082`

Top utility-only movements:
- `lag_02__T5__flash_duration`: contribution `+0.002288`
- `lag_07__CT_A_site_active_infernos`: contribution `-0.001886`
- `lag_00__CT2__molly`: contribution `-0.001697`

### tick `185445`, seconds `88.00`, LSTM delta `+0.1640`

Top all feature movements:
- `lag_02__CT_shots_fired_sum`: contribution `+0.016754`
- `lag_02__CT1__shots_fired`: contribution `+0.006926`
- `lag_01__T4__is_scoped`: contribution `+0.006640`
- `lag_00__kill_diff_last_3s`: contribution `+0.005082`
- `lag_04__CT_shots_fired_sum`: contribution `+0.004913`

Top utility-only movements:
- `lag_13__CT1__flash_duration`: contribution `+0.004499`
- `lag_06__T2__flash_duration`: contribution `+0.004301`
- `lag_06__T1__flash_duration`: contribution `+0.004214`
- `lag_00__T5__flash_duration`: contribution `+0.004143`
- `lag_13__CT3__flash_duration`: contribution `+0.003587`

### tick `183141`, seconds `52.00`, LSTM delta `-0.1526`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.005990`
- `lag_00__kill_diff_last_3s`: contribution `-0.005082`
- `lag_14__T_place_SECONDMID`: contribution `-0.004740`
- `lag_07__T4__is_scoped`: contribution `-0.004387`
- `lag_11__T3__duck_amount`: contribution `-0.003975`

Top utility-only movements:
- `lag_10__T4__molly`: contribution `-0.002251`
- `lag_00__CT5__flash`: contribution `-0.001773`

### tick `185317`, seconds `86.00`, LSTM delta `+0.1037`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.007710`
- `lag_02__CT_shots_fired_sum`: contribution `+0.007180`
- `lag_01__T4__is_scoped`: contribution `+0.006640`
- `lag_09__CT1__flash_duration`: contribution `+0.005101`
- `lag_02__T2__flash_duration`: contribution `+0.004616`

Top utility-only movements:
- `lag_09__CT1__flash_duration`: contribution `+0.005101`
- `lag_02__T2__flash_duration`: contribution `+0.004616`
- `lag_02__T1__flash_duration`: contribution `+0.004535`
- `lag_02__T4__flash_duration`: contribution `+0.004484`
- `lag_09__CT3__flash_duration`: contribution `+0.004130`
