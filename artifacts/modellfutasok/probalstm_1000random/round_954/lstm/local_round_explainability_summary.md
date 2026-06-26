# Local Round Explainability

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-ninjas-in-pyjamas-vs-natus-vincere-bo3-NyGFAYn592Hu0YfljTVG0N/ninjas-in-pyjamas-vs-natus-vincere-m3-train.csv`
- round_num: `4`

## Largest probability jumps

- tick `24161`, seconds `90.50`, LSTM `0.1548`, delta `-0.3891`
- tick `24097`, seconds `89.50`, LSTM `0.5416`, delta `-0.2707`
- tick `23201`, seconds `75.50`, LSTM `0.4100`, delta `-0.1654`
- tick `23969`, seconds `87.50`, LSTM `0.6779`, delta `+0.1359`
- tick `23873`, seconds `86.00`, LSTM `0.4694`, delta `+0.0816`
- tick `23681`, seconds `83.00`, LSTM `0.3851`, delta `-0.0699`
- tick `24065`, seconds `89.00`, LSTM `0.8123`, delta `+0.0609`
- tick `24001`, seconds `88.00`, LSTM `0.7360`, delta `+0.0581`
- tick `23233`, seconds `76.00`, LSTM `0.3521`, delta `-0.0579`
- tick `24193`, seconds `91.00`, LSTM `0.0977`, delta `-0.0571`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003562`, |coef| `0.003562`
- `lag_01__kill_diff_last_3s`: coefficient `0.002723`, |coef| `0.002723`
- `lag_00__CT_kills_last_3s`: coefficient `0.002292`, |coef| `0.002292`
- `lag_00__T_kills_last_3s`: coefficient `-0.002173`, |coef| `0.002173`
- `lag_05__T_place_ELECTRICALBOX`: coefficient `-0.002108`, |coef| `0.002108`
- `lag_01__CT_kills_last_3s`: coefficient `0.001983`, |coef| `0.001983`
- `lag_06__T_shots_fired_sum`: coefficient `0.001753`, |coef| `0.001753`
- `lag_02__kill_diff_last_3s`: coefficient `0.001700`, |coef| `0.001700`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001611`, |coef| `0.001611`
- `lag_02__T_utility_damage_last_5s`: coefficient `-0.001525`, |coef| `0.001525`
- `lag_03__CT_kills_last_3s`: coefficient `0.001470`, |coef| `0.001470`
- `lag_08__CT_place_ENTRANCE`: coefficient `-0.001463`, |coef| `0.001463`
- `lag_02__CT_place_ENTRANCE`: coefficient `0.001439`, |coef| `0.001439`
- `lag_10__T_place_LONGDOG`: coefficient `0.001432`, |coef| `0.001432`
- `lag_01__T_kills_last_3s`: coefficient `-0.001408`, |coef| `0.001408`

## Top 10 utility ridge features

- `lag_02__T_utility_damage_last_5s`: coefficient `-0.001525` (lowers CT win probability)
- `lag_02__utility_damage_diff_last_5s`: coefficient `0.001334` (raises CT win probability)
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.001106` (lowers CT win probability)
- `lag_00__CT5__molly`: coefficient `0.001090` (raises CT win probability)
- `lag_10__CT_B_site_active_infernos`: coefficient `-0.000953` (lowers CT win probability)
- `lag_07__T_utility_damage_last_5s`: coefficient `-0.000940` (lowers CT win probability)
- `lag_09__utility_damage_diff_last_5s`: coefficient `0.000910` (raises CT win probability)
- `lag_09__T_A_site_active_infernos`: coefficient `0.000846` (raises CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000846` (raises CT win probability)
- `lag_03__CT_B_site_active_infernos`: coefficient `0.000828` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003562` (raises CT win probability)
- `lag_01__kill_diff_last_3s`: coefficient `0.002723` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002292` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002173` (lowers CT win probability)
- `lag_05__T_place_ELECTRICALBOX`: coefficient `-0.002108` (lowers CT win probability)
- `lag_01__CT_kills_last_3s`: coefficient `0.001983` (raises CT win probability)
- `lag_06__T_shots_fired_sum`: coefficient `0.001753` (raises CT win probability)
- `lag_02__kill_diff_last_3s`: coefficient `0.001700` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001611` (raises CT win probability)
- `lag_03__CT_kills_last_3s`: coefficient `0.001470` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `24161`, seconds `90.50`, LSTM delta `-0.3891`

Top all feature movements:
- `lag_05__T_place_ELECTRICALBOX`: contribution `-0.055327`
- `lag_07__T_place_ELECTRICALBOX`: contribution `-0.026134`
- `lag_00__kill_diff_last_3s`: contribution `-0.017146`
- `lag_08__CT_place_ENTRANCE`: contribution `-0.012983`
- `lag_02__T_utility_damage_last_5s`: contribution `-0.010887`

Top utility-only movements:
- `lag_02__T_utility_damage_last_5s`: contribution `-0.010887`
- `lag_02__utility_damage_diff_last_5s`: contribution `-0.006022`

### tick `24097`, seconds `89.50`, LSTM delta `-0.2707`

Top all feature movements:
- `lag_05__T_place_ELECTRICALBOX`: contribution `-0.055327`
- `lag_03__T_place_ELECTRICALBOX`: contribution `-0.028854`
- `lag_06__T_shots_fired_sum`: contribution `-0.018403`
- `lag_04__CT_place_ENTRANCE`: contribution `-0.010140`
- `lag_00__kill_diff_last_3s`: contribution `-0.008573`

Top utility-only movements:
- `lag_00__T_utility_damage_last_5s`: contribution `-0.007894`

### tick `23201`, seconds `75.50`, LSTM delta `-0.1654`

Top all feature movements:
- `lag_10__T_place_ELECTRICALBOX`: contribution `-0.022912`
- `lag_13__T_place_DUMPSTER`: contribution `-0.009390`
- `lag_00__kill_diff_last_3s`: contribution `-0.008573`
- `lag_00__CT_shots_fired_sum`: contribution `-0.007832`
- `lag_00__T_kills_last_3s`: contribution `-0.006885`

Top utility-only movements:
- `lag_09__T_A_site_active_infernos`: contribution `-0.002517`
- `lag_00__CT4__flash_duration`: contribution `-0.002279`
- `lag_04__CT_utility_damage_last_5s`: contribution `-0.002256`

### tick `23969`, seconds `87.50`, LSTM delta `+0.1359`

Top all feature movements:
- `lag_02__CT_place_ENTRANCE`: contribution `+0.012765`
- `lag_01__T_place_ELECTRICALBOX`: contribution `+0.011816`
- `lag_00__kill_diff_last_3s`: contribution `+0.008573`
- `lag_00__T_place_ELECTRICALBOX`: contribution `-0.008137`
- `lag_00__CT_kills_last_3s`: contribution `+0.006617`

Top utility-only movements:
- `lag_10__CT_B_site_active_infernos`: contribution `+0.003274`
- `lag_09__T_A_site_active_infernos`: contribution `+0.002517`

### tick `23873`, seconds `86.00`, LSTM delta `+0.0816`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.008573`
- `lag_00__T_shots_fired_sum`: contribution `-0.006752`
- `lag_10__T_place_LONGDOG`: contribution `+0.006664`
- `lag_00__CT_kills_last_3s`: contribution `+0.006617`
- `lag_06__T_shots_fired_sum`: contribution `+0.006572`

Top utility-only movements:
- `lag_15__CT_utility_damage_last_5s`: contribution `+0.003294`
- `lag_15__utility_damage_diff_last_5s`: contribution `+0.002611`
