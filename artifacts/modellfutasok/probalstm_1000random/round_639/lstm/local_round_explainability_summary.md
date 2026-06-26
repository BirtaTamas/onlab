# Local Round Explainability

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-falcons-vs-g2-bo3-Xf_UKx9fB2btv0Vy_VBVUC/falcons-vs-g2-m3-mirage.csv`
- round_num: `3`

## Largest probability jumps

- tick `14132`, seconds `31.50`, LSTM `0.7222`, delta `+0.3760`
- tick `14228`, seconds `33.00`, LSTM `0.7673`, delta `+0.3506`
- tick `14260`, seconds `33.50`, LSTM `0.4810`, delta `-0.2863`
- tick `15124`, seconds `47.00`, LSTM `0.0720`, delta `-0.2694`
- tick `15156`, seconds `47.50`, LSTM `0.2828`, delta `+0.2108`
- tick `14100`, seconds `31.00`, LSTM `0.3463`, delta `-0.2040`
- tick `14196`, seconds `32.50`, LSTM `0.4167`, delta `-0.1836`
- tick `14164`, seconds `32.00`, LSTM `0.6003`, delta `-0.1219`
- tick `14292`, seconds `34.00`, LSTM `0.6012`, delta `+0.1202`
- tick `13524`, seconds `22.00`, LSTM `0.5686`, delta `+0.0922`

## Top 15 local ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.004630`, |coef| `0.004630`
- `lag_00__kill_diff_last_3s`: coefficient `0.003701`, |coef| `0.003701`
- `lag_04__T5__shots_fired`: coefficient `-0.003075`, |coef| `0.003075`
- `lag_00__T_kills_last_3s`: coefficient `-0.003066`, |coef| `0.003066`
- `lag_05__CT5__duck_amount`: coefficient `-0.002665`, |coef| `0.002665`
- `lag_01__T_shots_fired_sum`: coefficient `0.002656`, |coef| `0.002656`
- `lag_00__T3__duck_amount`: coefficient `-0.002495`, |coef| `0.002495`
- `lag_08__T5__duck_amount`: coefficient `0.002431`, |coef| `0.002431`
- `lag_01__CT5__duck_amount`: coefficient `0.002333`, |coef| `0.002333`
- `lag_05__CT5__is_walking`: coefficient `0.002267`, |coef| `0.002267`
- `lag_07__T5__duck_amount`: coefficient `0.002057`, |coef| `0.002057`
- `lag_00__CT5__alive`: coefficient `0.001960`, |coef| `0.001960`
- `lag_14__CT1__is_walking`: coefficient `-0.001904`, |coef| `0.001904`
- `lag_00__T3__shots_fired`: coefficient `-0.001890`, |coef| `0.001890`
- `lag_05__T_place_TRUCK`: coefficient `0.001860`, |coef| `0.001860`

## Top 10 utility ridge features

- `lag_10__T3__smoke`: coefficient `0.001670` (raises CT win probability)
- `lag_00__T_B_site_active_smokes`: coefficient `0.001569` (raises CT win probability)
- `lag_03__T_B_site_active_smokes`: coefficient `0.001499` (raises CT win probability)
- `lag_09__T3__smoke`: coefficient `0.001374` (raises CT win probability)
- `lag_05__CT_B_site_active_smokes`: coefficient `0.001284` (raises CT win probability)
- `lag_04__CT_B_site_active_smokes`: coefficient `0.001079` (raises CT win probability)
- `lag_02__T_B_site_active_smokes`: coefficient `0.001050` (raises CT win probability)
- `lag_00__T_active_smokes`: coefficient `0.001027` (raises CT win probability)
- `lag_05__CT_active_smokes`: coefficient `0.000963` (raises CT win probability)
- `lag_05__T_B_site_active_smokes`: coefficient `0.000956` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.004630` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003701` (raises CT win probability)
- `lag_04__T5__shots_fired`: coefficient `-0.003075` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003066` (lowers CT win probability)
- `lag_05__CT5__duck_amount`: coefficient `-0.002665` (lowers CT win probability)
- `lag_01__T_shots_fired_sum`: coefficient `0.002656` (raises CT win probability)
- `lag_00__T3__duck_amount`: coefficient `-0.002495` (lowers CT win probability)
- `lag_08__T5__duck_amount`: coefficient `0.002431` (raises CT win probability)
- `lag_01__CT5__duck_amount`: coefficient `0.002333` (raises CT win probability)
- `lag_05__CT5__is_walking`: coefficient `0.002267` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `14132`, seconds `31.50`, LSTM delta `+0.3760`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.059008`
- `lag_02__T_place_TRUCK`: contribution `+0.029281`
- `lag_01__T_shots_fired_sum`: contribution `+0.019913`
- `lag_04__T5__shots_fired`: contribution `+0.018902`
- `lag_08__CT_place_TRUCK`: contribution `+0.010679`

Top utility-only movements:
- `lag_12__CT2__flash_duration`: contribution `+0.004835`

### tick `14228`, seconds `33.00`, LSTM delta `+0.3506`

Top all feature movements:
- `lag_05__T_place_TRUCK`: contribution `+0.032309`
- `lag_00__T_shots_fired_sum`: contribution `+0.020826`
- `lag_00__T_place_TRUCK`: contribution `+0.018043`
- `lag_05__CT5__duck_amount`: contribution `+0.010059`
- `lag_01__T_shots_fired_sum`: contribution `+0.009956`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `14260`, seconds `33.50`, LSTM delta `-0.2863`

Top all feature movements:
- `lag_04__T_shots_fired_sum`: contribution `+0.014129`
- `lag_00__T_shots_fired_sum`: contribution `-0.013884`
- `lag_06__T_place_TRUCK`: contribution `-0.013118`
- `lag_01__T_shots_fired_sum`: contribution `-0.011948`
- `lag_00__T_kills_last_3s`: contribution `-0.009713`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `15124`, seconds `47.00`, LSTM delta `-0.2694`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.010413`
- `lag_05__CT5__duck_amount`: contribution `-0.009746`
- `lag_00__T_kills_last_3s`: contribution `-0.009713`
- `lag_00__T3__duck_amount`: contribution `-0.009407`
- `lag_08__T5__duck_amount`: contribution `-0.009231`

Top utility-only movements:
- `lag_10__T3__smoke`: contribution `-0.003630`

### tick `15156`, seconds `47.50`, LSTM delta `+0.2108`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.010413`
- `lag_05__CT5__duck_amount`: contribution `+0.009746`
- `lag_00__T3__duck_amount`: contribution `+0.009407`
- `lag_00__kill_diff_last_3s`: contribution `+0.008907`
- `lag_04__T5__shots_fired`: contribution `+0.007561`

Top utility-only movements:
- No utility movement among the top local contributors.
