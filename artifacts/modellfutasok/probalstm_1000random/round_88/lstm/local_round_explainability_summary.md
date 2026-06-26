# Local Round Explainability

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-falcons-vs-nrg-bo3-WMQcRUwgyUmu57EEkX9f3P/falcons-vs-nrg-m1-train.csv`
- round_num: `18`

## Largest probability jumps

- tick `151443`, seconds `90.50`, LSTM `0.4646`, delta `+0.4458`
- tick `151507`, seconds `91.50`, LSTM `0.1750`, delta `-0.2485`
- tick `150515`, seconds `76.00`, LSTM `0.4170`, delta `+0.2061`
- tick `151219`, seconds `87.00`, LSTM `0.1244`, delta `-0.1599`
- tick `151059`, seconds `84.50`, LSTM `0.3189`, delta `-0.1523`
- tick `150547`, seconds `76.50`, LSTM `0.3270`, delta `-0.0899`
- tick `150579`, seconds `77.00`, LSTM `0.4043`, delta `+0.0772`
- tick `151091`, seconds `85.00`, LSTM `0.2453`, delta `-0.0735`
- tick `146547`, seconds `14.00`, LSTM `0.2359`, delta `+0.0643`
- tick `146611`, seconds `15.00`, LSTM `0.2825`, delta `+0.0604`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003476`, |coef| `0.003476`
- `lag_07__CT4__flash_duration`: coefficient `-0.003141`, |coef| `0.003141`
- `lag_00__CT_kills_last_3s`: coefficient `0.003044`, |coef| `0.003044`
- `lag_00__T_place_TMAIN`: coefficient `-0.002969`, |coef| `0.002969`
- `lag_07__T_shots_fired_sum`: coefficient `0.002523`, |coef| `0.002523`
- `lag_07__CT5__flash_duration`: coefficient `0.002077`, |coef| `0.002077`
- `lag_09__CT_place_CONNECTOR`: coefficient `-0.002037`, |coef| `0.002037`
- `lag_00__damage_diff_last_5s`: coefficient `0.001999`, |coef| `0.001999`
- `lag_09__T_shots_fired_sum`: coefficient `-0.001734`, |coef| `0.001734`
- `lag_11__T3__duck_amount`: coefficient `0.001734`, |coef| `0.001734`
- `lag_01__CT4__duck_amount`: coefficient `0.001734`, |coef| `0.001734`
- `lag_08__T_shots_fired_sum`: coefficient `0.001688`, |coef| `0.001688`
- `lag_04__CT_A_site_active_infernos`: coefficient `-0.001642`, |coef| `0.001642`
- `lag_06__T_place_LONGDOG`: coefficient `0.001560`, |coef| `0.001560`
- `lag_06__T_shots_fired_sum`: coefficient `-0.001560`, |coef| `0.001560`

## Top 10 utility ridge features

- `lag_07__CT4__flash_duration`: coefficient `-0.003141` (lowers CT win probability)
- `lag_07__CT5__flash_duration`: coefficient `0.002077` (raises CT win probability)
- `lag_04__CT_A_site_active_infernos`: coefficient `-0.001642` (lowers CT win probability)
- `lag_08__CT4__flash_duration`: coefficient `-0.001544` (lowers CT win probability)
- `lag_08__CT_A_site_active_infernos`: coefficient `0.001395` (raises CT win probability)
- `lag_07__CT_flash_duration_sum`: coefficient `-0.001203` (lowers CT win probability)
- `lag_13__CT4__flash_duration`: coefficient `-0.001153` (lowers CT win probability)
- `lag_04__CT_active_infernos`: coefficient `-0.001083` (lowers CT win probability)
- `lag_09__CT5__flash_duration`: coefficient `-0.001038` (lowers CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `0.000990` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003476` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003044` (raises CT win probability)
- `lag_00__T_place_TMAIN`: coefficient `-0.002969` (lowers CT win probability)
- `lag_07__T_shots_fired_sum`: coefficient `0.002523` (raises CT win probability)
- `lag_09__CT_place_CONNECTOR`: coefficient `-0.002037` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001999` (raises CT win probability)
- `lag_09__T_shots_fired_sum`: coefficient `-0.001734` (lowers CT win probability)
- `lag_11__T3__duck_amount`: coefficient `0.001734` (raises CT win probability)
- `lag_01__CT4__duck_amount`: coefficient `0.001734` (raises CT win probability)
- `lag_08__T_shots_fired_sum`: coefficient `0.001688` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `151443`, seconds `90.50`, LSTM delta `+0.4458`

Top all feature movements:
- `lag_07__CT4__flash_duration`: contribution `+0.022840`
- `lag_07__T_shots_fired_sum`: contribution `+0.018914`
- `lag_00__CT_kills_last_3s`: contribution `+0.017576`
- `lag_00__kill_diff_last_3s`: contribution `+0.016732`
- `lag_06__T_shots_fired_sum`: contribution `+0.012864`

Top utility-only movements:
- `lag_07__CT4__flash_duration`: contribution `+0.022840`
- `lag_07__CT5__flash_duration`: contribution `+0.010885`
- `lag_04__CT_A_site_active_infernos`: contribution `+0.005796`
- `lag_08__CT_A_site_active_infernos`: contribution `+0.004924`

### tick `151507`, seconds `91.50`, LSTM delta `-0.2485`

Top all feature movements:
- `lag_08__T_shots_fired_sum`: contribution `-0.013918`
- `lag_09__T_shots_fired_sum`: contribution `-0.013001`
- `lag_09__CT_place_ELECTRICALBOX`: contribution `-0.012207`
- `lag_00__kill_diff_last_3s`: contribution `-0.008366`
- `lag_02__kill_diff_last_3s`: contribution `-0.007506`

Top utility-only movements:
- `lag_09__CT5__flash_duration`: contribution `-0.005439`

### tick `150515`, seconds `76.00`, LSTM delta `+0.2061`

Top all feature movements:
- `lag_05__CT_place_TMAIN`: contribution `+0.015208`
- `lag_09__CT_place_ELECTRICALBOX`: contribution `-0.012207`
- `lag_12__CT_place_ELECTRICALBOX`: contribution `+0.011751`
- `lag_00__T_place_TMAIN`: contribution `+0.011512`
- `lag_00__CT_kills_last_3s`: contribution `+0.008788`

Top utility-only movements:
- `lag_03__T_A_site_active_infernos`: contribution `+0.002551`

### tick `151219`, seconds `87.00`, LSTM delta `-0.1599`

Top all feature movements:
- `lag_00__CT_place_ELECTRICALBOX`: contribution `-0.015761`
- `lag_00__T_shots_fired_sum`: contribution `-0.011066`
- `lag_13__CT4__flash_duration`: contribution `-0.008383`
- `lag_00__kill_diff_last_3s`: contribution `-0.008366`
- `lag_09__CT_place_CONNECTOR`: contribution `-0.007284`

Top utility-only movements:
- `lag_13__CT4__flash_duration`: contribution `-0.008383`
- `lag_00__CT4__flash_duration`: contribution `-0.007201`
- `lag_00__CT5__flash_duration`: contribution `-0.004327`
- `lag_11__T_A_site_active_infernos`: contribution `-0.001943`

### tick `151059`, seconds `84.50`, LSTM delta `-0.1523`

Top all feature movements:
- `lag_08__CT4__flash_duration`: contribution `-0.011223`
- `lag_15__T_shots_fired_sum`: contribution `-0.008685`
- `lag_09__T_place_DUMPSTER`: contribution `-0.006495`
- `lag_15__T5__shots_fired`: contribution `-0.005648`
- `lag_00__damage_diff_last_5s`: contribution `-0.004104`

Top utility-only movements:
- `lag_08__CT4__flash_duration`: contribution `-0.011223`
- `lag_06__T_A_site_active_infernos`: contribution `-0.002920`
- `lag_08__CT_flash_duration_sum`: contribution `-0.002275`
