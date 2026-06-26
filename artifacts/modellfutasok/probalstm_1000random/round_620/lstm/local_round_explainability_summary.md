# Local Round Explainability

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-spirit-vs-g2-bo3-3aFk7fRwd7iUE0VJycUPHK/spirit-vs-g2-m3-ancient.csv`
- round_num: `2`

## Largest probability jumps

- tick `21771`, seconds `18.00`, LSTM `0.0710`, delta `-0.3818`
- tick `21707`, seconds `17.00`, LSTM `0.4455`, delta `-0.2478`
- tick `21515`, seconds `14.00`, LSTM `0.7409`, delta `+0.1922`
- tick `21355`, seconds `11.50`, LSTM `0.4728`, delta `+0.1009`
- tick `21547`, seconds `14.50`, LSTM `0.6547`, delta `-0.0862`
- tick `21451`, seconds `13.00`, LSTM `0.5805`, delta `+0.0791`
- tick `21067`, seconds `7.00`, LSTM `0.2543`, delta `+0.0734`
- tick `21099`, seconds `7.50`, LSTM `0.3031`, delta `+0.0487`
- tick `21675`, seconds `16.50`, LSTM `0.6933`, delta `+0.0452`
- tick `22667`, seconds `32.00`, LSTM `0.0099`, delta `-0.0450`

## Top 15 local ridge features

- `lag_08__T1__flash_duration`: coefficient `0.002257`, |coef| `0.002257`
- `lag_00__kill_diff_last_3s`: coefficient `0.002071`, |coef| `0.002071`
- `lag_00__CT1__flash_duration`: coefficient `0.001979`, |coef| `0.001979`
- `lag_10__T_place_TSIDELOWER`: coefficient `0.001602`, |coef| `0.001602`
- `lag_01__T_shots_fired_sum`: coefficient `0.001582`, |coef| `0.001582`
- `lag_00__T_kills_last_3s`: coefficient `-0.001576`, |coef| `0.001576`
- `lag_06__CT3__flash_duration`: coefficient `0.001542`, |coef| `0.001542`
- `lag_08__T1__shots_fired`: coefficient `-0.001432`, |coef| `0.001432`
- `lag_06__T1__flash_duration`: coefficient `0.001423`, |coef| `0.001423`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001413`, |coef| `0.001413`
- `lag_06__T_shots_fired_sum`: coefficient `0.001412`, |coef| `0.001412`
- `lag_14__T5__shots_fired`: coefficient `-0.001402`, |coef| `0.001402`
- `lag_13__T5__shots_fired`: coefficient `-0.001392`, |coef| `0.001392`
- `lag_09__T1__shots_fired`: coefficient `-0.001372`, |coef| `0.001372`
- `lag_03__T_shots_fired_sum`: coefficient `-0.001367`, |coef| `0.001367`

## Top 10 utility ridge features

- `lag_08__T1__flash_duration`: coefficient `0.002257` (raises CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `0.001979` (raises CT win probability)
- `lag_06__CT3__flash_duration`: coefficient `0.001542` (raises CT win probability)
- `lag_06__T1__flash_duration`: coefficient `0.001423` (raises CT win probability)
- `lag_13__T5__flash_duration`: coefficient `0.001330` (raises CT win probability)
- `lag_13__CT1__flash_duration`: coefficient `-0.001228` (lowers CT win probability)
- `lag_04__CT3__flash_duration`: coefficient `0.001125` (raises CT win probability)
- `lag_10__CT_utility_damage_last_5s`: coefficient `0.001079` (raises CT win probability)
- `lag_14__T1__flash_duration`: coefficient `-0.001072` (lowers CT win probability)
- `lag_08__T_flash_duration_sum`: coefficient `0.001003` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002071` (raises CT win probability)
- `lag_10__T_place_TSIDELOWER`: coefficient `0.001602` (raises CT win probability)
- `lag_01__T_shots_fired_sum`: coefficient `0.001582` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001576` (lowers CT win probability)
- `lag_08__T1__shots_fired`: coefficient `-0.001432` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001413` (lowers CT win probability)
- `lag_06__T_shots_fired_sum`: coefficient `0.001412` (raises CT win probability)
- `lag_14__T5__shots_fired`: coefficient `-0.001402` (lowers CT win probability)
- `lag_13__T5__shots_fired`: coefficient `-0.001392` (lowers CT win probability)
- `lag_09__T1__shots_fired`: coefficient `-0.001372` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `21771`, seconds `18.00`, LSTM delta `-0.3818`

Top all feature movements:
- `lag_01__T_shots_fired_sum`: contribution `-0.015414`
- `lag_08__T1__flash_duration`: contribution `-0.014330`
- `lag_00__CT1__flash_duration`: contribution `-0.012949`
- `lag_01__T4__shots_fired`: contribution `-0.009126`
- `lag_06__CT3__flash_duration`: contribution `-0.008006`

Top utility-only movements:
- `lag_08__T1__flash_duration`: contribution `-0.014330`
- `lag_00__CT1__flash_duration`: contribution `-0.012949`
- `lag_06__CT3__flash_duration`: contribution `-0.008006`
- `lag_13__CT1__flash_duration`: contribution `-0.007494`
- `lag_13__T5__flash_duration`: contribution `-0.006867`

### tick `21707`, seconds `17.00`, LSTM delta `-0.2478`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.009970`
- `lag_06__T_shots_fired_sum`: contribution `-0.009529`
- `lag_06__T1__flash_duration`: contribution `-0.009037`
- `lag_11__T_shots_fired_sum`: contribution `-0.007650`
- `lag_00__T_shots_fired_sum`: contribution `-0.007416`

Top utility-only movements:
- `lag_06__T1__flash_duration`: contribution `-0.009037`
- `lag_14__T1__flash_duration`: contribution `-0.006803`
- `lag_11__CT1__flash_duration`: contribution `-0.006097`
- `lag_04__CT3__flash_duration`: contribution `-0.005841`
- `lag_11__T5__flash_duration`: contribution `-0.004701`

### tick `21515`, seconds `14.00`, LSTM delta `+0.1922`

Top all feature movements:
- `lag_08__T1__flash_duration`: contribution `+0.014330`
- `lag_00__T_shots_fired_sum`: contribution `+0.009535`
- `lag_15__T_place_WATER`: contribution `+0.007655`
- `lag_15__T_place_RUINS`: contribution `+0.007214`
- `lag_01__T_shots_fired_sum`: contribution `+0.007114`

Top utility-only movements:
- `lag_08__T1__flash_duration`: contribution `+0.014330`
- `lag_12__CT2__flash_duration`: contribution `+0.003711`
- `lag_08__CT3__flash_duration`: contribution `+0.003605`
- `lag_08__T_flash_duration_sum`: contribution `+0.003585`
- `lag_00__T1__flash_duration`: contribution `+0.002849`

### tick `21355`, seconds `11.50`, LSTM delta `+0.1009`

Top all feature movements:
- `lag_00__CT1__flash_duration`: contribution `+0.012076`
- `lag_00__T_shots_fired_sum`: contribution `+0.011654`
- `lag_01__T_shots_fired_sum`: contribution `+0.007114`
- `lag_12__T_place_WATER`: contribution `+0.006335`
- `lag_00__kill_diff_last_3s`: contribution `+0.004985`

Top utility-only movements:
- `lag_00__CT1__flash_duration`: contribution `+0.012076`
- `lag_00__CT_flash_duration_sum`: contribution `+0.003336`
- `lag_07__CT2__flash_duration`: contribution `+0.002048`
- `lag_03__T1__flash_duration`: contribution `+0.001865`
- `lag_07__CT_flash_duration_sum`: contribution `+0.001816`

### tick `21547`, seconds `14.50`, LSTM delta `-0.0862`

Top all feature movements:
- `lag_06__T_shots_fired_sum`: contribution `-0.011647`
- `lag_01__T_shots_fired_sum`: contribution `-0.010672`
- `lag_00__kill_diff_last_3s`: contribution `-0.004985`
- `lag_15__CT_place_HOUSE`: contribution `-0.004440`
- `lag_13__CT_flash_duration_sum`: contribution `-0.003835`

Top utility-only movements:
- `lag_13__CT_flash_duration_sum`: contribution `-0.003835`
- `lag_06__CT_flash_duration_sum`: contribution `+0.003395`
- `lag_01__T1__flash_duration`: contribution `-0.002639`
- `lag_06__CT5__flash_duration`: contribution `+0.002202`
