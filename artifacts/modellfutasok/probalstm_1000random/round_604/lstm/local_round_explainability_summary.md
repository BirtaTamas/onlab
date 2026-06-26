# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-gamerlegion-vs-pain-bo3-zcuZjSa9VUSMkJoK5k8I3c/gamerlegion-vs-pain-m3-mirage.csv`
- round_num: `3`

## Largest probability jumps

- tick `13868`, seconds `25.00`, LSTM `0.5324`, delta `+0.2487`
- tick `14252`, seconds `31.00`, LSTM `0.5446`, delta `-0.2383`
- tick `20780`, seconds `133.00`, LSTM `0.6208`, delta `+0.2189`
- tick `13996`, seconds `27.00`, LSTM `0.4812`, delta `-0.2063`
- tick `15820`, seconds `55.50`, LSTM `0.1314`, delta `-0.1852`
- tick `18540`, seconds `98.00`, LSTM `0.3546`, delta `+0.1823`
- tick `13932`, seconds `26.00`, LSTM `0.5655`, delta `+0.1782`
- tick `14156`, seconds `29.50`, LSTM `0.7369`, delta `+0.1709`
- tick `13900`, seconds `25.50`, LSTM `0.3873`, delta `-0.1451`
- tick `13964`, seconds `26.50`, LSTM `0.6875`, delta `+0.1220`

## Top 15 local ridge features

- `lag_00__CT_defusing_count`: coefficient `0.015182`, |coef| `0.015182`
- `lag_00__CT_velocity_mean`: coefficient `-0.005663`, |coef| `0.005663`
- `lag_00__T_shots_fired_sum`: coefficient `-0.005632`, |coef| `0.005632`
- `lag_00__kill_diff_last_3s`: coefficient `0.004475`, |coef| `0.004475`
- `lag_01__CT_defusing_count`: coefficient `0.004463`, |coef| `0.004463`
- `lag_00__T_place_HOUSE`: coefficient `-0.003800`, |coef| `0.003800`
- `lag_14__T_duck_amount_mean`: coefficient `0.003643`, |coef| `0.003643`
- `lag_00__T_place_CTSPAWN`: coefficient `-0.003222`, |coef| `0.003222`
- `lag_13__CT_velocity_mean`: coefficient `-0.003109`, |coef| `0.003109`
- `lag_00__T_place_SHOP`: coefficient `-0.003082`, |coef| `0.003082`
- `lag_11__T_duck_amount_mean`: coefficient `0.003012`, |coef| `0.003012`
- `lag_00__T_kills_last_3s`: coefficient `-0.002944`, |coef| `0.002944`
- `lag_12__T_duck_amount_mean`: coefficient `0.002938`, |coef| `0.002938`
- `lag_14__T5__duck_amount`: coefficient `0.002911`, |coef| `0.002911`
- `lag_07__CT_place_JUNGLE`: coefficient `0.002845`, |coef| `0.002845`

## Top 10 utility ridge features

- `lag_00__CT5__smoke`: coefficient `0.002040` (raises CT win probability)
- `lag_12__CT_A_site_active_smokes`: coefficient `0.001573` (raises CT win probability)
- `lag_05__CT1__smoke`: coefficient `-0.001534` (lowers CT win probability)
- `lag_11__CT_A_site_active_smokes`: coefficient `0.001420` (raises CT win probability)
- `lag_14__CT1__smoke`: coefficient `-0.001364` (lowers CT win probability)
- `lag_13__CT_A_site_active_smokes`: coefficient `0.001310` (raises CT win probability)
- `lag_08__CT1__smoke`: coefficient `-0.001267` (lowers CT win probability)
- `lag_07__CT1__flash`: coefficient `0.001152` (raises CT win probability)
- `lag_14__CT_A_site_active_smokes`: coefficient `0.001140` (raises CT win probability)
- `lag_12__CT_active_smokes`: coefficient `0.001137` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_defusing_count`: coefficient `0.015182` (raises CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.005663` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.005632` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.004475` (raises CT win probability)
- `lag_01__CT_defusing_count`: coefficient `0.004463` (raises CT win probability)
- `lag_00__T_place_HOUSE`: coefficient `-0.003800` (lowers CT win probability)
- `lag_14__T_duck_amount_mean`: coefficient `0.003643` (raises CT win probability)
- `lag_00__T_place_CTSPAWN`: coefficient `-0.003222` (lowers CT win probability)
- `lag_13__CT_velocity_mean`: coefficient `-0.003109` (lowers CT win probability)
- `lag_00__T_place_SHOP`: coefficient `-0.003082` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `13868`, seconds `25.00`, LSTM delta `+0.2487`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.135112`
- `lag_00__T3__shots_fired`: contribution `+0.044935`
- `lag_00__kill_diff_last_3s`: contribution `+0.010772`
- `lag_02__T_shots_fired_sum`: contribution `-0.008630`
- `lag_00__T_place_JUNGLE`: contribution `+0.007924`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `14252`, seconds `31.00`, LSTM delta `-0.2383`

Top all feature movements:
- `lag_12__T_shots_fired_sum`: contribution `-0.037939`
- `lag_12__T3__shots_fired`: contribution `-0.036597`
- `lag_00__T_shots_fired_sum`: contribution `-0.021111`
- `lag_07__T_shots_fired_sum`: contribution `-0.016683`
- `lag_00__kill_diff_last_3s`: contribution `-0.010772`

Top utility-only movements:
- `lag_12__T2__flash_duration`: contribution `-0.005327`
- `lag_11__CT3__flash_duration`: contribution `-0.004699`
- `lag_02__T_utility_damage_last_5s`: contribution `-0.004107`

### tick `20780`, seconds `133.00`, LSTM delta `+0.2189`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `+0.147176`
- `lag_14__T_duck_amount_mean`: contribution `+0.020439`
- `lag_00__CT_velocity_mean`: contribution `+0.010844`
- `lag_14__T5__duck_amount`: contribution `+0.010665`
- `lag_06__CT_duck_amount_mean`: contribution `+0.007169`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `13996`, seconds `27.00`, LSTM delta `-0.2063`

Top all feature movements:
- `lag_04__T3__shots_fired`: contribution `-0.030833`
- `lag_00__T_shots_fired_sum`: contribution `-0.029556`
- `lag_04__T_shots_fired_sum`: contribution `-0.027797`
- `lag_00__CT_shots_fired_sum`: contribution `-0.018033`
- `lag_04__T_place_JUNGLE`: contribution `-0.011365`

Top utility-only movements:
- `lag_11__CT_A_site_active_smokes`: contribution `+0.004571`

### tick `15820`, seconds `55.50`, LSTM delta `-0.1852`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.025333`
- `lag_08__CT_place_JUNGLE`: contribution `-0.017185`
- `lag_15__T_velocity_mean`: contribution `-0.014341`
- `lag_14__T_velocity_mean`: contribution `-0.013322`
- `lag_00__kill_diff_last_3s`: contribution `-0.010772`

Top utility-only movements:
- `lag_00__CT5__smoke`: contribution `-0.004474`
