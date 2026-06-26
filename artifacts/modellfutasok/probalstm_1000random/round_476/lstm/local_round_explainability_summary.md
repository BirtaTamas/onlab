# Local Round Explainability

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-the-mongolz-vs-natus-vincere-bo3-C0GZxMhpGHBr28LeyjgICZ/the-mongolz-vs-natus-vincere-m1-mirage.csv`
- round_num: `9`

## Largest probability jumps

- tick `63702`, seconds `55.00`, LSTM `0.1218`, delta `-0.4199`
- tick `63414`, seconds `50.50`, LSTM `0.5119`, delta `+0.2617`
- tick `60726`, seconds `8.50`, LSTM `0.1204`, delta `-0.1938`
- tick `60694`, seconds `8.00`, LSTM `0.3141`, delta `+0.1797`
- tick `60566`, seconds `6.00`, LSTM `0.1610`, delta `-0.0725`
- tick `63734`, seconds `55.50`, LSTM `0.0503`, delta `-0.0715`
- tick `60246`, seconds `1.00`, LSTM `0.1575`, delta `+0.0682`
- tick `60982`, seconds `12.50`, LSTM `0.2398`, delta `+0.0611`
- tick `63030`, seconds `44.50`, LSTM `0.3127`, delta `+0.0551`
- tick `63670`, seconds `54.50`, LSTM `0.5418`, delta `+0.0454`

## Top 15 local ridge features

- `lag_00__CT_place_TRUCK`: coefficient `0.004976`, |coef| `0.004976`
- `lag_00__kill_diff_last_3s`: coefficient `0.003587`, |coef| `0.003587`
- `lag_00__CT_shots_fired_sum`: coefficient `0.003366`, |coef| `0.003366`
- `lag_12__T_place_SIDEALLEY`: coefficient `0.003312`, |coef| `0.003312`
- `lag_02__CT_place_TRUCK`: coefficient `0.002873`, |coef| `0.002873`
- `lag_00__damage_diff_last_5s`: coefficient `0.002872`, |coef| `0.002872`
- `lag_08__CT_shots_fired_sum`: coefficient `0.002731`, |coef| `0.002731`
- `lag_00__CT4__duck_amount`: coefficient `0.002555`, |coef| `0.002555`
- `lag_07__CT_place_CONNECTOR`: coefficient `-0.002530`, |coef| `0.002530`
- `lag_04__CT_place_TRUCK`: coefficient `-0.002500`, |coef| `0.002500`
- `lag_06__T_place_HOUSE`: coefficient `0.002351`, |coef| `0.002351`
- `lag_12__CT_place_CONNECTOR`: coefficient `0.002341`, |coef| `0.002341`
- `lag_00__CT_kills_last_3s`: coefficient `0.002324`, |coef| `0.002324`
- `lag_08__CT1__shots_fired`: coefficient `0.002318`, |coef| `0.002318`
- `lag_13__T4__is_walking`: coefficient `0.002278`, |coef| `0.002278`

## Top 10 utility ridge features

- `lag_13__T_he_last_5s`: coefficient `-0.001837` (lowers CT win probability)
- `lag_00__CT4__smoke`: coefficient `0.001763` (raises CT win probability)
- `lag_09__T3__molly`: coefficient `0.001735` (raises CT win probability)
- `lag_08__T_B_site_active_smokes`: coefficient `0.001719` (raises CT win probability)
- `lag_15__CT_flashes_last_5s`: coefficient `-0.001549` (lowers CT win probability)
- `lag_00__T3__molly`: coefficient `-0.001511` (lowers CT win probability)
- `lag_06__CT_smokes_last_5s`: coefficient `0.001500` (raises CT win probability)
- `lag_13__T3__smoke`: coefficient `-0.001492` (lowers CT win probability)
- `lag_07__T_B_site_active_smokes`: coefficient `0.001403` (raises CT win probability)
- `lag_14__T2__smoke`: coefficient `-0.001339` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_TRUCK`: coefficient `0.004976` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003587` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.003366` (raises CT win probability)
- `lag_12__T_place_SIDEALLEY`: coefficient `0.003312` (raises CT win probability)
- `lag_02__CT_place_TRUCK`: coefficient `0.002873` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002872` (raises CT win probability)
- `lag_08__CT_shots_fired_sum`: coefficient `0.002731` (raises CT win probability)
- `lag_00__CT4__duck_amount`: coefficient `0.002555` (raises CT win probability)
- `lag_07__CT_place_CONNECTOR`: coefficient `-0.002530` (lowers CT win probability)
- `lag_04__CT_place_TRUCK`: coefficient `-0.002500` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `63702`, seconds `55.00`, LSTM delta `-0.4199`

Top all feature movements:
- `lag_00__CT_place_TRUCK`: contribution `-0.032093`
- `lag_13__CT_place_TRUCK`: contribution `-0.014514`
- `lag_08__CT_shots_fired_sum`: contribution `-0.013282`
- `lag_01__CT_place_JUNGLE`: contribution `-0.012885`
- `lag_11__CT_place_TRUCK`: contribution `-0.011878`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `63414`, seconds `50.50`, LSTM delta `+0.2617`

Top all feature movements:
- `lag_02__CT_place_TRUCK`: contribution `+0.018530`
- `lag_04__CT_place_TRUCK`: contribution `+0.016123`
- `lag_00__CT_shots_fired_sum`: contribution `+0.011691`
- `lag_12__T_place_SIDEALLEY`: contribution `+0.010559`
- `lag_12__CT_place_TRUCK`: contribution `+0.010110`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `60726`, seconds `8.50`, LSTM delta `-0.1938`

Top all feature movements:
- `lag_06__CT_smokes_last_5s`: contribution `-0.025924`
- `lag_13__T_he_last_5s`: contribution `-0.023979`
- `lag_15__CT_flashes_last_5s`: contribution `-0.017036`
- `lag_00__CT_shots_fired_sum`: contribution `-0.016368`
- `lag_15__CT_smokes_last_5s`: contribution `-0.015714`

Top utility-only movements:
- `lag_06__CT_smokes_last_5s`: contribution `-0.025924`
- `lag_13__T_he_last_5s`: contribution `-0.023979`
- `lag_15__CT_flashes_last_5s`: contribution `-0.017036`
- `lag_15__CT_smokes_last_5s`: contribution `-0.015714`
- `lag_03__T_he_last_5s`: contribution `-0.014917`

### tick `60694`, seconds `8.00`, LSTM delta `+0.1797`

Top all feature movements:
- `lag_12__T_place_SIDEALLEY`: contribution `+0.021118`
- `lag_15__CT_smokes_last_5s`: contribution `-0.015714`
- `lag_12__T_he_last_5s`: contribution `+0.015081`
- `lag_04__CT_smokes_last_5s`: contribution `+0.013011`
- `lag_00__CT_shots_fired_sum`: contribution `+0.011691`

Top utility-only movements:
- `lag_15__CT_smokes_last_5s`: contribution `-0.015714`
- `lag_12__T_he_last_5s`: contribution `+0.015081`
- `lag_04__CT_smokes_last_5s`: contribution `+0.013011`
- `lag_02__T_he_last_5s`: contribution `+0.010857`
- `lag_14__CT_flashes_last_5s`: contribution `+0.010653`

### tick `60566`, seconds `6.00`, LSTM delta `-0.0725`

Top all feature movements:
- `lag_01__CT_smokes_last_5s`: contribution `-0.022985`
- `lag_00__CT_smokes_last_5s`: contribution `-0.010185`
- `lag_00__CT_flashes_last_5s`: contribution `-0.009738`
- `lag_11__CT_smokes_last_5s`: contribution `-0.009707`
- `lag_06__T_place_SIDEALLEY`: contribution `+0.008357`

Top utility-only movements:
- `lag_01__CT_smokes_last_5s`: contribution `-0.022985`
- `lag_00__CT_smokes_last_5s`: contribution `-0.010185`
- `lag_00__CT_flashes_last_5s`: contribution `-0.009738`
- `lag_11__CT_smokes_last_5s`: contribution `-0.009707`
- `lag_10__CT_smokes_last_5s`: contribution `-0.007356`
