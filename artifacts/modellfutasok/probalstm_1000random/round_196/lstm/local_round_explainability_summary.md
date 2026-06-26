# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-falcons-bo3-yayytstbo8IxTFlUpfbUPR/mouz-vs-falcons-m1-train.csv`
- round_num: `20`

## Largest probability jumps

- tick `174186`, seconds `52.00`, LSTM `0.0632`, delta `-0.2039`
- tick `174122`, seconds `51.00`, LSTM `0.3739`, delta `-0.1790`
- tick `172394`, seconds `24.00`, LSTM `0.5634`, delta `+0.1071`
- tick `174154`, seconds `51.50`, LSTM `0.2671`, delta `-0.1068`
- tick `172362`, seconds `23.50`, LSTM `0.4563`, delta `+0.0583`
- tick `172074`, seconds `19.00`, LSTM `0.4096`, delta `+0.0331`
- tick `171722`, seconds `13.50`, LSTM `0.3919`, delta `-0.0303`
- tick `172234`, seconds `21.50`, LSTM `0.4050`, delta `+0.0241`
- tick `172618`, seconds `27.50`, LSTM `0.5594`, delta `+0.0216`
- tick `172042`, seconds `18.50`, LSTM `0.3766`, delta `-0.0212`

## Top 15 local ridge features

- `lag_00__CT_place_TMAIN`: coefficient `0.004845`, |coef| `0.004845`
- `lag_01__CT_place_TMAIN`: coefficient `0.002537`, |coef| `0.002537`
- `lag_02__CT_place_TMAIN`: coefficient `0.001919`, |coef| `0.001919`
- `lag_02__T_place_DUMPSTER`: coefficient `-0.001886`, |coef| `0.001886`
- `lag_00__damage_diff_last_5s`: coefficient `0.001878`, |coef| `0.001878`
- `lag_00__T_kills_last_3s`: coefficient `-0.001661`, |coef| `0.001661`
- `lag_00__kill_diff_last_3s`: coefficient `0.001595`, |coef| `0.001595`
- `lag_02__T_place_ALLEY`: coefficient `0.001569`, |coef| `0.001569`
- `lag_03__T_place_ALLEY`: coefficient `0.001540`, |coef| `0.001540`
- `lag_00__T_damage_last_5s`: coefficient `-0.001539`, |coef| `0.001539`
- `lag_12__CT_place_LONGDOG`: coefficient `-0.001396`, |coef| `0.001396`
- `lag_04__T_place_ALLEY`: coefficient `0.001295`, |coef| `0.001295`
- `lag_03__CT3__duck_amount`: coefficient `0.001291`, |coef| `0.001291`
- `lag_04__CT5__duck_amount`: coefficient `-0.001271`, |coef| `0.001271`
- `lag_10__CT3__duck_amount`: coefficient `0.001251`, |coef| `0.001251`

## Top 10 utility ridge features

- `lag_00__CT3__smoke`: coefficient `0.001045` (raises CT win probability)
- `lag_01__CT3__smoke`: coefficient `0.000896` (raises CT win probability)
- `lag_02__CT3__smoke`: coefficient `0.000767` (raises CT win probability)
- `lag_08__T_A_site_active_infernos`: coefficient `-0.000645` (lowers CT win probability)
- `lag_00__CT2__molly`: coefficient `0.000624` (raises CT win probability)
- `lag_07__T_A_site_active_infernos`: coefficient `-0.000610` (lowers CT win probability)
- `lag_04__CT_B_site_active_smokes`: coefficient `0.000558` (raises CT win probability)
- `lag_15__T_A_site_active_infernos`: coefficient `-0.000557` (lowers CT win probability)
- `lag_04__CT_A_site_active_smokes`: coefficient `0.000535` (raises CT win probability)
- `lag_00__CT3__utility_total`: coefficient `0.000522` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_TMAIN`: coefficient `0.004845` (raises CT win probability)
- `lag_01__CT_place_TMAIN`: coefficient `0.002537` (raises CT win probability)
- `lag_02__CT_place_TMAIN`: coefficient `0.001919` (raises CT win probability)
- `lag_02__T_place_DUMPSTER`: coefficient `-0.001886` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001878` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001661` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001595` (raises CT win probability)
- `lag_02__T_place_ALLEY`: coefficient `0.001569` (raises CT win probability)
- `lag_03__T_place_ALLEY`: coefficient `0.001540` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001539` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `174186`, seconds `52.00`, LSTM delta `-0.2039`

Top all feature movements:
- `lag_00__CT_place_TMAIN`: contribution `-0.053690`
- `lag_02__CT_place_TMAIN`: contribution `-0.021261`
- `lag_04__T_place_DUMPSTER`: contribution `-0.010705`
- `lag_04__T_place_ALLEY`: contribution `-0.005487`
- `lag_00__T_kills_last_3s`: contribution `-0.005263`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `174122`, seconds `51.00`, LSTM delta `-0.1790`

Top all feature movements:
- `lag_00__CT_place_TMAIN`: contribution `-0.053690`
- `lag_02__T_place_DUMPSTER`: contribution `-0.017150`
- `lag_02__T_place_ALLEY`: contribution `-0.006646`
- `lag_00__T_kills_last_3s`: contribution `-0.005263`
- `lag_04__CT5__duck_amount`: contribution `-0.004797`

Top utility-only movements:
- `lag_00__CT3__smoke`: contribution `-0.002312`

### tick `172394`, seconds `24.00`, LSTM delta `+0.1071`

Top all feature movements:
- `lag_04__T_place_DUMPSTER`: contribution `-0.010705`
- `lag_12__CT_place_LONGDOG`: contribution `+0.009107`
- `lag_15__CT_place_ELECTRICALBOX`: contribution `+0.008445`
- `lag_10__CT3__duck_amount`: contribution `+0.004655`
- `lag_01__CT3__duck_amount`: contribution `+0.004554`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `174154`, seconds `51.50`, LSTM delta `-0.1068`

Top all feature movements:
- `lag_01__CT_place_TMAIN`: contribution `-0.028114`
- `lag_03__T_place_DUMPSTER`: contribution `-0.011335`
- `lag_03__T_place_ALLEY`: contribution `-0.006523`
- `lag_05__CT5__duck_amount`: contribution `-0.004461`
- `lag_10__CT3__duck_amount`: contribution `-0.004415`

Top utility-only movements:
- `lag_01__CT3__smoke`: contribution `-0.001982`

### tick `172362`, seconds `23.50`, LSTM delta `+0.0583`

Top all feature movements:
- `lag_03__T_place_DUMPSTER`: contribution `-0.011335`
- `lag_14__CT_place_ELECTRICALBOX`: contribution `+0.011282`
- `lag_11__CT_place_LONGDOG`: contribution `+0.006692`
- `lag_00__CT3__duck_amount`: contribution `+0.004163`
- `lag_07__CT_place_ELECTRICALBOX`: contribution `+0.003572`

Top utility-only movements:
- No utility movement among the top local contributors.
