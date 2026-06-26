# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-aurora-vs-falcons-bo3-5oHSxtVT-5F3Op7ZcgBMjW/aurora-vs-falcons-m1-inferno.csv`
- round_num: `1`

## Largest probability jumps

- tick `5567`, seconds `67.50`, LSTM `0.4693`, delta `+0.2745`
- tick `6175`, seconds `77.00`, LSTM `0.0725`, delta `-0.2304`
- tick `5343`, seconds `64.00`, LSTM `0.2913`, delta `-0.2199`
- tick `6111`, seconds `76.00`, LSTM `0.3384`, delta `-0.1260`
- tick `5375`, seconds `64.50`, LSTM `0.1914`, delta `-0.0999`
- tick `1279`, seconds `0.50`, LSTM `0.4957`, delta `+0.0640`
- tick `5663`, seconds `69.00`, LSTM `0.4519`, delta `-0.0458`
- tick `6239`, seconds `78.00`, LSTM `0.0412`, delta `-0.0398`
- tick `6143`, seconds `76.50`, LSTM `0.3028`, delta `-0.0356`
- tick `5695`, seconds `69.50`, LSTM `0.4822`, delta `+0.0303`

## Top 15 local ridge features

- `lag_00__damage_diff_last_5s`: coefficient `0.004119`, |coef| `0.004119`
- `lag_00__kill_diff_last_3s`: coefficient `0.003967`, |coef| `0.003967`
- `lag_00__T_place_BALCONY`: coefficient `-0.003675`, |coef| `0.003675`
- `lag_00__T_kills_last_3s`: coefficient `-0.003627`, |coef| `0.003627`
- `lag_00__CT_place_RUINS`: coefficient `-0.003502`, |coef| `0.003502`
- `lag_00__CT_place_APARTMENTS`: coefficient `0.003343`, |coef| `0.003343`
- `lag_13__T2__duck_amount`: coefficient `-0.003237`, |coef| `0.003237`
- `lag_01__T_kills_last_3s`: coefficient `-0.003066`, |coef| `0.003066`
- `lag_03__CT_place_RUINS`: coefficient `0.003035`, |coef| `0.003035`
- `lag_00__T_damage_last_5s`: coefficient `-0.002864`, |coef| `0.002864`
- `lag_06__T_place_ARCH`: coefficient `0.002782`, |coef| `0.002782`
- `lag_01__kill_diff_last_3s`: coefficient `0.002566`, |coef| `0.002566`
- `lag_08__T_place_ARCH`: coefficient `0.002530`, |coef| `0.002530`
- `lag_07__CT_place_APARTMENTS`: coefficient `-0.002480`, |coef| `0.002480`
- `lag_07__T_kills_last_3s`: coefficient `0.002372`, |coef| `0.002372`

## Top 10 utility ridge features

- `lag_02__T_utility_damage_last_5s`: coefficient `-0.001877` (lowers CT win probability)
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.001530` (lowers CT win probability)
- `lag_07__T1__smoke`: coefficient `0.001456` (raises CT win probability)
- `lag_01__T1__flash`: coefficient `0.001403` (raises CT win probability)
- `lag_03__T2__flash`: coefficient `0.001342` (raises CT win probability)
- `lag_14__T1__smoke`: coefficient `-0.001325` (lowers CT win probability)
- `lag_04__T_A_site_active_smokes`: coefficient `-0.001247` (lowers CT win probability)
- `lag_02__utility_damage_diff_last_5s`: coefficient `0.001187` (raises CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `-0.001113` (lowers CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000968` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__damage_diff_last_5s`: coefficient `0.004119` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003967` (raises CT win probability)
- `lag_00__T_place_BALCONY`: coefficient `-0.003675` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003627` (lowers CT win probability)
- `lag_00__CT_place_RUINS`: coefficient `-0.003502` (lowers CT win probability)
- `lag_00__CT_place_APARTMENTS`: coefficient `0.003343` (raises CT win probability)
- `lag_13__T2__duck_amount`: coefficient `-0.003237` (lowers CT win probability)
- `lag_01__T_kills_last_3s`: coefficient `-0.003066` (lowers CT win probability)
- `lag_03__CT_place_RUINS`: coefficient `0.003035` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002864` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `5567`, seconds `67.50`, LSTM delta `+0.2745`

Top all feature movements:
- `lag_13__T2__duck_amount`: contribution `+0.012376`
- `lag_00__CT_place_RUINS`: contribution `+0.012235`
- `lag_03__CT_place_RUINS`: contribution `+0.010602`
- `lag_01__T_kills_last_3s`: contribution `+0.009713`
- `lag_00__kill_diff_last_3s`: contribution `+0.009549`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `6175`, seconds `77.00`, LSTM delta `-0.2304`

Top all feature movements:
- `lag_00__T_place_BALCONY`: contribution `-0.050543`
- `lag_08__T_place_ARCH`: contribution `-0.023537`
- `lag_00__T_kills_last_3s`: contribution `-0.011492`
- `lag_02__T_utility_damage_last_5s`: contribution `-0.011253`
- `lag_00__kill_diff_last_3s`: contribution `-0.009549`

Top utility-only movements:
- `lag_02__T_utility_damage_last_5s`: contribution `-0.011253`
- `lag_02__utility_damage_diff_last_5s`: contribution `-0.004502`

### tick `5343`, seconds `64.00`, LSTM delta `-0.2199`

Top all feature movements:
- `lag_00__CT_place_APARTMENTS`: contribution `-0.012840`
- `lag_13__T2__duck_amount`: contribution `-0.012376`
- `lag_00__T_kills_last_3s`: contribution `-0.011492`
- `lag_00__kill_diff_last_3s`: contribution `-0.009549`
- `lag_00__damage_diff_last_5s`: contribution `-0.009293`

Top utility-only movements:
- `lag_07__T1__smoke`: contribution `-0.003142`

### tick `6111`, seconds `76.00`, LSTM delta `-0.1260`

Top all feature movements:
- `lag_06__T_place_ARCH`: contribution `-0.025883`
- `lag_00__T_utility_damage_last_5s`: contribution `-0.009177`
- `lag_07__T_kills_last_3s`: contribution `-0.007513`
- `lag_13__CT_place_PIT`: contribution `-0.005687`
- `lag_11__CT_place_RUINS`: contribution `-0.004368`

Top utility-only movements:
- `lag_00__T_utility_damage_last_5s`: contribution `-0.009177`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.003670`

### tick `5375`, seconds `64.50`, LSTM delta `-0.0999`

Top all feature movements:
- `lag_00__CT_place_RUINS`: contribution `-0.012235`
- `lag_01__T_kills_last_3s`: contribution `-0.009713`
- `lag_13__T3__duck_amount`: contribution `-0.007077`
- `lag_01__CT_place_APARTMENTS`: contribution `-0.006809`
- `lag_01__kill_diff_last_3s`: contribution `-0.006175`

Top utility-only movements:
- No utility movement among the top local contributors.
