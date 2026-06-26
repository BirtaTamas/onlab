# Local Round Explainability

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-ninjas-in-pyjamas-vs-natus-vincere-bo3-NyGFAYn592Hu0YfljTVG0N/ninjas-in-pyjamas-vs-natus-vincere-m2-inferno.csv`
- round_num: `13`

## Largest probability jumps

- tick `121031`, seconds `38.50`, LSTM `0.0790`, delta `-0.3517`
- tick `124231`, seconds `88.50`, LSTM `0.0827`, delta `-0.2630`
- tick `124135`, seconds `87.00`, LSTM `0.2549`, delta `+0.1507`
- tick `124103`, seconds `86.50`, LSTM `0.1042`, delta `+0.0723`
- tick `124199`, seconds `88.00`, LSTM `0.3457`, delta `+0.0530`
- tick `120999`, seconds `38.00`, LSTM `0.4307`, delta `-0.0449`
- tick `121063`, seconds `39.00`, LSTM `0.0356`, delta `-0.0434`
- tick `120071`, seconds `23.50`, LSTM `0.4562`, delta `-0.0399`
- tick `124167`, seconds `87.50`, LSTM `0.2928`, delta `+0.0379`
- tick `120135`, seconds `24.50`, LSTM `0.4720`, delta `+0.0219`

## Top 15 local ridge features

- `lag_00__closest_enemy_dist_diff`: coefficient `0.005432`, |coef| `0.005432`
- `lag_00__CT_place_BACKALLEY`: coefficient `0.003989`, |coef| `0.003989`
- `lag_00__kill_diff_last_3s`: coefficient `0.003846`, |coef| `0.003846`
- `lag_00__T_kills_last_3s`: coefficient `-0.003543`, |coef| `0.003543`
- `lag_00__spread_diff`: coefficient `0.003238`, |coef| `0.003238`
- `lag_00__CT_place_APARTMENTS`: coefficient `0.003219`, |coef| `0.003219`
- `lag_02__CT_place_BALCONY`: coefficient `0.003193`, |coef| `0.003193`
- `lag_00__T_closest_enemy_dist`: coefficient `-0.003132`, |coef| `0.003132`
- `lag_00__CT_spread_xy`: coefficient `0.003073`, |coef| `0.003073`
- `lag_04__T_place_BACKALLEY`: coefficient `0.002861`, |coef| `0.002861`
- `lag_08__CT_place_BALCONY`: coefficient `-0.002533`, |coef| `0.002533`
- `lag_12__T2__duck_amount`: coefficient `0.002493`, |coef| `0.002493`
- `lag_01__CT_place_APARTMENTS`: coefficient `0.002376`, |coef| `0.002376`
- `lag_06__CT_place_BALCONY`: coefficient `0.002341`, |coef| `0.002341`
- `lag_01__T_kills_last_3s`: coefficient `-0.002272`, |coef| `0.002272`

## Top 10 utility ridge features

- `lag_09__CT_A_site_active_smokes`: coefficient `-0.000792` (lowers CT win probability)
- `lag_12__CT_A_site_active_smokes`: coefficient `0.000529` (raises CT win probability)
- `lag_09__CT_active_smokes`: coefficient `-0.000521` (lowers CT win probability)
- `lag_00__CT4__smoke`: coefficient `-0.000484` (lowers CT win probability)
- `lag_08__CT_A_site_active_smokes`: coefficient `-0.000453` (lowers CT win probability)
- `lag_01__CT4__smoke`: coefficient `-0.000426` (lowers CT win probability)
- `lag_12__CT_active_smokes`: coefficient `0.000420` (raises CT win probability)
- `lag_11__CT_A_site_active_smokes`: coefficient `-0.000293` (lowers CT win probability)
- `lag_12__active_smokes_total`: coefficient `0.000291` (raises CT win probability)
- `lag_08__CT_active_smokes`: coefficient `-0.000274` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__closest_enemy_dist_diff`: coefficient `0.005432` (raises CT win probability)
- `lag_00__CT_place_BACKALLEY`: coefficient `0.003989` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003846` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003543` (lowers CT win probability)
- `lag_00__spread_diff`: coefficient `0.003238` (raises CT win probability)
- `lag_00__CT_place_APARTMENTS`: coefficient `0.003219` (raises CT win probability)
- `lag_02__CT_place_BALCONY`: coefficient `0.003193` (raises CT win probability)
- `lag_00__T_closest_enemy_dist`: coefficient `-0.003132` (lowers CT win probability)
- `lag_00__CT_spread_xy`: coefficient `0.003073` (raises CT win probability)
- `lag_04__T_place_BACKALLEY`: coefficient `0.002861` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `121031`, seconds `38.50`, LSTM delta `-0.3517`

Top all feature movements:
- `lag_00__closest_enemy_dist_diff`: contribution `-0.029404`
- `lag_02__CT_place_BALCONY`: contribution `-0.020493`
- `lag_08__CT_place_BALCONY`: contribution `-0.016255`
- `lag_06__CT_place_BALCONY`: contribution `-0.015021`
- `lag_00__CT_place_APARTMENTS`: contribution `-0.012365`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `124231`, seconds `88.50`, LSTM delta `-0.2630`

Top all feature movements:
- `lag_00__CT_place_BACKALLEY`: contribution `-0.059809`
- `lag_00__closest_enemy_dist_diff`: contribution `-0.020230`
- `lag_12__T_place_BALCONY`: contribution `-0.018746`
- `lag_00__T_closest_enemy_dist`: contribution `-0.011660`
- `lag_00__CT_spread_xy`: contribution `-0.011351`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `124135`, seconds `87.00`, LSTM delta `+0.1507`

Top all feature movements:
- `lag_07__T_place_BALCONY`: contribution `+0.017565`
- `lag_00__kill_diff_last_3s`: contribution `+0.009258`
- `lag_04__T_place_BACKALLEY`: contribution `+0.008655`
- `lag_04__CT_place_ARCH`: contribution `+0.008153`
- `lag_03__T_place_BACKALLEY`: contribution `+0.006112`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `124103`, seconds `86.50`, LSTM delta `+0.0723`

Top all feature movements:
- `lag_06__T_place_BALCONY`: contribution `+0.012818`
- `lag_08__T_place_BALCONY`: contribution `-0.010151`
- `lag_14__T_place_BALCONY`: contribution `+0.009586`
- `lag_00__kill_diff_last_3s`: contribution `+0.009258`
- `lag_03__T_place_BACKALLEY`: contribution `+0.006112`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `124199`, seconds `88.00`, LSTM delta `+0.0530`

Top all feature movements:
- `lag_11__T_place_BALCONY`: contribution `-0.014909`
- `lag_12__T2__duck_amount`: contribution `+0.009533`
- `lag_03__T_place_BACKALLEY`: contribution `-0.006112`
- `lag_11__T4__duck_amount`: contribution `+0.005781`
- `lag_09__T_place_BALCONY`: contribution `+0.005710`

Top utility-only movements:
- No utility movement among the top local contributors.
