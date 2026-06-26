# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-mouz-bo3-Ko5VJMvyF1OsCx2TbVU9pb/vitality-vs-mouz-m1-inferno.csv`
- round_num: `1`

## Largest probability jumps

- tick `4474`, seconds `50.00`, LSTM `0.5281`, delta `+0.2840`
- tick `4538`, seconds `51.00`, LSTM `0.5874`, delta `+0.2468`
- tick `4602`, seconds `52.00`, LSTM `0.7871`, delta `+0.2245`
- tick `4506`, seconds `50.50`, LSTM `0.3407`, delta `-0.1874`
- tick `4442`, seconds `49.50`, LSTM `0.2441`, delta `-0.1271`
- tick `4666`, seconds `53.00`, LSTM `0.9014`, delta `+0.0967`
- tick `4346`, seconds `48.00`, LSTM `0.4416`, delta `-0.0634`
- tick `2970`, seconds `26.50`, LSTM `0.4359`, delta `+0.0381`
- tick `4410`, seconds `49.00`, LSTM `0.3711`, delta `-0.0365`
- tick `4378`, seconds `48.50`, LSTM `0.4077`, delta `-0.0339`

## Top 15 local ridge features

- `lag_00__T_place_BALCONY`: coefficient `-0.004253`, |coef| `0.004253`
- `lag_02__CT5__flash_duration`: coefficient `0.003044`, |coef| `0.003044`
- `lag_02__T_place_BALCONY`: coefficient `-0.002479`, |coef| `0.002479`
- `lag_00__CT5__flash_duration`: coefficient `0.001816`, |coef| `0.001816`
- `lag_14__T4__is_walking`: coefficient `-0.001800`, |coef| `0.001800`
- `lag_00__CT_kills_last_3s`: coefficient `0.001799`, |coef| `0.001799`
- `lag_00__kill_diff_last_3s`: coefficient `0.001765`, |coef| `0.001765`
- `lag_02__CT_flash_duration_sum`: coefficient `0.001630`, |coef| `0.001630`
- `lag_02__CT_flashed_players`: coefficient `0.001630`, |coef| `0.001630`
- `lag_01__T5__duck_amount`: coefficient `0.001617`, |coef| `0.001617`
- `lag_00__damage_diff_last_5s`: coefficient `0.001543`, |coef| `0.001543`
- `lag_02__CT5__duck_amount`: coefficient `-0.001531`, |coef| `0.001531`
- `lag_04__CT5__flash_duration`: coefficient `0.001523`, |coef| `0.001523`
- `lag_08__T_place_BALCONY`: coefficient `0.001419`, |coef| `0.001419`
- `lag_06__T_place_TOPOFMID`: coefficient `-0.001379`, |coef| `0.001379`

## Top 10 utility ridge features

- `lag_02__CT5__flash_duration`: coefficient `0.003044` (raises CT win probability)
- `lag_00__CT5__flash_duration`: coefficient `0.001816` (raises CT win probability)
- `lag_02__CT_flash_duration_sum`: coefficient `0.001630` (raises CT win probability)
- `lag_04__CT5__flash_duration`: coefficient `0.001523` (raises CT win probability)
- `lag_01__CT5__flash_duration`: coefficient `-0.001355` (lowers CT win probability)
- `lag_06__CT5__flash_duration`: coefficient `0.001180` (raises CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `0.001163` (raises CT win probability)
- `lag_00__T5__flash_duration`: coefficient `0.001108` (raises CT win probability)
- `lag_03__CT5__flash_duration`: coefficient `-0.001091` (lowers CT win probability)
- `lag_01__CT_utility_damage_last_5s`: coefficient `0.001087` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_BALCONY`: coefficient `-0.004253` (lowers CT win probability)
- `lag_02__T_place_BALCONY`: coefficient `-0.002479` (lowers CT win probability)
- `lag_14__T4__is_walking`: coefficient `-0.001800` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001799` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001765` (raises CT win probability)
- `lag_02__CT_flashed_players`: coefficient `0.001630` (raises CT win probability)
- `lag_01__T5__duck_amount`: coefficient `0.001617` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001543` (raises CT win probability)
- `lag_02__CT5__duck_amount`: coefficient `-0.001531` (lowers CT win probability)
- `lag_08__T_place_BALCONY`: coefficient `0.001419` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `4474`, seconds `50.00`, LSTM delta `+0.2840`

Top all feature movements:
- `lag_00__T_place_BALCONY`: contribution `+0.058487`
- `lag_02__CT5__flash_duration`: contribution `+0.016311`
- `lag_01__T_place_BALCONY`: contribution `+0.014142`
- `lag_03__CT_place_LIBRARY`: contribution `+0.008671`
- `lag_02__CT_flash_duration_sum`: contribution `+0.007513`

Top utility-only movements:
- `lag_02__CT5__flash_duration`: contribution `+0.016311`
- `lag_02__CT_flash_duration_sum`: contribution `+0.007513`
- `lag_02__T2__flash_duration`: contribution `+0.005140`
- `lag_01__CT_utility_damage_last_5s`: contribution `+0.004069`
- `lag_02__CT2__flash_duration`: contribution `+0.003895`

### tick `4538`, seconds `51.00`, LSTM delta `+0.2468`

Top all feature movements:
- `lag_02__T_place_BALCONY`: contribution `+0.034086`
- `lag_00__CT5__flash_duration`: contribution `+0.011902`
- `lag_01__CT_place_BALCONY`: contribution `+0.008740`
- `lag_00__CT1__flash_duration`: contribution `+0.008471`
- `lag_04__CT5__flash_duration`: contribution `+0.008160`

Top utility-only movements:
- `lag_00__CT5__flash_duration`: contribution `+0.011902`
- `lag_00__CT1__flash_duration`: contribution `+0.008471`
- `lag_04__CT5__flash_duration`: contribution `+0.008160`
- `lag_01__CT5__flash_duration`: contribution `+0.007259`
- `lag_00__T5__flash_duration`: contribution `+0.005736`

### tick `4602`, seconds `52.00`, LSTM delta `+0.2245`

Top all feature movements:
- `lag_02__CT5__flash_duration`: contribution `+0.019948`
- `lag_08__T_place_BALCONY`: contribution `+0.019513`
- `lag_01__T_place_BALCONY`: contribution `-0.014142`
- `lag_02__CT1__flash_duration`: contribution `+0.007789`
- `lag_07__CT_place_LIBRARY`: contribution `+0.007467`

Top utility-only movements:
- `lag_02__CT5__flash_duration`: contribution `+0.019948`
- `lag_02__CT1__flash_duration`: contribution `+0.007789`
- `lag_02__T3__flash_duration`: contribution `+0.007064`
- `lag_06__CT5__flash_duration`: contribution `+0.006324`
- `lag_03__CT5__flash_duration`: contribution `+0.005846`

### tick `4506`, seconds `50.50`, LSTM delta `-0.1874`

Top all feature movements:
- `lag_02__T_place_BALCONY`: contribution `-0.034086`
- `lag_01__T_place_BALCONY`: contribution `-0.014142`
- `lag_00__CT5__flash_duration`: contribution `-0.009733`
- `lag_01__T5__duck_amount`: contribution `-0.006138`
- `lag_03__CT5__flash_duration`: contribution `-0.005846`

Top utility-only movements:
- `lag_00__CT5__flash_duration`: contribution `-0.009733`
- `lag_03__CT5__flash_duration`: contribution `-0.005846`
- `lag_03__CT_flash_duration_sum`: contribution `-0.003694`
- `lag_03__T2__flash_duration`: contribution `-0.003245`

### tick `4442`, seconds `49.50`, LSTM delta `-0.1271`

Top all feature movements:
- `lag_00__T_place_BALCONY`: contribution `-0.058487`
- `lag_01__CT5__flash_duration`: contribution `-0.007259`
- `lag_02__CT_place_LIBRARY`: contribution `-0.005552`
- `lag_01__CT_flash_duration_sum`: contribution `-0.004321`
- `lag_00__T5__duck_amount`: contribution `-0.003985`

Top utility-only movements:
- `lag_01__CT5__flash_duration`: contribution `-0.007259`
- `lag_01__CT_flash_duration_sum`: contribution `-0.004321`
- `lag_01__T2__flash_duration`: contribution `-0.002006`
- `lag_01__CT2__flash_duration`: contribution `-0.001789`
