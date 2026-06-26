# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-vitality-vs-virtuspro-bo3-8Z0L17IYJlstHvIADVy9G9/vitality-vs-virtus-pro-m3-mirage.csv`
- round_num: `13`

## Largest probability jumps

- tick `113609`, seconds `56.00`, LSTM `0.4286`, delta `+0.2521`
- tick `113641`, seconds `56.50`, LSTM `0.6656`, delta `+0.2370`
- tick `113929`, seconds `61.00`, LSTM `0.1330`, delta `-0.2351`
- tick `113545`, seconds `55.00`, LSTM `0.2148`, delta `-0.2092`
- tick `113865`, seconds `60.00`, LSTM `0.5402`, delta `-0.1963`
- tick `113769`, seconds `58.50`, LSTM `0.7631`, delta `+0.1889`
- tick `113897`, seconds `60.50`, LSTM `0.3681`, delta `-0.1721`
- tick `113673`, seconds `57.00`, LSTM `0.5714`, delta `-0.0942`
- tick `113321`, seconds `51.50`, LSTM `0.4344`, delta `-0.0744`
- tick `113513`, seconds `54.50`, LSTM `0.4240`, delta `-0.0594`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003364`, |coef| `0.003364`
- `lag_10__CT_flash_duration_sum`: coefficient `0.002898`, |coef| `0.002898`
- `lag_10__CT3__flash_duration`: coefficient `0.002625`, |coef| `0.002625`
- `lag_10__CT4__flash_duration`: coefficient `0.002416`, |coef| `0.002416`
- `lag_00__T_kills_last_3s`: coefficient `-0.002416`, |coef| `0.002416`
- `lag_09__T5__flash_duration`: coefficient `0.002346`, |coef| `0.002346`
- `lag_10__CT_flashed_players`: coefficient `0.002179`, |coef| `0.002179`
- `lag_11__CT3__flash_duration`: coefficient `0.002164`, |coef| `0.002164`
- `lag_00__damage_diff_last_5s`: coefficient `0.002155`, |coef| `0.002155`
- `lag_12__CT1__duck_amount`: coefficient `-0.002073`, |coef| `0.002073`
- `lag_09__CT_place_SHOP`: coefficient `0.002022`, |coef| `0.002022`
- `lag_01__kill_diff_last_3s`: coefficient `0.001974`, |coef| `0.001974`
- `lag_09__T3__flash_duration`: coefficient `0.001968`, |coef| `0.001968`
- `lag_00__CT3__flash_duration`: coefficient `0.001864`, |coef| `0.001864`
- `lag_07__CT2__flash_duration`: coefficient `-0.001855`, |coef| `0.001855`

## Top 10 utility ridge features

- `lag_10__CT_flash_duration_sum`: coefficient `0.002898` (raises CT win probability)
- `lag_10__CT3__flash_duration`: coefficient `0.002625` (raises CT win probability)
- `lag_10__CT4__flash_duration`: coefficient `0.002416` (raises CT win probability)
- `lag_09__T5__flash_duration`: coefficient `0.002346` (raises CT win probability)
- `lag_11__CT3__flash_duration`: coefficient `0.002164` (raises CT win probability)
- `lag_09__T3__flash_duration`: coefficient `0.001968` (raises CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `0.001864` (raises CT win probability)
- `lag_07__CT2__flash_duration`: coefficient `-0.001855` (lowers CT win probability)
- `lag_02__CT2__flash_duration`: coefficient `0.001831` (raises CT win probability)
- `lag_09__T_flash_duration_sum`: coefficient `0.001764` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003364` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002416` (lowers CT win probability)
- `lag_10__CT_flashed_players`: coefficient `0.002179` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002155` (raises CT win probability)
- `lag_12__CT1__duck_amount`: coefficient `-0.002073` (lowers CT win probability)
- `lag_09__CT_place_SHOP`: coefficient `0.002022` (raises CT win probability)
- `lag_01__kill_diff_last_3s`: coefficient `0.001974` (raises CT win probability)
- `lag_10__CT4__duck_amount`: coefficient `-0.001845` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001834` (raises CT win probability)
- `lag_11__CT_flashed_players`: coefficient `0.001796` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `113609`, seconds `56.00`, LSTM delta `+0.2521`

Top all feature movements:
- `lag_10__CT_flash_duration_sum`: contribution `+0.017005`
- `lag_10__CT3__flash_duration`: contribution `+0.015890`
- `lag_10__CT_flashed_players`: contribution `+0.014317`
- `lag_10__CT4__flash_duration`: contribution `+0.014022`
- `lag_09__T5__flash_duration`: contribution `+0.013171`

Top utility-only movements:
- `lag_10__CT_flash_duration_sum`: contribution `+0.017005`
- `lag_10__CT3__flash_duration`: contribution `+0.015890`
- `lag_10__CT4__flash_duration`: contribution `+0.014022`
- `lag_09__T5__flash_duration`: contribution `+0.013171`
- `lag_09__CT2__flash_duration`: contribution `+0.010373`

### tick `113641`, seconds `56.50`, LSTM delta `+0.2370`

Top all feature movements:
- `lag_11__CT3__flash_duration`: contribution `+0.013099`
- `lag_11__CT_flashed_players`: contribution `+0.011798`
- `lag_01__CT4__flash_duration`: contribution `+0.010159`
- `lag_10__CT2__flash_duration`: contribution `+0.009637`
- `lag_00__T5__flash_duration`: contribution `+0.009581`

Top utility-only movements:
- `lag_11__CT3__flash_duration`: contribution `+0.013099`
- `lag_01__CT4__flash_duration`: contribution `+0.010159`
- `lag_10__CT2__flash_duration`: contribution `+0.009637`
- `lag_00__T5__flash_duration`: contribution `+0.009581`
- `lag_10__CT_flash_duration_sum`: contribution `+0.009175`

### tick `113929`, seconds `61.00`, LSTM delta `-0.2351`

Top all feature movements:
- `lag_10__CT4__flash_duration`: contribution `-0.014022`
- `lag_09__T5__flash_duration`: contribution `-0.013171`
- `lag_03__CT2__flash_duration`: contribution `-0.011586`
- `lag_00__damage_diff_last_5s`: contribution `-0.008652`
- `lag_00__kill_diff_last_3s`: contribution `-0.008097`

Top utility-only movements:
- `lag_10__CT4__flash_duration`: contribution `-0.014022`
- `lag_09__T5__flash_duration`: contribution `-0.013171`
- `lag_03__CT2__flash_duration`: contribution `-0.011586`
- `lag_10__CT_flash_duration_sum`: contribution `-0.007522`
- `lag_09__T_flash_duration_sum`: contribution `-0.004023`

### tick `113545`, seconds `55.00`, LSTM delta `-0.2092`

Top all feature movements:
- `lag_07__CT2__flash_duration`: contribution `-0.012628`
- `lag_00__CT3__flash_duration`: contribution `-0.011590`
- `lag_00__kill_diff_last_3s`: contribution `-0.008097`
- `lag_12__CT1__duck_amount`: contribution `-0.007908`
- `lag_08__CT3__flash_duration`: contribution `-0.007760`

Top utility-only movements:
- `lag_07__CT2__flash_duration`: contribution `-0.012628`
- `lag_00__CT3__flash_duration`: contribution `-0.011590`
- `lag_08__CT3__flash_duration`: contribution `-0.007760`
- `lag_08__CT_flash_duration_sum`: contribution `-0.005756`
- `lag_07__CT_flash_duration_sum`: contribution `-0.005053`

### tick `113865`, seconds `60.00`, LSTM delta `-0.1963`

Top all feature movements:
- `lag_10__CT3__flash_duration`: contribution `-0.016325`
- `lag_01__CT2__flash_duration`: contribution `-0.013116`
- `lag_09__CT_place_SHOP`: contribution `-0.010140`
- `lag_09__T3__flash_duration`: contribution `-0.009135`
- `lag_06__CT_place_SHOP`: contribution `-0.008680`

Top utility-only movements:
- `lag_10__CT3__flash_duration`: contribution `-0.016325`
- `lag_01__CT2__flash_duration`: contribution `-0.013116`
- `lag_09__T3__flash_duration`: contribution `-0.009135`
- `lag_10__CT_flash_duration_sum`: contribution `-0.008062`
- `lag_09__T_flash_duration_sum`: contribution `-0.003413`
