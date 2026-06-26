# Local Round Explainability

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-faze-vs-g2-bo3-ldI7_iFRuThMOXF8zIbBwX/faze-vs-g2-m1-inferno.csv`
- round_num: `2`

## Largest probability jumps

- tick `20764`, seconds `20.00`, LSTM `0.1733`, delta `-0.2659`
- tick `20636`, seconds `18.00`, LSTM `0.3835`, delta `+0.0560`
- tick `20988`, seconds `23.50`, LSTM `0.0563`, delta `-0.0481`
- tick `20092`, seconds `9.50`, LSTM `0.2187`, delta `-0.0455`
- tick `20284`, seconds `12.50`, LSTM `0.2074`, delta `+0.0424`
- tick `20124`, seconds `10.00`, LSTM `0.1776`, delta `-0.0412`
- tick `19964`, seconds `7.50`, LSTM `0.3167`, delta `-0.0388`
- tick `20668`, seconds `18.50`, LSTM `0.4211`, delta `+0.0377`
- tick `21020`, seconds `24.00`, LSTM `0.0193`, delta `-0.0370`
- tick `19516`, seconds `0.50`, LSTM `0.2975`, delta `-0.0343`

## Top 15 local ridge features

- `lag_01__T_place_ARCH`: coefficient `-0.003109`, |coef| `0.003109`
- `lag_15__CT_place_QUAD`: coefficient `-0.001670`, |coef| `0.001670`
- `lag_00__CT_place_QUAD`: coefficient `0.001567`, |coef| `0.001567`
- `lag_11__CT5__flash_duration`: coefficient `-0.001213`, |coef| `0.001213`
- `lag_03__CT_place_LIBRARY`: coefficient `-0.001186`, |coef| `0.001186`
- `lag_11__T_flashed_players`: coefficient `-0.001166`, |coef| `0.001166`
- `lag_00__CT5__flash_duration`: coefficient `0.001113`, |coef| `0.001113`
- `lag_05__T_flashed_players`: coefficient `0.000991`, |coef| `0.000991`
- `lag_04__CT5__duck_amount`: coefficient `0.000883`, |coef| `0.000883`
- `lag_04__T_shots_fired_sum`: coefficient `0.000847`, |coef| `0.000847`
- `lag_00__T_shots_fired_sum`: coefficient `-0.000809`, |coef| `0.000809`
- `lag_12__T_A_site_active_infernos`: coefficient `0.000789`, |coef| `0.000789`
- `lag_15__CT_place_TOPOFMID`: coefficient `0.000787`, |coef| `0.000787`
- `lag_08__CT_he_last_5s`: coefficient `-0.000773`, |coef| `0.000773`
- `lag_08__T_place_ARCH`: coefficient `-0.000771`, |coef| `0.000771`

## Top 10 utility ridge features

- `lag_11__CT5__flash_duration`: coefficient `-0.001213` (lowers CT win probability)
- `lag_00__CT5__flash_duration`: coefficient `0.001113` (raises CT win probability)
- `lag_12__T_A_site_active_infernos`: coefficient `0.000789` (raises CT win probability)
- `lag_08__CT_he_last_5s`: coefficient `-0.000773` (lowers CT win probability)
- `lag_14__CT_he_last_5s`: coefficient `-0.000733` (lowers CT win probability)
- `lag_07__CT5__flash_duration`: coefficient `0.000710` (raises CT win probability)
- `lag_06__T_A_site_active_infernos`: coefficient `0.000591` (raises CT win probability)
- `lag_10__CT_he_last_5s`: coefficient `-0.000585` (lowers CT win probability)
- `lag_03__CT_he_last_5s`: coefficient `-0.000581` (lowers CT win probability)
- `lag_12__T_active_infernos`: coefficient `0.000568` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_01__T_place_ARCH`: coefficient `-0.003109` (lowers CT win probability)
- `lag_15__CT_place_QUAD`: coefficient `-0.001670` (lowers CT win probability)
- `lag_00__CT_place_QUAD`: coefficient `0.001567` (raises CT win probability)
- `lag_03__CT_place_LIBRARY`: coefficient `-0.001186` (lowers CT win probability)
- `lag_11__T_flashed_players`: coefficient `-0.001166` (lowers CT win probability)
- `lag_05__T_flashed_players`: coefficient `0.000991` (raises CT win probability)
- `lag_04__CT5__duck_amount`: coefficient `0.000883` (raises CT win probability)
- `lag_04__T_shots_fired_sum`: coefficient `0.000847` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.000809` (lowers CT win probability)
- `lag_15__CT_place_TOPOFMID`: coefficient `0.000787` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `20764`, seconds `20.00`, LSTM delta `-0.2659`

Top all feature movements:
- `lag_01__T_place_ARCH`: contribution `-0.057858`
- `lag_15__CT_place_QUAD`: contribution `-0.013162`
- `lag_00__CT_place_QUAD`: contribution `-0.012352`
- `lag_11__T_flashed_players`: contribution `-0.008999`
- `lag_11__CT5__flash_duration`: contribution `-0.008534`

Top utility-only movements:
- `lag_11__CT5__flash_duration`: contribution `-0.008534`
- `lag_00__CT5__flash_duration`: contribution `-0.007830`
- `lag_12__T_A_site_active_infernos`: contribution `-0.002349`

### tick `20636`, seconds `18.00`, LSTM delta `+0.0560`

Top all feature movements:
- `lag_14__CT_he_last_5s`: contribution `+0.013447`
- `lag_07__CT5__flash_duration`: contribution `+0.004993`
- `lag_00__T_shots_fired_sum`: contribution `+0.004849`
- `lag_11__CT_place_QUAD`: contribution `+0.004650`
- `lag_07__T_flashed_players`: contribution `+0.004069`

Top utility-only movements:
- `lag_14__CT_he_last_5s`: contribution `+0.013447`
- `lag_07__CT5__flash_duration`: contribution `+0.004993`
- `lag_07__CT_flash_duration_sum`: contribution `+0.001099`

### tick `20988`, seconds `23.50`, LSTM delta `-0.0481`

Top all feature movements:
- `lag_08__T_place_ARCH`: contribution `-0.014337`
- `lag_02__T_place_ARCH`: contribution `+0.005222`
- `lag_07__CT5__flash_duration`: contribution `-0.004993`
- `lag_07__CT_place_QUAD`: contribution `-0.003959`
- `lag_00__T_shots_fired_sum`: contribution `-0.003031`

Top utility-only movements:
- `lag_07__CT5__flash_duration`: contribution `-0.004993`
- `lag_07__CT_flash_duration_sum`: contribution `-0.001099`
- `lag_13__T_A_site_active_infernos`: contribution `-0.000977`
- `lag_11__T5__flash_duration`: contribution `+0.000696`

### tick `20092`, seconds `9.50`, LSTM delta `-0.0455`

Top all feature movements:
- `lag_07__CT_he_last_5s`: contribution `-0.006588`
- `lag_07__CT_place_RUINS`: contribution `-0.004975`
- `lag_09__T_place_LOWERMID`: contribution `-0.004404`
- `lag_04__CT_place_RUINS`: contribution `-0.002473`
- `lag_08__CT_place_RUINS`: contribution `-0.002187`

Top utility-only movements:
- `lag_07__CT_he_last_5s`: contribution `-0.006588`
- `lag_00__CT5__flash_duration`: contribution `+0.001367`
- `lag_00__T_A_site_active_infernos`: contribution `-0.000944`

### tick `20284`, seconds `12.50`, LSTM delta `+0.0424`

Top all feature movements:
- `lag_00__CT_place_QUAD`: contribution `+0.012352`
- `lag_03__CT_he_last_5s`: contribution `+0.010656`
- `lag_12__T_place_LOWERMID`: contribution `+0.002150`
- `lag_06__T_place_TRAMP`: contribution `+0.001878`
- `lag_06__T_A_site_active_infernos`: contribution `+0.001758`

Top utility-only movements:
- `lag_03__CT_he_last_5s`: contribution `+0.010656`
- `lag_06__T_A_site_active_infernos`: contribution `+0.001758`
- `lag_13__CT_he_last_5s`: contribution `-0.001279`
- `lag_06__T_active_infernos`: contribution `+0.000863`
