# Local Round Explainability

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-faze-vs-bcgame-bo3-daIGc_M_y7Qq42fF9AoQhi/faze-vs-bcgame-m2-anubis.csv`
- round_num: `9`

## Largest probability jumps

- tick `57671`, seconds `49.00`, LSTM `0.7705`, delta `+0.0932`
- tick `58023`, seconds `54.50`, LSTM `0.8621`, delta `+0.0858`
- tick `58183`, seconds `57.00`, LSTM `0.9421`, delta `+0.0492`
- tick `55303`, seconds `12.00`, LSTM `0.7319`, delta `+0.0463`
- tick `57831`, seconds `51.50`, LSTM `0.7561`, delta `-0.0379`
- tick `57383`, seconds `44.50`, LSTM `0.6908`, delta `+0.0370`
- tick `54823`, seconds `4.50`, LSTM `0.7042`, delta `+0.0367`
- tick `54663`, seconds `2.00`, LSTM `0.6614`, delta `-0.0327`
- tick `55015`, seconds `7.50`, LSTM `0.6650`, delta `-0.0325`
- tick `55335`, seconds `12.50`, LSTM `0.7002`, delta `-0.0316`

## Top 15 local ridge features

- `lag_00__CT_place_BRICKS`: coefficient `0.001053`, |coef| `0.001053`
- `lag_03__CT_place_TUNNELSTAIRS`: coefficient `0.000845`, |coef| `0.000845`
- `lag_07__CT_place_OUTSIDELONG`: coefficient `-0.000766`, |coef| `0.000766`
- `lag_09__T_place_MIDDOORS`: coefficient `-0.000737`, |coef| `0.000737`
- `lag_01__CT_place_TUNNEL`: coefficient `0.000725`, |coef| `0.000725`
- `lag_04__CT_place_BRIDGE`: coefficient `0.000722`, |coef| `0.000722`
- `lag_03__CT_place_BRIDGE`: coefficient `0.000686`, |coef| `0.000686`
- `lag_00__CT_kills_last_3s`: coefficient `0.000631`, |coef| `0.000631`
- `lag_04__CT_place_CTSIDEUPPER`: coefficient `-0.000631`, |coef| `0.000631`
- `lag_00__CT_place_BRIDGE`: coefficient `0.000593`, |coef| `0.000593`
- `lag_02__CT_place_BRIDGE`: coefficient `0.000566`, |coef| `0.000566`
- `lag_13__CT_place_MAIN`: coefficient `-0.000558`, |coef| `0.000558`
- `lag_00__kill_diff_last_3s`: coefficient `0.000526`, |coef| `0.000526`
- `lag_01__CT_place_BRIDGE`: coefficient `0.000515`, |coef| `0.000515`
- `lag_03__CT_place_BRICKS`: coefficient `-0.000510`, |coef| `0.000510`

## Top 10 utility ridge features

- `lag_14__CT5__molly`: coefficient `-0.000269` (lowers CT win probability)
- `lag_09__T5__flash_duration`: coefficient `0.000234` (raises CT win probability)
- `lag_01__CT3__molly`: coefficient `-0.000229` (lowers CT win probability)
- `lag_03__T_flash_duration_sum`: coefficient `0.000221` (raises CT win probability)
- `lag_07__T4__smoke`: coefficient `-0.000220` (lowers CT win probability)
- `lag_03__T2__flash_duration`: coefficient `0.000212` (raises CT win probability)
- `lag_08__CT_active_infernos`: coefficient `0.000209` (raises CT win probability)
- `lag_00__CT_A_site_active_infernos`: coefficient `-0.000201` (lowers CT win probability)
- `lag_02__CT_molly_inv`: coefficient `-0.000193` (lowers CT win probability)
- `lag_00__T3__flash_duration`: coefficient `0.000192` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_BRICKS`: coefficient `0.001053` (raises CT win probability)
- `lag_03__CT_place_TUNNELSTAIRS`: coefficient `0.000845` (raises CT win probability)
- `lag_07__CT_place_OUTSIDELONG`: coefficient `-0.000766` (lowers CT win probability)
- `lag_09__T_place_MIDDOORS`: coefficient `-0.000737` (lowers CT win probability)
- `lag_01__CT_place_TUNNEL`: coefficient `0.000725` (raises CT win probability)
- `lag_04__CT_place_BRIDGE`: coefficient `0.000722` (raises CT win probability)
- `lag_03__CT_place_BRIDGE`: coefficient `0.000686` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000631` (raises CT win probability)
- `lag_04__CT_place_CTSIDEUPPER`: coefficient `-0.000631` (lowers CT win probability)
- `lag_00__CT_place_BRIDGE`: coefficient `0.000593` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `57671`, seconds `49.00`, LSTM delta `+0.0932`

Top all feature movements:
- `lag_03__CT_place_BRICKS`: contribution `+0.009788`
- `lag_07__CT_place_OUTSIDELONG`: contribution `+0.007765`
- `lag_09__CT_place_BRICKS`: contribution `+0.006153`
- `lag_05__CT_place_BRICKS`: contribution `+0.006033`
- `lag_13__CT_place_MAIN`: contribution `+0.003759`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `58023`, seconds `54.50`, LSTM delta `+0.0858`

Top all feature movements:
- `lag_03__CT_place_TUNNELSTAIRS`: contribution `+0.011902`
- `lag_01__CT_place_TUNNEL`: contribution `+0.011651`
- `lag_01__CT_place_TUNNELSTAIRS`: contribution `+0.005220`
- `lag_14__CT_place_BRICKS`: contribution `+0.004847`
- `lag_00__CT_kills_last_3s`: contribution `+0.001823`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `58183`, seconds `57.00`, LSTM delta `+0.0492`

Top all feature movements:
- `lag_04__CT_place_BRIDGE`: contribution `+0.008275`
- `lag_06__CT_place_TUNNEL`: contribution `+0.006514`
- `lag_08__CT_place_TUNNELSTAIRS`: contribution `+0.004741`
- `lag_02__CT_place_TUNNEL`: contribution `+0.002895`
- `lag_06__CT_place_TUNNELSTAIRS`: contribution `+0.002708`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `55303`, seconds `12.00`, LSTM delta `+0.0463`

Top all feature movements:
- `lag_00__CT_place_BRICKS`: contribution `+0.020214`
- `lag_09__CT_place_BRICKS`: contribution `-0.006153`
- `lag_11__CT_place_BRICKS`: contribution `+0.005404`
- `lag_15__CT_place_LOWERTUNNEL`: contribution `+0.003321`
- `lag_07__CT_place_BACKOFB`: contribution `+0.001674`

Top utility-only movements:
- `lag_08__CT_B_site_active_infernos`: contribution `+0.000592`
- `lag_08__CT_active_infernos`: contribution `+0.000481`
- `lag_10__CT1__molly`: contribution `+0.000292`

### tick `57831`, seconds `51.50`, LSTM delta `-0.0379`

Top all feature movements:
- `lag_14__CT_place_BRICKS`: contribution `-0.004847`
- `lag_08__CT_place_BRICKS`: contribution `-0.003331`
- `lag_12__CT_place_OUTSIDELONG`: contribution `-0.002899`
- `lag_10__CT_place_BRICKS`: contribution `+0.002121`
- `lag_00__CT_place_HEAVEN`: contribution `-0.001867`

Top utility-only movements:
- No utility movement among the top local contributors.
