# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-3dmax-vs-betboom-anubis-9yOMu3EhAmKzkIxUzvijXH/3dmax-vs-betboom-anubis.csv`
- round_num: `13`

## Largest probability jumps

- tick `118822`, seconds `32.50`, LSTM `0.4010`, delta `+0.2867`
- tick `118342`, seconds `25.00`, LSTM `0.2517`, delta `-0.2657`
- tick `118886`, seconds `33.50`, LSTM `0.6737`, delta `+0.2001`
- tick `119270`, seconds `39.50`, LSTM `0.8582`, delta `+0.1466`
- tick `121286`, seconds `71.00`, LSTM `0.7275`, delta `+0.1429`
- tick `120582`, seconds `60.00`, LSTM `0.5987`, delta `-0.1365`
- tick `118374`, seconds `25.50`, LSTM `0.1393`, delta `-0.1125`
- tick `121030`, seconds `67.00`, LSTM `0.4429`, delta `+0.1089`
- tick `121190`, seconds `69.50`, LSTM `0.5363`, delta `+0.1089`
- tick `120550`, seconds `59.50`, LSTM `0.7352`, delta `-0.0850`

## Top 15 local ridge features

- `lag_04__T_place_CONNECTOR`: coefficient `0.003677`, |coef| `0.003677`
- `lag_14__CT_place_FOUNTAIN`: coefficient `-0.003349`, |coef| `0.003349`
- `lag_12__T_place_CONNECTOR`: coefficient `-0.003220`, |coef| `0.003220`
- `lag_01__T_place_CONNECTOR`: coefficient `0.002958`, |coef| `0.002958`
- `lag_00__kill_diff_last_3s`: coefficient `0.002920`, |coef| `0.002920`
- `lag_11__T_place_BRIDGE`: coefficient `-0.002804`, |coef| `0.002804`
- `lag_00__damage_diff_last_5s`: coefficient `0.002754`, |coef| `0.002754`
- `lag_00__T_place_MIDDOORS`: coefficient `-0.002677`, |coef| `0.002677`
- `lag_00__CT_place_WALKWAY`: coefficient `-0.002541`, |coef| `0.002541`
- `lag_00__CT_place_FOUNTAIN`: coefficient `0.002526`, |coef| `0.002526`
- `lag_14__T_place_CONNECTOR`: coefficient `-0.002517`, |coef| `0.002517`
- `lag_12__T4__is_walking`: coefficient `-0.002261`, |coef| `0.002261`
- `lag_00__T_kills_last_3s`: coefficient `-0.002166`, |coef| `0.002166`
- `lag_07__T2__is_walking`: coefficient `-0.002148`, |coef| `0.002148`
- `lag_03__T_place_CONNECTOR`: coefficient `0.002088`, |coef| `0.002088`

## Top 10 utility ridge features

- `lag_14__T5__flash_duration`: coefficient `0.001001` (raises CT win probability)
- `lag_00__T2__smoke`: coefficient `-0.000843` (lowers CT win probability)
- `lag_02__T2__smoke`: coefficient `-0.000679` (lowers CT win probability)
- `lag_12__CT_active_smokes`: coefficient `-0.000672` (lowers CT win probability)
- `lag_13__CT_active_smokes`: coefficient `-0.000583` (lowers CT win probability)
- `lag_11__CT_active_smokes`: coefficient `-0.000577` (lowers CT win probability)
- `lag_00__T_B_site_active_smokes`: coefficient `-0.000512` (lowers CT win probability)
- `lag_00__CT_B_site_active_smokes`: coefficient `-0.000491` (lowers CT win probability)
- `lag_13__active_smokes_total`: coefficient `-0.000482` (lowers CT win probability)
- `lag_15__T4__flash_duration`: coefficient `0.000481` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_04__T_place_CONNECTOR`: coefficient `0.003677` (raises CT win probability)
- `lag_14__CT_place_FOUNTAIN`: coefficient `-0.003349` (lowers CT win probability)
- `lag_12__T_place_CONNECTOR`: coefficient `-0.003220` (lowers CT win probability)
- `lag_01__T_place_CONNECTOR`: coefficient `0.002958` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002920` (raises CT win probability)
- `lag_11__T_place_BRIDGE`: coefficient `-0.002804` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002754` (raises CT win probability)
- `lag_00__T_place_MIDDOORS`: coefficient `-0.002677` (lowers CT win probability)
- `lag_00__CT_place_WALKWAY`: coefficient `-0.002541` (lowers CT win probability)
- `lag_00__CT_place_FOUNTAIN`: coefficient `0.002526` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `118822`, seconds `32.50`, LSTM delta `+0.2867`

Top all feature movements:
- `lag_14__CT_place_FOUNTAIN`: contribution `+0.035228`
- `lag_00__T_place_MIDDOORS`: contribution `+0.022753`
- `lag_01__T_place_CONNECTOR`: contribution `+0.014326`
- `lag_11__T_place_BRIDGE`: contribution `+0.012142`
- `lag_02__T_place_CONNECTOR`: contribution `+0.009286`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `118342`, seconds `25.00`, LSTM delta `-0.2657`

Top all feature movements:
- `lag_04__T_place_CONNECTOR`: contribution `-0.017805`
- `lag_12__T_place_CONNECTOR`: contribution `-0.015591`
- `lag_01__T_place_CONNECTOR`: contribution `-0.014326`
- `lag_14__T_place_CONNECTOR`: contribution `-0.012189`
- `lag_00__kill_diff_last_3s`: contribution `-0.007029`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `118886`, seconds `33.50`, LSTM delta `+0.2001`

Top all feature movements:
- `lag_04__T_place_CONNECTOR`: contribution `+0.017805`
- `lag_02__T_place_MIDDOORS`: contribution `+0.017482`
- `lag_11__T_place_BRIDGE`: contribution `+0.012142`
- `lag_03__T_place_CONNECTOR`: contribution `+0.010109`
- `lag_11__T_place_MIDDOORS`: contribution `+0.008369`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `119270`, seconds `39.50`, LSTM delta `+0.1466`

Top all feature movements:
- `lag_12__T_place_CONNECTOR`: contribution `+0.015591`
- `lag_01__T_place_MAIN`: contribution `+0.011214`
- `lag_14__T_place_MIDDOORS`: contribution `+0.010690`
- `lag_05__T_place_CONNECTOR`: contribution `+0.007822`
- `lag_00__kill_diff_last_3s`: contribution `+0.007029`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `121286`, seconds `71.00`, LSTM delta `+0.1429`

Top all feature movements:
- `lag_00__CT_place_STREET`: contribution `+0.029631`
- `lag_00__CT_place_TSTAIRS`: contribution `+0.011354`
- `lag_11__CT_place_HEAVEN`: contribution `+0.008287`
- `lag_08__CT_place_WALKWAY`: contribution `+0.007585`
- `lag_04__T_place_MAIN`: contribution `+0.007314`

Top utility-only movements:
- No utility movement among the top local contributors.
