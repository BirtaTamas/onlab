# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-spirit-vs-heroic-bo3-NSK1XoxyhXK-sIe0204dzp/spirit-vs-heroic-m2-nuke.csv`
- round_num: `1`

## Largest probability jumps

- tick `6072`, seconds `75.50`, LSTM `0.8599`, delta `+0.3168`
- tick `7864`, seconds `103.50`, LSTM `0.6300`, delta `-0.1944`
- tick `4728`, seconds `54.50`, LSTM `0.3052`, delta `-0.1893`
- tick `5816`, seconds `71.50`, LSTM `0.5517`, delta `+0.1428`
- tick `5048`, seconds `59.50`, LSTM `0.4040`, delta `+0.0965`
- tick `5784`, seconds `71.00`, LSTM `0.4089`, delta `+0.0939`
- tick `5624`, seconds `68.50`, LSTM `0.3841`, delta `-0.0935`
- tick `4760`, seconds `55.00`, LSTM `0.2217`, delta `-0.0835`
- tick `5080`, seconds `60.00`, LSTM `0.4781`, delta `+0.0742`
- tick `6296`, seconds `79.00`, LSTM `0.8869`, delta `-0.0665`

## Top 15 local ridge features

- `lag_03__T_place_ADMIN`: coefficient `-0.005392`, |coef| `0.005392`
- `lag_08__T_place_RAFTERS`: coefficient `-0.003660`, |coef| `0.003660`
- `lag_00__kill_diff_last_3s`: coefficient `0.002990`, |coef| `0.002990`
- `lag_02__CT_place_VENDING`: coefficient `0.002362`, |coef| `0.002362`
- `lag_00__damage_diff_last_5s`: coefficient `0.002282`, |coef| `0.002282`
- `lag_00__T_place_HEAVEN`: coefficient `-0.002257`, |coef| `0.002257`
- `lag_14__T_place_SECRET`: coefficient `-0.002234`, |coef| `0.002234`
- `lag_00__CT_kills_last_3s`: coefficient `0.002230`, |coef| `0.002230`
- `lag_06__T_place_SECRET`: coefficient `-0.001876`, |coef| `0.001876`
- `lag_09__T_place_RAFTERS`: coefficient `-0.001731`, |coef| `0.001731`
- `lag_04__CT_place_VENDING`: coefficient `0.001601`, |coef| `0.001601`
- `lag_05__CT_place_VENDING`: coefficient `0.001589`, |coef| `0.001589`
- `lag_02__CT_place_TROPHY`: coefficient `-0.001507`, |coef| `0.001507`
- `lag_00__CT_place_GARAGE`: coefficient `-0.001494`, |coef| `0.001494`
- `lag_00__T_kills_last_3s`: coefficient `-0.001488`, |coef| `0.001488`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `0.000329` (raises CT win probability)
- `lag_02__CT_B_site_active_smokes`: coefficient `0.000201` (raises CT win probability)
- `lag_02__CT_A_site_active_smokes`: coefficient `0.000198` (raises CT win probability)
- `lag_07__CT5__smoke`: coefficient `-0.000177` (lowers CT win probability)
- `lag_07__CT_flash_alpha_mean`: coefficient `0.000177` (raises CT win probability)
- `lag_09__CT_B_site_active_smokes`: coefficient `-0.000152` (lowers CT win probability)
- `lag_09__CT_A_site_active_smokes`: coefficient `-0.000144` (lowers CT win probability)
- `lag_05__CT_flash_alpha_mean`: coefficient `0.000135` (raises CT win probability)
- `lag_06__T_flash_alpha_mean`: coefficient `0.000131` (raises CT win probability)
- `lag_08__CT5__smoke`: coefficient `-0.000128` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_03__T_place_ADMIN`: coefficient `-0.005392` (lowers CT win probability)
- `lag_08__T_place_RAFTERS`: coefficient `-0.003660` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002990` (raises CT win probability)
- `lag_02__CT_place_VENDING`: coefficient `0.002362` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002282` (raises CT win probability)
- `lag_00__T_place_HEAVEN`: coefficient `-0.002257` (lowers CT win probability)
- `lag_14__T_place_SECRET`: coefficient `-0.002234` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002230` (raises CT win probability)
- `lag_06__T_place_SECRET`: coefficient `-0.001876` (lowers CT win probability)
- `lag_09__T_place_RAFTERS`: coefficient `-0.001731` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `6072`, seconds `75.50`, LSTM delta `+0.3168`

Top all feature movements:
- `lag_02__CT_place_VENDING`: contribution `+0.040484`
- `lag_00__T_place_HEAVEN`: contribution `+0.027699`
- `lag_14__T_place_SECRET`: contribution `+0.023505`
- `lag_02__CT_place_TROPHY`: contribution `+0.022265`
- `lag_06__CT_place_TROPHY`: contribution `+0.021465`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `7864`, seconds `103.50`, LSTM delta `-0.1944`

Top all feature movements:
- `lag_03__T_place_ADMIN`: contribution `-0.104819`
- `lag_13__T_place_CONTROL`: contribution `-0.009823`
- `lag_09__T_place_CONTROL`: contribution `-0.009343`
- `lag_00__kill_diff_last_3s`: contribution `-0.007196`
- `lag_00__damage_diff_last_5s`: contribution `-0.005148`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `4728`, seconds `54.50`, LSTM delta `-0.1893`

Top all feature movements:
- `lag_08__T_place_RAFTERS`: contribution `-0.095797`
- `lag_13__T_place_HEAVEN`: contribution `-0.017198`
- `lag_08__T_place_HEAVEN`: contribution `-0.012009`
- `lag_00__kill_diff_last_3s`: contribution `-0.007196`
- `lag_00__damage_diff_last_5s`: contribution `-0.005148`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `5816`, seconds `71.50`, LSTM delta `+0.1428`

Top all feature movements:
- `lag_06__T_place_SECRET`: contribution `+0.019745`
- `lag_09__T_place_HUT`: contribution `+0.008101`
- `lag_00__kill_diff_last_3s`: contribution `+0.007196`
- `lag_00__CT_kills_last_3s`: contribution `+0.006439`
- `lag_10__CT1__duck_amount`: contribution `+0.005356`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `5048`, seconds `59.50`, LSTM delta `+0.0965`

Top all feature movements:
- `lag_08__T_place_RAFTERS`: contribution `+0.095797`
- `lag_08__T_place_HEAVEN`: contribution `+0.012009`
- `lag_00__damage_diff_last_5s`: contribution `+0.005148`
- `lag_04__CT_place_LOCKERROOM`: contribution `-0.003473`
- `lag_09__T_place_TROPHY`: contribution `-0.002754`

Top utility-only movements:
- No utility movement among the top local contributors.
