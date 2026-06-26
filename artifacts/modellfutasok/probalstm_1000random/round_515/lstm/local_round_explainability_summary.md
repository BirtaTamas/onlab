# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-spirit-vs-heroic-bo3-8PNegF4uXnTykkGvqzloIi/spirit-vs-heroic-m2-nuke.csv`
- round_num: `1`

## Largest probability jumps

- tick `7247`, seconds `61.50`, LSTM `0.3264`, delta `-0.2377`
- tick `7119`, seconds `59.50`, LSTM `0.6070`, delta `+0.2299`
- tick `7087`, seconds `59.00`, LSTM `0.3770`, delta `+0.1728`
- tick `7375`, seconds `63.50`, LSTM `0.1145`, delta `-0.1466`
- tick `4655`, seconds `21.00`, LSTM `0.3818`, delta `-0.0942`
- tick `7311`, seconds `62.50`, LSTM `0.2434`, delta `-0.0839`
- tick `7407`, seconds `64.00`, LSTM `0.0352`, delta `-0.0794`
- tick `7055`, seconds `58.50`, LSTM `0.2043`, delta `+0.0770`
- tick `6991`, seconds `57.50`, LSTM `0.1256`, delta `+0.0636`
- tick `6927`, seconds `56.50`, LSTM `0.0842`, delta `-0.0633`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004079`, |coef| `0.004079`
- `lag_00__T_kills_last_3s`: coefficient `-0.002709`, |coef| `0.002709`
- `lag_00__CT_kills_last_3s`: coefficient `0.002424`, |coef| `0.002424`
- `lag_12__CT_place_VENTS`: coefficient `0.002271`, |coef| `0.002271`
- `lag_11__CT_place_VENTS`: coefficient `0.002221`, |coef| `0.002221`
- `lag_00__damage_diff_last_5s`: coefficient `0.001980`, |coef| `0.001980`
- `lag_00__T_place_CONTROL`: coefficient `-0.001800`, |coef| `0.001800`
- `lag_13__CT_place_DECON`: coefficient `0.001609`, |coef| `0.001609`
- `lag_02__CT_kills_last_3s`: coefficient `0.001587`, |coef| `0.001587`
- `lag_02__damage_diff_last_5s`: coefficient `0.001578`, |coef| `0.001578`
- `lag_01__kill_diff_last_3s`: coefficient `0.001560`, |coef| `0.001560`
- `lag_01__damage_diff_last_5s`: coefficient `0.001552`, |coef| `0.001552`
- `lag_05__CT_place_DECON`: coefficient `-0.001544`, |coef| `0.001544`
- `lag_02__kill_diff_last_3s`: coefficient `0.001544`, |coef| `0.001544`
- `lag_10__T_kills_last_3s`: coefficient `-0.001510`, |coef| `0.001510`

## Top 10 utility ridge features

- `lag_15__CT_flash_alpha_mean`: coefficient `0.000636` (raises CT win probability)
- `lag_13__CT_flash_alpha_mean`: coefficient `0.000624` (raises CT win probability)
- `lag_08__CT_flash_alpha_mean`: coefficient `0.000608` (raises CT win probability)
- `lag_11__CT_flash_alpha_mean`: coefficient `0.000585` (raises CT win probability)
- `lag_15__T_B_site_active_smokes`: coefficient `0.000576` (raises CT win probability)
- `lag_10__CT_flash_alpha_mean`: coefficient `0.000558` (raises CT win probability)
- `lag_12__CT_flash_alpha_mean`: coefficient `0.000551` (raises CT win probability)
- `lag_15__T_A_site_active_smokes`: coefficient `0.000544` (raises CT win probability)
- `lag_06__CT_flash_alpha_mean`: coefficient `0.000533` (raises CT win probability)
- `lag_07__CT_flash_alpha_mean`: coefficient `0.000530` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004079` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002709` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002424` (raises CT win probability)
- `lag_12__CT_place_VENTS`: coefficient `0.002271` (raises CT win probability)
- `lag_11__CT_place_VENTS`: coefficient `0.002221` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001980` (raises CT win probability)
- `lag_00__T_place_CONTROL`: coefficient `-0.001800` (lowers CT win probability)
- `lag_13__CT_place_DECON`: coefficient `0.001609` (raises CT win probability)
- `lag_02__CT_kills_last_3s`: coefficient `0.001587` (raises CT win probability)
- `lag_02__damage_diff_last_5s`: coefficient `0.001578` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `7247`, seconds `61.50`, LSTM delta `-0.2377`

Top all feature movements:
- `lag_02__CT_place_DECON`: contribution `-0.017256`
- `lag_14__CT_place_DECON`: contribution `-0.016398`
- `lag_00__kill_diff_last_3s`: contribution `-0.009817`
- `lag_00__T_kills_last_3s`: contribution `-0.008582`
- `lag_02__CT_place_VENTS`: contribution `-0.006392`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `7119`, seconds `59.50`, LSTM delta `+0.2299`

Top all feature movements:
- `lag_05__CT_place_DECON`: contribution `+0.024556`
- `lag_00__kill_diff_last_3s`: contribution `+0.019634`
- `lag_12__CT_place_VENTS`: contribution `+0.019059`
- `lag_10__CT_place_DECON`: contribution `+0.016842`
- `lag_00__T_kills_last_3s`: contribution `+0.008582`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `7087`, seconds `59.00`, LSTM delta `+0.1728`

Top all feature movements:
- `lag_11__CT_place_VENTS`: contribution `+0.018634`
- `lag_04__CT_place_DECON`: contribution `+0.011083`
- `lag_00__kill_diff_last_3s`: contribution `+0.009817`
- `lag_07__T_place_CONTROL`: contribution `+0.009195`
- `lag_00__CT_kills_last_3s`: contribution `+0.006998`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `7375`, seconds `63.50`, LSTM delta `-0.1466`

Top all feature movements:
- `lag_13__CT_place_DECON`: contribution `-0.025589`
- `lag_00__kill_diff_last_3s`: contribution `-0.009817`
- `lag_00__T_kills_last_3s`: contribution `-0.008582`
- `lag_00__damage_diff_last_5s`: contribution `-0.008486`
- `lag_06__CT_place_VENTS`: contribution `-0.005305`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `4655`, seconds `21.00`, LSTM delta `-0.0942`

Top all feature movements:
- `lag_00__T_place_CONTROL`: contribution `-0.025583`
- `lag_00__T_place_TROPHY`: contribution `-0.016126`
- `lag_02__T_place_CONTROL`: contribution `-0.009505`
- `lag_02__T_place_VENDING`: contribution `-0.006186`
- `lag_13__CT_place_HUTROOF`: contribution `-0.004538`

Top utility-only movements:
- No utility movement among the top local contributors.
