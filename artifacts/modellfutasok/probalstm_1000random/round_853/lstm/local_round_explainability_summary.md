# Local Round Explainability

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-vitality-vs-g2-bo5-ieXHvClzA7f_aJ_85fPFqK/vitality-vs-g2-m1-dust2.csv`
- round_num: `10`

## Largest probability jumps

- tick `82983`, seconds `73.00`, LSTM `0.3982`, delta `-0.3862`
- tick `85095`, seconds `106.00`, LSTM `0.7758`, delta `+0.3027`
- tick `84967`, seconds `104.00`, LSTM `0.2064`, delta `-0.2377`
- tick `85031`, seconds `105.00`, LSTM `0.4861`, delta `+0.1992`
- tick `83111`, seconds `75.00`, LSTM `0.5704`, delta `+0.1728`
- tick `82759`, seconds `69.50`, LSTM `0.8646`, delta `+0.1329`
- tick `84999`, seconds `104.50`, LSTM `0.2869`, delta `+0.0805`
- tick `85191`, seconds `107.50`, LSTM `0.9046`, delta `+0.0767`
- tick `83207`, seconds `76.50`, LSTM `0.6344`, delta `+0.0617`
- tick `83303`, seconds `78.00`, LSTM `0.5655`, delta `-0.0583`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.003448`, |coef| `0.003448`
- `lag_00__kill_diff_last_3s`: coefficient `0.003398`, |coef| `0.003398`
- `lag_04__T5__flash_duration`: coefficient `-0.002676`, |coef| `0.002676`
- `lag_00__damage_diff_last_5s`: coefficient `0.002639`, |coef| `0.002639`
- `lag_01__CT_place_ARAMP`: coefficient `0.002024`, |coef| `0.002024`
- `lag_02__CT_place_BDOORS`: coefficient `-0.001899`, |coef| `0.001899`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.001854`, |coef| `0.001854`
- `lag_03__T4__flash_duration`: coefficient `-0.001802`, |coef| `0.001802`
- `lag_04__CT1__flash_duration`: coefficient `-0.001783`, |coef| `0.001783`
- `lag_00__CT_defusing_count`: coefficient `0.001748`, |coef| `0.001748`
- `lag_01__T_place_BDOORS`: coefficient `0.001693`, |coef| `0.001693`
- `lag_02__T4__flash_duration`: coefficient `-0.001645`, |coef| `0.001645`
- `lag_01__CT_kills_last_3s`: coefficient `0.001609`, |coef| `0.001609`
- `lag_13__CT_duck_amount_mean`: coefficient `-0.001592`, |coef| `0.001592`
- `lag_03__T_place_BDOORS`: coefficient `0.001565`, |coef| `0.001565`

## Top 10 utility ridge features

- `lag_04__T5__flash_duration`: coefficient `-0.002676` (lowers CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.001854` (lowers CT win probability)
- `lag_03__T4__flash_duration`: coefficient `-0.001802` (lowers CT win probability)
- `lag_04__CT1__flash_duration`: coefficient `-0.001783` (lowers CT win probability)
- `lag_02__T4__flash_duration`: coefficient `-0.001645` (lowers CT win probability)
- `lag_07__T4__flash_duration`: coefficient `0.001277` (raises CT win probability)
- `lag_05__CT4__flash_duration`: coefficient `0.001256` (raises CT win probability)
- `lag_06__CT1__flash_duration`: coefficient `0.001216` (raises CT win probability)
- `lag_03__CT4__flash_duration`: coefficient `-0.001209` (lowers CT win probability)
- `lag_03__T_flash_alpha_mean`: coefficient `-0.001089` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.003448` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003398` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002639` (raises CT win probability)
- `lag_01__CT_place_ARAMP`: coefficient `0.002024` (raises CT win probability)
- `lag_02__CT_place_BDOORS`: coefficient `-0.001899` (lowers CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.001748` (raises CT win probability)
- `lag_01__T_place_BDOORS`: coefficient `0.001693` (raises CT win probability)
- `lag_01__CT_kills_last_3s`: coefficient `0.001609` (raises CT win probability)
- `lag_13__CT_duck_amount_mean`: coefficient `-0.001592` (lowers CT win probability)
- `lag_03__T_place_BDOORS`: coefficient `0.001565` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `82983`, seconds `73.00`, LSTM delta `-0.3862`

Top all feature movements:
- `lag_04__T5__flash_duration`: contribution `-0.020632`
- `lag_01__CT_place_ARAMP`: contribution `-0.012610`
- `lag_00__T_place_TUNNELSTAIRS`: contribution `-0.010140`
- `lag_00__CT_kills_last_3s`: contribution `-0.009954`
- `lag_04__CT1__flash_duration`: contribution `-0.008406`

Top utility-only movements:
- `lag_04__T5__flash_duration`: contribution `-0.020632`
- `lag_04__CT1__flash_duration`: contribution `-0.008406`
- `lag_06__CT1__flash_duration`: contribution `-0.005942`
- `lag_07__CT1__flash_duration`: contribution `-0.004817`

### tick `85095`, seconds `106.00`, LSTM delta `+0.3027`

Top all feature movements:
- `lag_03__T_place_HOLE`: contribution `+0.031787`
- `lag_08__T_place_HOLE`: contribution `+0.029557`
- `lag_03__T_place_BDOORS`: contribution `+0.019578`
- `lag_00__T_place_BDOORS`: contribution `+0.018624`
- `lag_04__CT_place_BDOORS`: contribution `+0.011873`

Top utility-only movements:
- `lag_02__T4__flash_duration`: contribution `+0.011460`
- `lag_00__T_flash_alpha_mean`: contribution `+0.011250`
- `lag_07__T4__flash_duration`: contribution `+0.008745`
- `lag_07__CT4__flash_duration`: contribution `+0.005277`
- `lag_02__T_flash_duration_sum`: contribution `+0.003071`

### tick `84967`, seconds `104.00`, LSTM delta `-0.2377`

Top all feature movements:
- `lag_04__T_place_HOLE`: contribution `-0.035969`
- `lag_00__CT_place_BDOORS`: contribution `-0.014248`
- `lag_03__T4__flash_duration`: contribution `-0.012338`
- `lag_00__kill_diff_last_3s`: contribution `-0.008180`
- `lag_03__CT4__flash_duration`: contribution `-0.006353`

Top utility-only movements:
- `lag_03__T4__flash_duration`: contribution `-0.012338`
- `lag_03__CT4__flash_duration`: contribution `-0.006353`
- `lag_03__T_flash_duration_sum`: contribution `-0.003614`

### tick `85031`, seconds `105.00`, LSTM delta `+0.1992`

Top all feature movements:
- `lag_01__T_place_BDOORS`: contribution `+0.021171`
- `lag_02__CT_place_BDOORS`: contribution `+0.018265`
- `lag_00__CT_kills_last_3s`: contribution `+0.009954`
- `lag_00__kill_diff_last_3s`: contribution `+0.008180`
- `lag_00__T4__flash_duration`: contribution `+0.007459`

Top utility-only movements:
- `lag_00__T4__flash_duration`: contribution `+0.007459`
- `lag_05__CT4__flash_duration`: contribution `+0.006601`
- `lag_05__T4__flash_duration`: contribution `+0.004227`

### tick `83111`, seconds `75.00`, LSTM delta `+0.1728`

Top all feature movements:
- `lag_03__CT_place_SIDE`: contribution `+0.022265`
- `lag_00__CT_kills_last_3s`: contribution `+0.009954`
- `lag_00__kill_diff_last_3s`: contribution `+0.008180`
- `lag_08__T5__flash_duration`: contribution `+0.007117`
- `lag_06__T_place_TUNNELSTAIRS`: contribution `+0.006902`

Top utility-only movements:
- `lag_08__T5__flash_duration`: contribution `+0.007117`
