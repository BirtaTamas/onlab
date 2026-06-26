# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-eternal-fire-vs-spirit-bo5-7H36TpK_LYGHtCXpF3Cgdr/eternal-fire-vs-spirit-m3-dust2.csv`
- round_num: `6`

## Largest probability jumps

- tick `43819`, seconds `95.50`, LSTM `0.7747`, delta `+0.2708`
- tick `42123`, seconds `69.00`, LSTM `0.7544`, delta `-0.1740`
- tick `41739`, seconds `63.00`, LSTM `0.8132`, delta `+0.1578`
- tick `41931`, seconds `66.00`, LSTM `0.9125`, delta `+0.1386`
- tick `42059`, seconds `68.00`, LSTM `0.7967`, delta `-0.1322`
- tick `42091`, seconds `68.50`, LSTM `0.9284`, delta `+0.1317`
- tick `44075`, seconds `99.50`, LSTM `0.9371`, delta `+0.1293`
- tick `44043`, seconds `99.00`, LSTM `0.8078`, delta `-0.0892`
- tick `42187`, seconds `70.00`, LSTM `0.7113`, delta `-0.0591`
- tick `42251`, seconds `71.00`, LSTM `0.6815`, delta `-0.0515`

## Top 15 local ridge features

- `lag_12__CT3__duck_amount`: coefficient `0.002937`, |coef| `0.002937`
- `lag_00__CT_kills_last_3s`: coefficient `0.002914`, |coef| `0.002914`
- `lag_00__kill_diff_last_3s`: coefficient `0.002739`, |coef| `0.002739`
- `lag_12__T5__duck_amount`: coefficient `-0.002687`, |coef| `0.002687`
- `lag_01__CT_B_site_active_infernos`: coefficient `0.002560`, |coef| `0.002560`
- `lag_05__CT3__duck_amount`: coefficient `-0.002493`, |coef| `0.002493`
- `lag_00__CT_damage_last_5s`: coefficient `0.002397`, |coef| `0.002397`
- `lag_00__damage_diff_last_5s`: coefficient `0.002365`, |coef| `0.002365`
- `lag_00__T5__alive`: coefficient `-0.002190`, |coef| `0.002190`
- `lag_00__T5__hp`: coefficient `-0.002151`, |coef| `0.002151`
- `lag_15__CT_place_TUNNELSTAIRS`: coefficient `-0.002084`, |coef| `0.002084`
- `lag_05__T4__is_walking`: coefficient `-0.002067`, |coef| `0.002067`
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.002054`, |coef| `0.002054`
- `lag_00__T_macro_B`: coefficient `-0.002054`, |coef| `0.002054`
- `lag_05__CT1__molly`: coefficient `-0.002049`, |coef| `0.002049`

## Top 10 utility ridge features

- `lag_01__CT_B_site_active_infernos`: coefficient `0.002560` (raises CT win probability)
- `lag_05__CT1__molly`: coefficient `-0.002049` (lowers CT win probability)
- `lag_01__CT_active_infernos`: coefficient `0.001597` (raises CT win probability)
- `lag_00__CT_B_site_active_infernos`: coefficient `0.001322` (raises CT win probability)
- `lag_06__CT1__molly`: coefficient `-0.001248` (lowers CT win probability)
- `lag_02__CT_B_site_active_infernos`: coefficient `0.001178` (raises CT win probability)
- `lag_10__T4__flash_duration`: coefficient `0.001093` (raises CT win probability)
- `lag_10__T_flash_duration_sum`: coefficient `0.000951` (raises CT win probability)
- `lag_10__T3__flash_duration`: coefficient `0.000936` (raises CT win probability)
- `lag_01__active_infernos_total`: coefficient `0.000892` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_12__CT3__duck_amount`: coefficient `0.002937` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002914` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002739` (raises CT win probability)
- `lag_12__T5__duck_amount`: coefficient `-0.002687` (lowers CT win probability)
- `lag_05__CT3__duck_amount`: coefficient `-0.002493` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002397` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002365` (raises CT win probability)
- `lag_00__T5__alive`: coefficient `-0.002190` (lowers CT win probability)
- `lag_00__T5__hp`: coefficient `-0.002151` (lowers CT win probability)
- `lag_15__CT_place_TUNNELSTAIRS`: coefficient `-0.002084` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `43819`, seconds `95.50`, LSTM delta `+0.2708`

Top all feature movements:
- `lag_12__CT3__duck_amount`: contribution `+0.010331`
- `lag_12__T5__duck_amount`: contribution `+0.009570`
- `lag_05__CT3__duck_amount`: contribution `+0.009067`
- `lag_01__CT_B_site_active_infernos`: contribution `+0.008794`
- `lag_00__CT_kills_last_3s`: contribution `+0.008412`

Top utility-only movements:
- `lag_01__CT_B_site_active_infernos`: contribution `+0.008794`
- `lag_05__CT1__molly`: contribution `+0.005099`
- `lag_01__CT_active_infernos`: contribution `+0.003679`

### tick `42123`, seconds `69.00`, LSTM delta `-0.1740`

Top all feature movements:
- `lag_00__T_place_HOLE`: contribution `+0.036134`
- `lag_00__T_place_BDOORS`: contribution `-0.022190`
- `lag_03__T_place_BDOORS`: contribution `-0.017552`
- `lag_12__T_place_BDOORS`: contribution `-0.014303`
- `lag_00__kill_diff_last_3s`: contribution `-0.013188`

Top utility-only movements:
- `lag_10__T4__flash_duration`: contribution `-0.005801`
- `lag_09__T1__flash_duration`: contribution `-0.005347`
- `lag_01__T2__flash_duration`: contribution `-0.004385`
- `lag_08__T2__flash_duration`: contribution `-0.003411`

### tick `41739`, seconds `63.00`, LSTM delta `+0.1578`

Top all feature movements:
- `lag_00__T_place_BDOORS`: contribution `+0.022190`
- `lag_00__CT_kills_last_3s`: contribution `+0.008412`
- `lag_10__T_flash_duration_sum`: contribution `+0.007873`
- `lag_10__T4__flash_duration`: contribution `+0.007275`
- `lag_00__kill_diff_last_3s`: contribution `+0.006594`

Top utility-only movements:
- `lag_10__T_flash_duration_sum`: contribution `+0.007873`
- `lag_10__T4__flash_duration`: contribution `+0.007275`
- `lag_10__T3__flash_duration`: contribution `+0.006220`
- `lag_10__T1__flash_duration`: contribution `+0.003662`
- `lag_00__T3__flash_duration`: contribution `+0.002478`

### tick `41931`, seconds `66.00`, LSTM delta `+0.1386`

Top all feature movements:
- `lag_06__T_place_BDOORS`: contribution `+0.017750`
- `lag_02__CT_place_HOLE`: contribution `+0.009961`
- `lag_07__CT4__is_scoped`: contribution `-0.006541`
- `lag_06__T_place_MIDDOORS`: contribution `+0.005892`
- `lag_02__T2__flash_duration`: contribution `+0.005862`

Top utility-only movements:
- `lag_02__T2__flash_duration`: contribution `+0.005862`
- `lag_04__T4__flash_duration`: contribution `+0.003800`
- `lag_06__T3__flash_duration`: contribution `+0.003086`
- `lag_03__T1__flash_duration`: contribution `+0.002861`

### tick `42059`, seconds `68.00`, LSTM delta `-0.1322`

Top all feature movements:
- `lag_10__T_place_BDOORS`: contribution `-0.012841`
- `lag_00__damage_diff_last_5s`: contribution `-0.008003`
- `lag_02__T_place_BDOORS`: contribution `-0.007504`
- `lag_06__CT_place_HOLE`: contribution `-0.006979`
- `lag_00__kill_diff_last_3s`: contribution `-0.006594`

Top utility-only movements:
- `lag_10__T3__flash_duration`: contribution `-0.004989`
- `lag_07__T1__flash_duration`: contribution `-0.004116`
- `lag_06__T2__flash_duration`: contribution `-0.003841`
- `lag_10__T_flash_duration_sum`: contribution `-0.002113`
