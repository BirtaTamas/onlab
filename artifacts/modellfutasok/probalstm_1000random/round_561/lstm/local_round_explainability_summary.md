# Local Round Explainability

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-spirit-vs-astralis-bo3-GZVTrKsE-zdG9dH6juITei/spirit-vs-astralis-m1-nuke.csv`
- round_num: `3`

## Largest probability jumps

- tick `17115`, seconds `54.50`, LSTM `0.0771`, delta `-0.1100`
- tick `14395`, seconds `12.00`, LSTM `0.1770`, delta `+0.0900`
- tick `15675`, seconds `32.00`, LSTM `0.0695`, delta `-0.0898`
- tick `16987`, seconds `52.50`, LSTM `0.1246`, delta `+0.0801`
- tick `14715`, seconds `17.00`, LSTM `0.0944`, delta `-0.0734`
- tick `14747`, seconds `17.50`, LSTM `0.1485`, delta `+0.0541`
- tick `13659`, seconds `0.50`, LSTM `0.0402`, delta `-0.0496`
- tick `15579`, seconds `30.50`, LSTM `0.1394`, delta `+0.0481`
- tick `14587`, seconds `15.00`, LSTM `0.1655`, delta `-0.0475`
- tick `14491`, seconds `13.50`, LSTM `0.1869`, delta `+0.0302`

## Top 15 local ridge features

- `lag_11__CT_place_SECRET`: coefficient `0.001508`, |coef| `0.001508`
- `lag_07__CT_place_SECRET`: coefficient `-0.001499`, |coef| `0.001499`
- `lag_00__T_place_TROPHY`: coefficient `0.001261`, |coef| `0.001261`
- `lag_00__CT_place_CRANE`: coefficient `-0.001224`, |coef| `0.001224`
- `lag_10__CT_place_HUTROOF`: coefficient `0.001042`, |coef| `0.001042`
- `lag_05__CT_place_SECRET`: coefficient `-0.000931`, |coef| `0.000931`
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.000841`, |coef| `0.000841`
- `lag_13__CT4__duck_amount`: coefficient `0.000835`, |coef| `0.000835`
- `lag_00__T_place_VENDING`: coefficient `-0.000832`, |coef| `0.000832`
- `lag_00__kill_diff_last_3s`: coefficient `0.000816`, |coef| `0.000816`
- `lag_00__damage_diff_last_5s`: coefficient `0.000804`, |coef| `0.000804`
- `lag_00__CT2__is_walking`: coefficient `-0.000786`, |coef| `0.000786`
- `lag_08__CT_place_SECRET`: coefficient `-0.000767`, |coef| `0.000767`
- `lag_14__CT_place_HUTROOF`: coefficient `-0.000751`, |coef| `0.000751`
- `lag_15__T2__duck_amount`: coefficient `0.000685`, |coef| `0.000685`

## Top 10 utility ridge features

- `lag_00__CT_utility_damage_last_5s`: coefficient `0.000841` (raises CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000564` (raises CT win probability)
- `lag_01__CT_flash_alpha_mean`: coefficient `0.000489` (raises CT win probability)
- `lag_13__CT4__smoke`: coefficient `-0.000326` (lowers CT win probability)
- `lag_12__T_A_site_active_infernos`: coefficient `0.000320` (raises CT win probability)
- `lag_00__CT_B_site_active_smokes`: coefficient `0.000319` (raises CT win probability)
- `lag_00__CT_A_site_active_smokes`: coefficient `0.000309` (raises CT win probability)
- `lag_12__T_B_site_active_infernos`: coefficient `0.000304` (raises CT win probability)
- `lag_09__T_A_site_active_infernos`: coefficient `-0.000295` (lowers CT win probability)
- `lag_05__CT_utility_damage_last_5s`: coefficient `0.000280` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_11__CT_place_SECRET`: coefficient `0.001508` (raises CT win probability)
- `lag_07__CT_place_SECRET`: coefficient `-0.001499` (lowers CT win probability)
- `lag_00__T_place_TROPHY`: coefficient `0.001261` (raises CT win probability)
- `lag_00__CT_place_CRANE`: coefficient `-0.001224` (lowers CT win probability)
- `lag_10__CT_place_HUTROOF`: coefficient `0.001042` (raises CT win probability)
- `lag_05__CT_place_SECRET`: coefficient `-0.000931` (lowers CT win probability)
- `lag_13__CT4__duck_amount`: coefficient `0.000835` (raises CT win probability)
- `lag_00__T_place_VENDING`: coefficient `-0.000832` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000816` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000804` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `17115`, seconds `54.50`, LSTM delta `-0.1100`

Top all feature movements:
- `lag_11__CT_place_SECRET`: contribution `-0.015523`
- `lag_14__CT_place_HUTROOF`: contribution `-0.005253`
- `lag_03__T_place_SQUEAKY`: contribution `-0.003781`
- `lag_15__T2__duck_amount`: contribution `-0.002621`
- `lag_13__CT4__duck_amount`: contribution `-0.002410`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `14395`, seconds `12.00`, LSTM delta `+0.0900`

Top all feature movements:
- `lag_00__T_place_TROPHY`: contribution `+0.008000`
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.007869`
- `lag_05__T_place_VENDING`: contribution `+0.005035`
- `lag_13__CT_place_HELL`: contribution `+0.004386`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.004329`

Top utility-only movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.007869`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.004329`

### tick `15675`, seconds `32.00`, LSTM delta `-0.0898`

Top all feature movements:
- `lag_00__CT_place_CRANE`: contribution `-0.020081`
- `lag_05__CT_place_SECRET`: contribution `-0.009579`
- `lag_02__CT_place_GARAGE`: contribution `-0.004789`
- `lag_03__CT_place_HUTROOF`: contribution `-0.003898`
- `lag_13__CT4__duck_amount`: contribution `-0.003068`

Top utility-only movements:
- `lag_12__T_A_site_active_infernos`: contribution `-0.000952`
- `lag_12__T_B_site_active_infernos`: contribution `-0.000858`

### tick `16987`, seconds `52.50`, LSTM delta `+0.0801`

Top all feature movements:
- `lag_07__CT_place_SECRET`: contribution `+0.015431`
- `lag_10__CT_place_HUTROOF`: contribution `+0.007293`
- `lag_00__kill_diff_last_3s`: contribution `+0.001964`
- `lag_00__CT2__is_walking`: contribution `+0.001856`
- `lag_10__T5__duck_amount`: contribution `+0.001678`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `14715`, seconds `17.00`, LSTM delta `-0.0734`

Top all feature movements:
- `lag_06__T_place_CONTROL`: contribution `-0.008671`
- `lag_00__T_place_TROPHY`: contribution `-0.008000`
- `lag_00__CT_utility_damage_last_5s`: contribution `-0.007869`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.004635`
- `lag_00__T_place_VENDING`: contribution `-0.004218`

Top utility-only movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `-0.007869`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.004635`
