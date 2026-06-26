# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-eternal-fire-vs-fluxo-bo3-Kqy3ohBVu1ANumI6Qdn26R/eternal-fire-vs-fluxo-m2-dust2.csv`
- round_num: `3`

## Largest probability jumps

- tick `19001`, seconds `0.50`, LSTM `0.9318`, delta `+0.0215`
- tick `20601`, seconds `25.50`, LSTM `0.9561`, delta `+0.0167`
- tick `19449`, seconds `7.50`, LSTM `0.9121`, delta `+0.0152`
- tick `20121`, seconds `18.00`, LSTM `0.9507`, delta `+0.0132`
- tick `19705`, seconds `11.50`, LSTM `0.9499`, delta `+0.0122`
- tick `21465`, seconds `39.00`, LSTM `0.9800`, delta `+0.0115`
- tick `19129`, seconds `2.50`, LSTM `0.9019`, delta `-0.0100`
- tick `19033`, seconds `1.00`, LSTM `0.9238`, delta `-0.0080`
- tick `20089`, seconds `17.50`, LSTM `0.9375`, delta `+0.0079`
- tick `19673`, seconds `11.00`, LSTM `0.9377`, delta `+0.0077`

## Top 15 local ridge features

- `lag_10__T_place_TUNNELSTAIRS`: coefficient `-0.000237`, |coef| `0.000237`
- `lag_00__T_place_OUTSIDETUNNEL`: coefficient `-0.000225`, |coef| `0.000225`
- `lag_01__CT_place_UNDERA`: coefficient `-0.000220`, |coef| `0.000220`
- `lag_01__CT_place_CTSPAWN`: coefficient `0.000195`, |coef| `0.000195`
- `lag_00__CT_place_EXTENDEDA`: coefficient `-0.000193`, |coef| `0.000193`
- `lag_01__CT_flash_alpha_mean`: coefficient `-0.000187`, |coef| `0.000187`
- `lag_00__CT5__duck_amount`: coefficient `0.000178`, |coef| `0.000178`
- `lag_00__CT_kills_last_3s`: coefficient `0.000167`, |coef| `0.000167`
- `lag_03__CT_utility_damage_last_5s`: coefficient `0.000161`, |coef| `0.000161`
- `lag_03__CT_place_MIDDOORS`: coefficient `0.000158`, |coef| `0.000158`
- `lag_00__T_place_OUTSIDELONG`: coefficient `-0.000154`, |coef| `0.000154`
- `lag_01__T_place_TSPAWN`: coefficient `0.000151`, |coef| `0.000151`
- `lag_01__T_velocity_mean`: coefficient `-0.000149`, |coef| `0.000149`
- `lag_01__T_place_OUTSIDETUNNEL`: coefficient `-0.000148`, |coef| `0.000148`
- `lag_08__CT_place_MIDDOORS`: coefficient `0.000148`, |coef| `0.000148`

## Top 10 utility ridge features

- `lag_01__CT_flash_alpha_mean`: coefficient `-0.000187` (lowers CT win probability)
- `lag_03__CT_utility_damage_last_5s`: coefficient `0.000161` (raises CT win probability)
- `lag_03__utility_damage_diff_last_5s`: coefficient `0.000131` (raises CT win probability)
- `lag_15__CT_flash_alpha_mean`: coefficient `-0.000120` (lowers CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000112` (raises CT win probability)
- `lag_08__CT_utility_damage_last_5s`: coefficient `-0.000107` (lowers CT win probability)
- `lag_01__CT3__utility_total`: coefficient `0.000104` (raises CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000097` (raises CT win probability)
- `lag_01__flash_inv_diff`: coefficient `0.000090` (raises CT win probability)
- `lag_08__utility_damage_diff_last_5s`: coefficient `-0.000088` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_10__T_place_TUNNELSTAIRS`: coefficient `-0.000237` (lowers CT win probability)
- `lag_00__T_place_OUTSIDETUNNEL`: coefficient `-0.000225` (lowers CT win probability)
- `lag_01__CT_place_UNDERA`: coefficient `-0.000220` (lowers CT win probability)
- `lag_01__CT_place_CTSPAWN`: coefficient `0.000195` (raises CT win probability)
- `lag_00__CT_place_EXTENDEDA`: coefficient `-0.000193` (lowers CT win probability)
- `lag_00__CT5__duck_amount`: coefficient `0.000178` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000167` (raises CT win probability)
- `lag_03__CT_place_MIDDOORS`: coefficient `0.000158` (raises CT win probability)
- `lag_00__T_place_OUTSIDELONG`: coefficient `-0.000154` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `0.000151` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `19001`, seconds `0.50`, LSTM delta `+0.0215`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `+0.000933`
- `lag_01__CT_flash_alpha_mean`: contribution `+0.000820`
- `lag_01__T_place_TSPAWN`: contribution `+0.000668`
- `lag_01__utility_inv_diff`: contribution `+0.000348`
- `lag_01__molly_inv_diff`: contribution `+0.000318`

Top utility-only movements:
- `lag_01__CT_flash_alpha_mean`: contribution `+0.000820`
- `lag_01__utility_inv_diff`: contribution `+0.000348`
- `lag_01__molly_inv_diff`: contribution `+0.000318`
- `lag_01__smoke_inv_diff`: contribution `+0.000224`
- `lag_01__CT_molly_inv`: contribution `+0.000221`

### tick `20601`, seconds `25.50`, LSTM delta `+0.0167`

Top all feature movements:
- `lag_10__T_place_TUNNELSTAIRS`: contribution `+0.001656`
- `lag_15__T_place_TUNNELSTAIRS`: contribution `+0.001016`
- `lag_08__CT_utility_damage_last_5s`: contribution `+0.000757`
- `lag_01__CT_place_ARAMP`: contribution `+0.000745`
- `lag_07__CT_place_ARAMP`: contribution `+0.000626`

Top utility-only movements:
- `lag_08__CT_utility_damage_last_5s`: contribution `+0.000757`
- `lag_08__utility_damage_diff_last_5s`: contribution `+0.000509`

### tick `19449`, seconds `7.50`, LSTM delta `+0.0152`

Top all feature movements:
- `lag_00__T_place_OUTSIDETUNNEL`: contribution `+0.001123`
- `lag_00__CT_place_EXTENDEDA`: contribution `+0.001086`
- `lag_08__CT_place_MIDDOORS`: contribution `+0.000854`
- `lag_15__CT_flash_alpha_mean`: contribution `+0.000525`
- `lag_00__CT_place_SHORTSTAIRS`: contribution `+0.000418`

Top utility-only movements:
- `lag_15__CT_flash_alpha_mean`: contribution `+0.000525`

### tick `20121`, seconds `18.00`, LSTM delta `+0.0132`

Top all feature movements:
- `lag_03__CT_utility_damage_last_5s`: contribution `+0.001133`
- `lag_00__CT_place_EXTENDEDA`: contribution `+0.001086`
- `lag_15__T_place_TUNNELSTAIRS`: contribution `+0.001016`
- `lag_00__T_place_TUNNELSTAIRS`: contribution `+0.000991`
- `lag_11__T_place_TUNNELSTAIRS`: contribution `+0.000982`

Top utility-only movements:
- `lag_03__CT_utility_damage_last_5s`: contribution `+0.001133`
- `lag_03__utility_damage_diff_last_5s`: contribution `+0.000759`

### tick `19705`, seconds `11.50`, LSTM delta `+0.0122`

Top all feature movements:
- `lag_02__T_place_TUNNELSTAIRS`: contribution `+0.000837`
- `lag_14__T_place_OUTSIDETUNNEL`: contribution `+0.000596`
- `lag_00__CT5__duck_amount`: contribution `+0.000530`
- `lag_11__CT_place_EXTENDEDA`: contribution `+0.000489`
- `lag_08__CT_place_EXTENDEDA`: contribution `+0.000433`

Top utility-only movements:
- `lag_04__CT_B_site_active_infernos`: contribution `+0.000181`
- `lag_07__CT1__molly`: contribution `+0.000173`
