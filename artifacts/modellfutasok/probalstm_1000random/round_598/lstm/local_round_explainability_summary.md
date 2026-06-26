# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-aurora-vs-astralis-bo3-EH-le-_LuObR5nGefXVoZY/aurora-vs-astralis-m3-overpass.csv`
- round_num: `11`

## Largest probability jumps

- tick `98772`, seconds `77.50`, LSTM `0.1553`, delta `-0.3773`
- tick `98708`, seconds `76.50`, LSTM `0.5545`, delta `-0.1670`
- tick `97396`, seconds `56.00`, LSTM `0.7718`, delta `+0.1147`
- tick `97108`, seconds `51.50`, LSTM `0.6842`, delta `+0.0511`
- tick `97588`, seconds `59.00`, LSTM `0.6809`, delta `-0.0506`
- tick `97300`, seconds `54.50`, LSTM `0.6270`, delta `-0.0487`
- tick `97460`, seconds `57.00`, LSTM `0.7357`, delta `-0.0468`
- tick `98804`, seconds `78.00`, LSTM `0.1092`, delta `-0.0461`
- tick `96788`, seconds `46.50`, LSTM `0.5642`, delta `-0.0460`
- tick `96948`, seconds `49.00`, LSTM `0.6157`, delta `+0.0457`

## Top 15 local ridge features

- `lag_06__CT2__flash_duration`: coefficient `-0.003074`, |coef| `0.003074`
- `lag_15__CT4__flash_duration`: coefficient `-0.003028`, |coef| `0.003028`
- `lag_00__CT2__flash_duration`: coefficient `0.002908`, |coef| `0.002908`
- `lag_00__T_kills_last_3s`: coefficient `-0.002508`, |coef| `0.002508`
- `lag_00__CT_place_CANAL`: coefficient `0.002446`, |coef| `0.002446`
- `lag_00__kill_diff_last_3s`: coefficient `0.002407`, |coef| `0.002407`
- `lag_03__T_place_ALLEY`: coefficient `0.002342`, |coef| `0.002342`
- `lag_00__damage_diff_last_5s`: coefficient `0.002296`, |coef| `0.002296`
- `lag_00__T_damage_last_5s`: coefficient `-0.002245`, |coef| `0.002245`
- `lag_05__CT4__flash_duration`: coefficient `0.002229`, |coef| `0.002229`
- `lag_13__CT4__flash_duration`: coefficient `-0.002142`, |coef| `0.002142`
- `lag_13__CT_place_WATER`: coefficient `-0.002020`, |coef| `0.002020`
- `lag_00__CT2__duck_amount`: coefficient `-0.001976`, |coef| `0.001976`
- `lag_03__T_place_CANAL`: coefficient `-0.001906`, |coef| `0.001906`
- `lag_05__CT_place_WATER`: coefficient `0.001810`, |coef| `0.001810`

## Top 10 utility ridge features

- `lag_06__CT2__flash_duration`: coefficient `-0.003074` (lowers CT win probability)
- `lag_15__CT4__flash_duration`: coefficient `-0.003028` (lowers CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `0.002908` (raises CT win probability)
- `lag_05__CT4__flash_duration`: coefficient `0.002229` (raises CT win probability)
- `lag_13__CT4__flash_duration`: coefficient `-0.002142` (lowers CT win probability)
- `lag_04__CT2__flash_duration`: coefficient `-0.001690` (lowers CT win probability)
- `lag_15__CT_flash_duration_sum`: coefficient `-0.001565` (lowers CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `0.001474` (raises CT win probability)
- `lag_03__CT4__flash_duration`: coefficient `0.001224` (raises CT win probability)
- `lag_13__CT_flash_duration_sum`: coefficient `-0.001192` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.002508` (lowers CT win probability)
- `lag_00__CT_place_CANAL`: coefficient `0.002446` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002407` (raises CT win probability)
- `lag_03__T_place_ALLEY`: coefficient `0.002342` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002296` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002245` (lowers CT win probability)
- `lag_13__CT_place_WATER`: coefficient `-0.002020` (lowers CT win probability)
- `lag_00__CT2__duck_amount`: coefficient `-0.001976` (lowers CT win probability)
- `lag_03__T_place_CANAL`: coefficient `-0.001906` (lowers CT win probability)
- `lag_05__CT_place_WATER`: coefficient `0.001810` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `98772`, seconds `77.50`, LSTM delta `-0.3773`

Top all feature movements:
- `lag_06__CT2__flash_duration`: contribution `-0.020628`
- `lag_00__CT2__flash_duration`: contribution `-0.019515`
- `lag_15__CT4__flash_duration`: contribution `-0.018393`
- `lag_00__CT_place_CANAL`: contribution `-0.014867`
- `lag_05__CT4__flash_duration`: contribution `-0.013538`

Top utility-only movements:
- `lag_06__CT2__flash_duration`: contribution `-0.020628`
- `lag_00__CT2__flash_duration`: contribution `-0.019515`
- `lag_15__CT4__flash_duration`: contribution `-0.018393`
- `lag_05__CT4__flash_duration`: contribution `-0.013538`

### tick `98708`, seconds `76.50`, LSTM delta `-0.1670`

Top all feature movements:
- `lag_13__CT4__flash_duration`: contribution `-0.013011`
- `lag_04__CT2__flash_duration`: contribution `-0.011342`
- `lag_00__T_kills_last_3s`: contribution `-0.007946`
- `lag_03__CT_place_WATER`: contribution `-0.007864`
- `lag_03__CT4__flash_duration`: contribution `-0.007435`

Top utility-only movements:
- `lag_13__CT4__flash_duration`: contribution `-0.013011`
- `lag_04__CT2__flash_duration`: contribution `-0.011342`
- `lag_03__CT4__flash_duration`: contribution `-0.007435`
- `lag_13__CT_flash_duration_sum`: contribution `-0.003237`

### tick `97396`, seconds `56.00`, LSTM delta `+0.1147`

Top all feature movements:
- `lag_13__CT_place_BRIDGE`: contribution `+0.013727`
- `lag_12__CT_place_BACKOFA`: contribution `+0.012098`
- `lag_11__CT_place_BRIDGE`: contribution `+0.008155`
- `lag_12__CT_place_STAIRS`: contribution `+0.008038`
- `lag_15__CT_place_STAIRS`: contribution `+0.006721`

Top utility-only movements:
- `lag_00__CT2__flash_duration`: contribution `+0.003981`
- `lag_14__CT_B_site_active_infernos`: contribution `+0.002088`

### tick `97108`, seconds `51.50`, LSTM delta `+0.0511`

Top all feature movements:
- `lag_07__CT_place_CONSTRUCTION`: contribution `+0.007882`
- `lag_01__T_place_ALLEY`: contribution `+0.007208`
- `lag_04__CT_place_BRIDGE`: contribution `+0.006129`
- `lag_03__CT_place_STAIRS`: contribution `+0.005076`
- `lag_06__CT_place_STAIRS`: contribution `+0.004809`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `97588`, seconds `59.00`, LSTM delta `-0.0506`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.005793`
- `lag_02__CT_place_CANAL`: contribution `+0.005769`
- `lag_00__CT_place_LOBBY`: contribution `-0.004381`
- `lag_06__CT2__flash_duration`: contribution `-0.004208`
- `lag_02__T_place_ALLEY`: contribution `-0.003751`

Top utility-only movements:
- `lag_06__CT2__flash_duration`: contribution `-0.004208`
- `lag_04__CT2__flash_duration`: contribution `+0.002314`
