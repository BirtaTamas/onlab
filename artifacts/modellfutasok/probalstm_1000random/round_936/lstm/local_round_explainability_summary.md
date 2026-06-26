# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-nemiga-vs-m80-bo3-A9YADMgFNfEy-U6IHDyx-U/nemiga-vs-m80-m2-dust2.csv`
- round_num: `8`

## Largest probability jumps

- tick `66743`, seconds `33.50`, LSTM `0.3356`, delta `+0.2152`
- tick `65943`, seconds `21.00`, LSTM `0.1795`, delta `-0.1804`
- tick `66519`, seconds `30.00`, LSTM `0.1870`, delta `+0.1012`
- tick `66487`, seconds `29.50`, LSTM `0.0859`, delta `-0.0880`
- tick `67703`, seconds `48.50`, LSTM `0.0137`, delta `-0.0697`
- tick `66551`, seconds `30.50`, LSTM `0.1326`, delta `-0.0545`
- tick `67287`, seconds `42.00`, LSTM `0.1421`, delta `-0.0541`
- tick `66263`, seconds `26.00`, LSTM `0.1046`, delta `+0.0439`
- tick `67255`, seconds `41.50`, LSTM `0.1962`, delta `-0.0412`
- tick `66455`, seconds `29.00`, LSTM `0.1738`, delta `+0.0408`

## Top 15 local ridge features

- `lag_00__CT_place_UPPERTUNNEL`: coefficient `-0.001809`, |coef| `0.001809`
- `lag_00__T_kills_last_3s`: coefficient `-0.001363`, |coef| `0.001363`
- `lag_15__T_place_SIDE`: coefficient `-0.001267`, |coef| `0.001267`
- `lag_03__CT_place_TUNNELSTAIRS`: coefficient `0.001254`, |coef| `0.001254`
- `lag_05__CT_place_EXTENDEDA`: coefficient `-0.001228`, |coef| `0.001228`
- `lag_15__T4__flash_duration`: coefficient `0.001173`, |coef| `0.001173`
- `lag_02__CT_place_EXTENDEDA`: coefficient `0.001172`, |coef| `0.001172`
- `lag_14__T_place_EXTENDEDA`: coefficient `-0.001108`, |coef| `0.001108`
- `lag_07__CT4__flash_duration`: coefficient `0.001098`, |coef| `0.001098`
- `lag_00__kill_diff_last_3s`: coefficient `0.001097`, |coef| `0.001097`
- `lag_03__CT_place_UPPERTUNNEL`: coefficient `-0.001079`, |coef| `0.001079`
- `lag_08__T_place_SIDE`: coefficient `-0.001040`, |coef| `0.001040`
- `lag_13__T_place_LONGDOORS`: coefficient `-0.001023`, |coef| `0.001023`
- `lag_13__CT_place_UNDERA`: coefficient `-0.001015`, |coef| `0.001015`
- `lag_09__T_place_SIDE`: coefficient `0.001010`, |coef| `0.001010`

## Top 10 utility ridge features

- `lag_15__T4__flash_duration`: coefficient `0.001173` (raises CT win probability)
- `lag_07__CT4__flash_duration`: coefficient `0.001098` (raises CT win probability)
- `lag_05__T1__flash_duration`: coefficient `0.000899` (raises CT win probability)
- `lag_08__T_flash_duration_sum`: coefficient `0.000831` (raises CT win probability)
- `lag_00__CT1__molly`: coefficient `0.000736` (raises CT win probability)
- `lag_08__T4__flash_duration`: coefficient `0.000730` (raises CT win probability)
- `lag_08__T5__flash_duration`: coefficient `0.000722` (raises CT win probability)
- `lag_07__T5__flash_duration`: coefficient `0.000693` (raises CT win probability)
- `lag_00__CT1__smoke`: coefficient `0.000641` (raises CT win probability)
- `lag_12__CT_B_site_active_infernos`: coefficient `0.000612` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_UPPERTUNNEL`: coefficient `-0.001809` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001363` (lowers CT win probability)
- `lag_15__T_place_SIDE`: coefficient `-0.001267` (lowers CT win probability)
- `lag_03__CT_place_TUNNELSTAIRS`: coefficient `0.001254` (raises CT win probability)
- `lag_05__CT_place_EXTENDEDA`: coefficient `-0.001228` (lowers CT win probability)
- `lag_02__CT_place_EXTENDEDA`: coefficient `0.001172` (raises CT win probability)
- `lag_14__T_place_EXTENDEDA`: coefficient `-0.001108` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001097` (raises CT win probability)
- `lag_03__CT_place_UPPERTUNNEL`: coefficient `-0.001079` (lowers CT win probability)
- `lag_08__T_place_SIDE`: coefficient `-0.001040` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `66743`, seconds `33.50`, LSTM delta `+0.2152`

Top all feature movements:
- `lag_15__T_place_SIDE`: contribution `+0.024507`
- `lag_03__CT_place_TUNNELSTAIRS`: contribution `+0.017660`
- `lag_02__CT_place_TUNNELSTAIRS`: contribution `+0.012925`
- `lag_03__CT_place_UPPERTUNNEL`: contribution `+0.008276`
- `lag_08__T_flash_duration_sum`: contribution `+0.007609`

Top utility-only movements:
- `lag_08__T_flash_duration_sum`: contribution `+0.007609`
- `lag_08__T4__flash_duration`: contribution `+0.005211`
- `lag_08__T5__flash_duration`: contribution `+0.004633`
- `lag_08__T1__flash_duration`: contribution `+0.004591`
- `lag_07__T5__flash_duration`: contribution `-0.004445`

### tick `65943`, seconds `21.00`, LSTM delta `-0.1804`

Top all feature movements:
- `lag_00__CT_place_UPPERTUNNEL`: contribution `-0.013873`
- `lag_05__CT_place_EXTENDEDA`: contribution `-0.006896`
- `lag_07__CT4__flash_duration`: contribution `-0.006795`
- `lag_02__CT_place_EXTENDEDA`: contribution `-0.006579`
- `lag_13__CT_place_UNDERA`: contribution `-0.006199`

Top utility-only movements:
- `lag_07__CT4__flash_duration`: contribution `-0.006795`
- `lag_15__T4__flash_duration`: contribution `-0.006183`
- `lag_05__T1__flash_duration`: contribution `-0.005979`
- `lag_07__T5__flash_duration`: contribution `-0.003965`

### tick `66519`, seconds `30.00`, LSTM delta `+0.1012`

Top all feature movements:
- `lag_08__T_place_SIDE`: contribution `+0.020112`
- `lag_09__T_place_SIDE`: contribution `+0.019545`
- `lag_05__CT_place_EXTENDEDA`: contribution `+0.006896`
- `lag_07__T_place_TUNNELSTAIRS`: contribution `-0.005522`
- `lag_12__T_place_LONGA`: contribution `+0.004061`

Top utility-only movements:
- `lag_00__T5__flash_duration`: contribution `+0.003430`
- `lag_01__T4__flash_duration`: contribution `+0.003325`
- `lag_15__CT5__flash_duration`: contribution `+0.003009`
- `lag_03__CT5__flash_duration`: contribution `+0.002935`
- `lag_01__T_flash_duration_sum`: contribution `+0.002247`

### tick `66487`, seconds `29.50`, LSTM delta `-0.0880`

Top all feature movements:
- `lag_08__T_place_SIDE`: contribution `-0.020112`
- `lag_07__T_place_SIDE`: contribution `-0.013083`
- `lag_00__T_kills_last_3s`: contribution `-0.004319`
- `lag_07__T_place_LONGA`: contribution `-0.003753`
- `lag_06__T_place_LOWERTUNNEL`: contribution `-0.003574`

Top utility-only movements:
- `lag_00__T5__flash_duration`: contribution `-0.003430`
- `lag_00__T1__flash_duration`: contribution `-0.003388`
- `lag_00__T_flash_duration_sum`: contribution `-0.003357`

### tick `67703`, seconds `48.50`, LSTM delta `-0.0697`

Top all feature movements:
- `lag_00__CT_place_OUTSIDELONG`: contribution `-0.009631`
- `lag_01__T3__is_scoped`: contribution `-0.005871`
- `lag_14__CT_place_OUTSIDELONG`: contribution `-0.005238`
- `lag_00__T_kills_last_3s`: contribution `-0.004319`
- `lag_09__CT_place_LONGDOORS`: contribution `-0.002840`

Top utility-only movements:
- `lag_09__CT_A_site_active_infernos`: contribution `-0.001109`
- `lag_06__T_A_site_active_infernos`: contribution `-0.000987`
