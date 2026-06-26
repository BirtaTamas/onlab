# Local Round Explainability

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-pain-vs-rare-atom-bo3-Rmb0_mvtIpTmOfUIJVjwOw/pain-vs-rare-atom-m1-inferno.csv`
- round_num: `6`

## Largest probability jumps

- tick `37174`, seconds `91.50`, LSTM `0.8512`, delta `+0.1811`
- tick `36982`, seconds `88.50`, LSTM `0.7056`, delta `+0.0660`
- tick `37014`, seconds `89.00`, LSTM `0.6456`, delta `-0.0600`
- tick `36822`, seconds `86.00`, LSTM `0.5815`, delta `+0.0537`
- tick `38006`, seconds `104.50`, LSTM `0.8592`, delta `+0.0392`
- tick `38038`, seconds `105.00`, LSTM `0.8880`, delta `+0.0288`
- tick `37366`, seconds `94.50`, LSTM `0.8282`, delta `-0.0284`
- tick `37558`, seconds `97.50`, LSTM `0.8099`, delta `+0.0276`
- tick `36886`, seconds `87.00`, LSTM `0.6292`, delta `+0.0242`
- tick `37526`, seconds `97.00`, LSTM `0.7823`, delta `-0.0239`

## Top 15 local ridge features

- `lag_01__T_place_ARCH`: coefficient `0.001720`, |coef| `0.001720`
- `lag_00__T_place_ARCH`: coefficient `-0.001484`, |coef| `0.001484`
- `lag_00__CT_kills_last_3s`: coefficient `0.001440`, |coef| `0.001440`
- `lag_06__T4__flash_duration`: coefficient `0.001406`, |coef| `0.001406`
- `lag_00__CT_place_BALCONY`: coefficient `-0.001289`, |coef| `0.001289`
- `lag_00__kill_diff_last_3s`: coefficient `0.001200`, |coef| `0.001200`
- `lag_08__CT4__flash_duration`: coefficient `0.001131`, |coef| `0.001131`
- `lag_00__T_place_UPSTAIRS`: coefficient `0.001120`, |coef| `0.001120`
- `lag_06__T2__flash_duration`: coefficient `0.000910`, |coef| `0.000910`
- `lag_13__CT3__duck_amount`: coefficient `0.000895`, |coef| `0.000895`
- `lag_02__CT4__flash_duration`: coefficient `0.000880`, |coef| `0.000880`
- `lag_06__T_flash_duration_sum`: coefficient `0.000838`, |coef| `0.000838`
- `lag_00__T1__shots_fired`: coefficient `0.000825`, |coef| `0.000825`
- `lag_06__CT3__duck_amount`: coefficient `-0.000818`, |coef| `0.000818`
- `lag_13__T_place_TRAMP`: coefficient `0.000801`, |coef| `0.000801`

## Top 10 utility ridge features

- `lag_06__T4__flash_duration`: coefficient `0.001406` (raises CT win probability)
- `lag_08__CT4__flash_duration`: coefficient `0.001131` (raises CT win probability)
- `lag_06__T2__flash_duration`: coefficient `0.000910` (raises CT win probability)
- `lag_02__CT4__flash_duration`: coefficient `0.000880` (raises CT win probability)
- `lag_06__T_flash_duration_sum`: coefficient `0.000838` (raises CT win probability)
- `lag_00__T4__flash_duration`: coefficient `0.000724` (raises CT win probability)
- `lag_07__T_B_site_active_infernos`: coefficient `0.000648` (raises CT win probability)
- `lag_00__T2__flash_duration`: coefficient `0.000645` (raises CT win probability)
- `lag_14__T5__smoke`: coefficient `-0.000641` (lowers CT win probability)
- `lag_11__T2__molly`: coefficient `-0.000621` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__T_place_ARCH`: coefficient `0.001720` (raises CT win probability)
- `lag_00__T_place_ARCH`: coefficient `-0.001484` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001440` (raises CT win probability)
- `lag_00__CT_place_BALCONY`: coefficient `-0.001289` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001200` (raises CT win probability)
- `lag_00__T_place_UPSTAIRS`: coefficient `0.001120` (raises CT win probability)
- `lag_13__CT3__duck_amount`: coefficient `0.000895` (raises CT win probability)
- `lag_00__T1__shots_fired`: coefficient `0.000825` (raises CT win probability)
- `lag_06__CT3__duck_amount`: coefficient `-0.000818` (lowers CT win probability)
- `lag_13__T_place_TRAMP`: coefficient `0.000801` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `37174`, seconds `91.50`, LSTM delta `+0.1811`

Top all feature movements:
- `lag_01__T_place_ARCH`: contribution `+0.016003`
- `lag_00__T_place_ARCH`: contribution `+0.013811`
- `lag_06__T4__flash_duration`: contribution `+0.008674`
- `lag_08__CT4__flash_duration`: contribution `+0.007425`
- `lag_06__T2__flash_duration`: contribution `+0.005077`

Top utility-only movements:
- `lag_06__T4__flash_duration`: contribution `+0.008674`
- `lag_08__CT4__flash_duration`: contribution `+0.007425`
- `lag_06__T2__flash_duration`: contribution `+0.005077`
- `lag_06__T_flash_duration_sum`: contribution `+0.004014`

### tick `36982`, seconds `88.50`, LSTM delta `+0.0660`

Top all feature movements:
- `lag_02__CT4__flash_duration`: contribution `+0.005778`
- `lag_00__T4__flash_duration`: contribution `+0.004468`
- `lag_00__T2__flash_duration`: contribution `+0.003599`
- `lag_06__CT3__duck_amount`: contribution `+0.003045`
- `lag_00__T_flash_duration_sum`: contribution `+0.002934`

Top utility-only movements:
- `lag_02__CT4__flash_duration`: contribution `+0.005778`
- `lag_00__T4__flash_duration`: contribution `+0.004468`
- `lag_00__T2__flash_duration`: contribution `+0.003599`
- `lag_00__T_flash_duration_sum`: contribution `+0.002934`
- `lag_02__CT4__molly`: contribution `+0.001273`

### tick `37014`, seconds `89.00`, LSTM delta `-0.0600`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `-0.004156`
- `lag_03__CT4__flash_duration`: contribution `-0.003036`
- `lag_00__kill_diff_last_3s`: contribution `-0.002889`
- `lag_01__T4__flash_duration`: contribution `-0.002807`
- `lag_08__CT3__duck_amount`: contribution `-0.001880`

Top utility-only movements:
- `lag_03__CT4__flash_duration`: contribution `-0.003036`
- `lag_01__T4__flash_duration`: contribution `-0.002807`
- `lag_02__CT_B_site_active_infernos`: contribution `-0.001557`
- `lag_06__T3__smoke`: contribution `-0.001146`

### tick `36822`, seconds `86.00`, LSTM delta `+0.0537`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.004156`
- `lag_00__kill_diff_last_3s`: contribution `+0.002889`
- `lag_02__T5__duck_amount`: contribution `+0.002274`
- `lag_02__CT3__duck_amount`: contribution `+0.001956`
- `lag_00__CT4__duck_amount`: contribution `+0.001899`

Top utility-only movements:
- `lag_00__T2__molly`: contribution `+0.001170`
- `lag_14__CT_B_site_active_infernos`: contribution `+0.001164`
- `lag_06__T3__smoke`: contribution `+0.001146`
- `lag_06__T2__smoke`: contribution `+0.001040`

### tick `38006`, seconds `104.50`, LSTM delta `+0.0392`

Top all feature movements:
- `lag_00__CT_place_BALCONY`: contribution `+0.008273`
- `lag_14__CT3__duck_amount`: contribution `+0.002345`
- `lag_13__T_place_TRAMP`: contribution `+0.002343`
- `lag_07__CT3__is_walking`: contribution `+0.001533`
- `lag_15__T2__duck_amount`: contribution `+0.001490`

Top utility-only movements:
- No utility movement among the top local contributors.
