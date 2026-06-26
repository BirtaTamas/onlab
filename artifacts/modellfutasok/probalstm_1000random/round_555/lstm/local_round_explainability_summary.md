# Local Round Explainability

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-tyloo-vs-rare-atom-bo3-8GB1HWZtKOlh9_707n2A62/tyloo-vs-rare-atom-m2-inferno.csv`
- round_num: `5`

## Largest probability jumps

- tick `39853`, seconds `21.50`, LSTM `0.1515`, delta `-0.3956`
- tick `39821`, seconds `21.00`, LSTM `0.5470`, delta `-0.1437`
- tick `39885`, seconds `22.00`, LSTM `0.0727`, delta `-0.0787`
- tick `39181`, seconds `11.00`, LSTM `0.6869`, delta `-0.0292`
- tick `39245`, seconds `12.00`, LSTM `0.6753`, delta `-0.0268`
- tick `38989`, seconds `8.00`, LSTM `0.6966`, delta `+0.0242`
- tick `39917`, seconds `22.50`, LSTM `0.0505`, delta `-0.0223`
- tick `39117`, seconds `10.00`, LSTM `0.7253`, delta `+0.0202`
- tick `39277`, seconds `12.50`, LSTM `0.6926`, delta `+0.0173`
- tick `39213`, seconds `11.50`, LSTM `0.7022`, delta `+0.0153`

## Top 15 local ridge features

- `lag_00__CT_place_APARTMENTS`: coefficient `0.002688`, |coef| `0.002688`
- `lag_12__T3__flash_duration`: coefficient `-0.002200`, |coef| `0.002200`
- `lag_01__CT_place_APARTMENTS`: coefficient `0.002144`, |coef| `0.002144`
- `lag_07__T_place_BALCONY`: coefficient `-0.002096`, |coef| `0.002096`
- `lag_00__T_kills_last_3s`: coefficient `-0.002020`, |coef| `0.002020`
- `lag_03__T3__flash_duration`: coefficient `0.001950`, |coef| `0.001950`
- `lag_00__T_damage_last_5s`: coefficient `-0.001828`, |coef| `0.001828`
- `lag_00__CT3__utility_total`: coefficient `0.001743`, |coef| `0.001743`
- `lag_01__T_kills_last_3s`: coefficient `-0.001733`, |coef| `0.001733`
- `lag_00__damage_diff_last_5s`: coefficient `0.001677`, |coef| `0.001677`
- `lag_01__CT1__duck_amount`: coefficient `0.001589`, |coef| `0.001589`
- `lag_01__T_damage_last_5s`: coefficient `-0.001564`, |coef| `0.001564`
- `lag_06__CT_B_site_active_infernos`: coefficient `-0.001558`, |coef| `0.001558`
- `lag_00__kill_diff_last_3s`: coefficient `0.001535`, |coef| `0.001535`
- `lag_00__CT3__molly`: coefficient `0.001503`, |coef| `0.001503`

## Top 10 utility ridge features

- `lag_12__T3__flash_duration`: coefficient `-0.002200` (lowers CT win probability)
- `lag_03__T3__flash_duration`: coefficient `0.001950` (raises CT win probability)
- `lag_00__CT3__utility_total`: coefficient `0.001743` (raises CT win probability)
- `lag_06__CT_B_site_active_infernos`: coefficient `-0.001558` (lowers CT win probability)
- `lag_00__CT3__molly`: coefficient `0.001503` (raises CT win probability)
- `lag_00__CT3__smoke`: coefficient `0.001347` (raises CT win probability)
- `lag_07__T_B_site_active_infernos`: coefficient `0.001290` (raises CT win probability)
- `lag_11__T3__flash_duration`: coefficient `-0.001280` (lowers CT win probability)
- `lag_08__CT5__molly`: coefficient `0.001208` (raises CT win probability)
- `lag_02__T3__flash_duration`: coefficient `0.001132` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_APARTMENTS`: coefficient `0.002688` (raises CT win probability)
- `lag_01__CT_place_APARTMENTS`: coefficient `0.002144` (raises CT win probability)
- `lag_07__T_place_BALCONY`: coefficient `-0.002096` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002020` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001828` (lowers CT win probability)
- `lag_01__T_kills_last_3s`: coefficient `-0.001733` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001677` (raises CT win probability)
- `lag_01__CT1__duck_amount`: coefficient `0.001589` (raises CT win probability)
- `lag_01__T_damage_last_5s`: coefficient `-0.001564` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001535` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `39853`, seconds `21.50`, LSTM delta `-0.3956`

Top all feature movements:
- `lag_07__T_place_BALCONY`: contribution `-0.028817`
- `lag_05__T_place_BALCONY`: contribution `-0.019955`
- `lag_12__T3__flash_duration`: contribution `-0.011100`
- `lag_00__CT_place_APARTMENTS`: contribution `-0.010325`
- `lag_03__T3__flash_duration`: contribution `-0.009838`

Top utility-only movements:
- `lag_12__T3__flash_duration`: contribution `-0.011100`
- `lag_03__T3__flash_duration`: contribution `-0.009838`
- `lag_06__CT_B_site_active_infernos`: contribution `-0.005352`
- `lag_00__CT3__utility_total`: contribution `-0.004991`

### tick `39821`, seconds `21.00`, LSTM delta `-0.1437`

Top all feature movements:
- `lag_04__T_place_BALCONY`: contribution `-0.013775`
- `lag_00__CT_place_APARTMENTS`: contribution `-0.010325`
- `lag_06__T_place_BALCONY`: contribution `-0.009152`
- `lag_11__T3__flash_duration`: contribution `-0.006458`
- `lag_00__T_kills_last_3s`: contribution `-0.006401`

Top utility-only movements:
- `lag_11__T3__flash_duration`: contribution `-0.006458`
- `lag_02__T3__flash_duration`: contribution `-0.005711`
- `lag_05__CT_B_site_active_infernos`: contribution `-0.003270`
- `lag_06__T_B_site_active_infernos`: contribution `-0.002268`

### tick `39885`, seconds `22.00`, LSTM delta `-0.0787`

Top all feature movements:
- `lag_08__T_place_BALCONY`: contribution `-0.012788`
- `lag_06__T_place_BALCONY`: contribution `+0.009152`
- `lag_01__CT_place_APARTMENTS`: contribution `-0.008236`
- `lag_01__T_kills_last_3s`: contribution `-0.005491`
- `lag_13__T3__flash_duration`: contribution `-0.004702`

Top utility-only movements:
- `lag_13__T3__flash_duration`: contribution `-0.004702`
- `lag_04__T3__flash_duration`: contribution `-0.003444`
- `lag_07__CT_B_site_active_infernos`: contribution `-0.002217`
- `lag_01__CT3__utility_total`: contribution `-0.002148`

### tick `39181`, seconds `11.00`, LSTM delta `-0.0292`

Top all feature movements:
- `lag_06__T3__duck_amount`: contribution `+0.003348`
- `lag_02__CT_place_BANANA`: contribution `-0.003222`
- `lag_12__CT_place_LIBRARY`: contribution `-0.002500`
- `lag_09__T_place_LOWERMID`: contribution `+0.002138`
- `lag_02__CT1__duck_amount`: contribution `-0.002104`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `39245`, seconds `12.00`, LSTM delta `-0.0268`

Top all feature movements:
- `lag_01__CT_place_APARTMENTS`: contribution `+0.008236`
- `lag_15__T_place_LOWERMID`: contribution `+0.003899`
- `lag_08__CT_place_ARCH`: contribution `-0.003281`
- `lag_15__CT_place_LIBRARY`: contribution `-0.002991`
- `lag_11__T_place_TRAMP`: contribution `+0.002616`

Top utility-only movements:
- No utility movement among the top local contributors.
