# Modellfutasok osszegzes

Ebben a mappaban jelenleg 3 osszehasonlithato XGBoost futas van ugyanazon a kisebb mintan:

- `xgboost_30pct_no_tick_basic_d5_n400_lr005`
- `xgboost_30pct_no_tick_basic_d6_n400_lr005`
- `xgboost_30pct_no_tick_basic_d6_n600_lr005`

## Alaprangsor

Az alabbi sorrendet hasznalom vegig ebben az osszegzesben:

1. `xgboost_30pct_no_tick_basic_d6_n400_lr005`
2. `xgboost_30pct_no_tick_basic_d5_n400_lr005`
3. `xgboost_30pct_no_tick_basic_d6_n600_lr005`

Ez a fo sorrend elsosorban az altalanos osztalyozasi teljesitmenyt priorizalja:

- `f1_score`
- `accuracy`
- `precision`
- `roc_auc`

Megjegyzes:

- ha a probability quality a fontosabb, akkor a `d5_n400` nagyon eros, mert `brier_score`, `logloss` es `roc_auc` alapjan egy hajszallal jobb, mint a `d6_n400`
- ennek ellenere az alaprangsorban a `d6_n400` van elol, mert nala a klasszikus osztalyozasi metrikak osszessegeben kicsit jobbak

## Fobb metrikak

| Modell | Accuracy | Precision | Recall | F1 | Brier | Logloss | ROC AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| `xgboost_30pct_no_tick_basic_d6_n400_lr005` | `0.74898` | `0.74786` | `0.78032` | `0.76375` | `0.16008` | `0.47345` | `0.84455` |
| `xgboost_30pct_no_tick_basic_d5_n400_lr005` | `0.74761` | `0.74526` | `0.78185` | `0.76312` | `0.15959` | `0.47178` | `0.84460` |
| `xgboost_30pct_no_tick_basic_d6_n600_lr005` | `0.74278` | `0.74592` | `0.76635` | `0.75600` | `0.16305` | `0.48149` | `0.84049` |

## Melyik modell van melyik elott

### 1. hely: `xgboost_30pct_no_tick_basic_d6_n400_lr005`

Ez a jelenlegi alapgyoztes.

Miert van elol:

- a legjobb `accuracy`
- a legjobb `precision`
- a legjobb `f1_score`
- `roc_auc`-ban gyakorlatilag egy szinten van a `d5_n400` modellel

Miert nincs nagy folenye:

- `brier_score` es `logloss` alapjan nem ez a legjobb
- a `d5_n400` probability-minosegben nagyon kozel van, sot ott jobb is

### 2. hely: `xgboost_30pct_no_tick_basic_d5_n400_lr005`

Ez a legkozvetlenebb kihivo.

Miert van a `d6_n400` mogott:

- kicsit gyengebb `accuracy`
- kicsit gyengebb `precision`
- kicsit gyengebb `f1_score`

Miert eros modell:

- a legjobb `recall`
- a legjobb `brier_score`
- a legjobb `logloss`
- a legjobb `roc_auc`

Ertelmezes:

- ha a cel az, hogy a modell valoszinusegi outputja legyen egy picit jobb, akkor ez akar elonyosebb valasztas is lehet
- ha a cel a klasszikus cimkealapu osztalyozas, akkor hajszallal a `d6_n400` marad elotte

### 3. hely: `xgboost_30pct_no_tick_basic_d6_n600_lr005`

Ez a jelenlegi leggyengebb a harom kozul.

Miert csuszott hatra:

- rosszabb `accuracy`
- rosszabb `recall`
- rosszabb `f1_score`
- rosszabb `roc_auc`
- rosszabb `logloss`
- rosszabb `brier_score`

Valoszinuleg mi tortent:

- a tobb fa (`n_estimators=600`) trainen jobban rafeszul a mintara
- validon es teszten ez mar nem generalizal olyan jol
- ez enyhe tulillesztesre utal

## Gyors kovetkeztetes

Ha most egyetlen modellt kell valasztani altalanos baseline-nak, akkor:

1. `xgboost_30pct_no_tick_basic_d6_n400_lr005`

Ha viszont a probability output minosege fontosabb, mint a nagyon pici `accuracy` vagy `f1` kulonbseg, akkor ezt is erdemes komolyan nezni:

1. `xgboost_30pct_no_tick_basic_d5_n400_lr005`

## Javasolt kovetkezo irany

A mostani kep alapjan:

- a `n_estimators=600` irany nem segitett
- a `depth=5` erdekesebb, mert nagyon kozel jott a gyozteshez

Ezert a kovetkezo ertelmes probak:

1. `d5_n400_lr005` korul finomhangolni
2. vagy `d6_n400`-rol indulva a `learning_rate`-et csokkenteni

## Kapcsolodo mappak

- [d6_n400](/Users/birtatamas/Tanulás/onlab_data/onlab/artifacts/modellfutasok/xgboost_30pct_no_tick_basic_d6_n400_lr005)
- [d5_n400](/Users/birtatamas/Tanulás/onlab_data/onlab/artifacts/modellfutasok/xgboost_30pct_no_tick_basic_d5_n400_lr005)
- [d6_n600](/Users/birtatamas/Tanulás/onlab_data/onlab/artifacts/modellfutasok/xgboost_30pct_no_tick_basic_d6_n600_lr005)
