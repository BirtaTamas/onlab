# 3 Foparameter Osszehasonlitas

Ez a jegyzet az elso nagyon egyszeru hyperparameter-kort foglalja ossze, ahol csak 3 fo parameterhez nyultunk:

- `max_depth`
- `n_estimators`
- `learning_rate`

Ebben a korben a `30%`-os mintan neztuk meg, mi tortenik, ha a korabbi baseline `6 / 400 / 0.05` beallitast egy nagyon kozeli iranyba toljuk el:

- `6 / 500 / 0.05`

Es ezt mindket feature-csaladban lefuttattuk:

- `no_utility`
- `all_utility`

## Osszehasonlitott Futasok

No utility par:

- `artifacts/xgboost_baseline_no_utility`
- `artifacts/xgboost_30pct_no_utility_d6_n500_lr005`

All utility par:

- `artifacts/xgboost_baseline_all_utility`
- `artifacts/xgboost_30pct_all_utility_d6_n500_lr005`

## Mi Valtozott

### Regi 30%-os baseline

- `max_depth = 6`
- `n_estimators = 400`
- `learning_rate = 0.05`

A tobbi parameter a script defaultja maradt:

- `min_child_weight = 5`
- `subsample = 0.8`
- `colsample_bytree = 0.8`
- `reg_lambda = 2.0`
- `reg_alpha = 0.0`

### Uj proba

- `max_depth = 6`
- `n_estimators = 500`
- `learning_rate = 0.05`

Vagyis ebben a korben csak egyetlen dolgot valtoztattunk:

- `400 -> 500` fa

## Eredmenyek

### Baseline No Utility

- train: accuracy `0.8951`, logloss `0.2994`, roc_auc `0.9642`
- valid: accuracy `0.7458`, logloss `0.4713`, roc_auc `0.8452`
- test: accuracy `0.7465`, logloss `0.4729`, roc_auc `0.8435`

### Uj No Utility: D6 N500 LR0.05

- train: accuracy `0.9138`, logloss `0.2742`, roc_auc `0.9746`
- valid: accuracy `0.7426`, logloss `0.4749`, roc_auc `0.8435`
- test: accuracy `0.7438`, logloss `0.4779`, roc_auc `0.8409`

### Baseline All Utility

- train: accuracy `0.8975`, logloss `0.2982`, roc_auc `0.9654`
- valid: accuracy `0.7491`, logloss `0.4689`, roc_auc `0.8470`
- test: accuracy `0.7457`, logloss `0.4731`, roc_auc `0.8443`

### Uj All Utility: D6 N500 LR0.05

- train: accuracy `0.9156`, logloss `0.2736`, roc_auc `0.9755`
- valid: accuracy `0.7486`, logloss `0.4718`, roc_auc `0.8462`
- test: accuracy `0.7451`, logloss `0.4760`, roc_auc `0.8433`

## Kozvetlen Osszehasonlitas

### No Utility: `uj - baseline`

- train accuracy: `+0.0187`
- train logloss: `-0.0252`
- train roc_auc: `+0.0103`
- valid accuracy: `-0.0032`
- valid logloss: `+0.0035`
- valid roc_auc: `-0.0016`
- test accuracy: `-0.0027`
- test logloss: `+0.0049`
- test roc_auc: `-0.0026`

Olvasat:

- trainen egyertelmuen jobb lett
- validon es teszten viszont minden fontos metrika picit romlott
- ez enyhe tulilleszkedesre utal

### All Utility: `uj - baseline`

- train accuracy: `+0.0180`
- train logloss: `-0.0246`
- train roc_auc: `+0.0101`
- valid accuracy: `-0.0005`
- valid logloss: `+0.0029`
- valid roc_auc: `-0.0009`
- test accuracy: `-0.0006`
- test logloss: `+0.0029`
- test roc_auc: `-0.0011`

Olvasat:

- ugyanaz a minta latszik, mint `no_utility` mellett
- trainen jobb, de validon es teszten nem hoz javulast
- itt is az latszik, hogy a plusz `100` fa inkabb tovabb illeszti a modellt a train adatra

### Uj par osszehasonlitasa: `all_utility - no_utility`

- train accuracy: `+0.0018`
- train logloss: `-0.0006`
- train roc_auc: `+0.0010`
- valid accuracy: `+0.0061`
- valid logloss: `-0.0031`
- valid roc_auc: `+0.0026`
- test accuracy: `+0.0013`
- test logloss: `-0.0019`
- test roc_auc: `+0.0024`

Olvasat:

- az uj `6 / 500 / 0.05` pontban a utilitys modell tovabbra is hajszallal jobb, mint a no-utilitys
- de ez a kulonbseg kicsi
- a fo jel itt nem a utility vs no-utility kulonbseg, hanem az, hogy az uj parameterpont egyik csaladban sem verte meg a sajat baseline-jat

## Mit Jelent Ez Egyutt

A mostani 3-foparameteres kiserletbol a legerosebb minta ez:

- a `400 -> 500` emeles mindket csaladban javitotta a train score-t
- de nem javitotta a valid es test teljesitmenyt
- vagyis ez a pont nem a jobb generalizacio, hanem inkabb a hosszabb train-illeszkedes iranyaba vitte a modellt

Maskepp fogalmazva:

- a `6 / 400 / 0.05` jelenleg jobb referencia marad, mint a `6 / 500 / 0.05`

## Kapcsolat A Reduced Kiserletekkel

Itt mar erdemes kulon kezelni ket szintet:

- a fair hyperparameter-osszehasonlitast a sajat baseline-jukhoz
- es egy kulon, ovatosabb osszevetest a reduced csaladdal

A reduced korok mas kerdesre valaszoltak:

- ott azt neztuk, hogy a utility onallo hozzajarulasa jobban latszik-e, ha a legerosebb nem-utility feature-okat kivesszuk

Itt viszont a kerdes ez volt:

- a teljes, normal feature-csaladban mit csinal egy nagyon kozeli hyperparameter-valtoztatas

Tehat:

- a reduced futasok jo diagnosztikai kontrollok
- de a mostani `6 / 500 / 0.05` kor fo referenciapontja a sajat teljes-feature-os baseline-ja

### Normal No Utility vs Reduced No Utility

- normal: `artifacts/xgboost_30pct_no_utility_d6_n500_lr005`
- reduced: `artifacts/xgboost_ablation_reduced_no_utility_libdefaults`

Kulonbseg `normal - reduced`:

- feature count: `+63`
- train accuracy: `+0.0863`
- train logloss: `-0.1172`
- train roc_auc: `+0.0555`
- valid accuracy: `+0.0173`
- valid logloss: `-0.0296`
- valid roc_auc: `+0.0269`
- test accuracy: `+0.0045`
- test logloss: `-0.0172`
- test roc_auc: `+0.0186`

Olvasat:

- a teljes `no_utility` feature-keszlet egyertelmuen erosebb, mint a reduced `no_utility`
- ez vart eredmeny, mert a reduced verzio pont az eros `alive / hp / armor / economy / equip` blokkokat dobta ki

### Normal All Utility vs Reduced With Utility

- normal: `artifacts/xgboost_30pct_all_utility_d6_n500_lr005`
- reduced: `artifacts/xgboost_ablation_reduced_with_utility_libdefaults`

Kulonbseg `normal - reduced`:

- feature count: `+63`
- train accuracy: `+0.0677`
- train logloss: `-0.0912`
- train roc_auc: `+0.0432`
- valid accuracy: `+0.0058`
- valid logloss: `-0.0077`
- valid roc_auc: `+0.0075`
- test accuracy: `+0.0058`
- test logloss: `-0.0068`
- test roc_auc: `+0.0095`

Olvasat:

- a teljes `all_utility` feature-keszlet is jobb marad, mint a reduced utilitys verzio
- ez is vart, mert a reduced setup itt sem vegso teljesitmenyre, hanem diagnosztikai szetvalasztasra lett kitalalva

### Mit Mutat Ez A Negyes Osszevetes

Ha a negy modellt egyutt nezzuk:

- `normal no_utility`
- `normal all_utility`
- `reduced no_utility`
- `reduced with_utility`

akkor ket kulon allitas latszik egyszerre:

- a normal modellek jobbak a reduced modelleknel, mert tobb eros feature-informaciot hasznalnak
- a reduced paron belul a utility javit

Ez fontos, mert igy nem keveredik ossze ket kulon kerdes:

- mi adja a legjobb osszteljesitmenyt
- es mi mutatja meg tisztabban a utility marginalis hasznat

Rovid valasz:

- legjobb osszteljesitmeny: a normal teljes feature-keszletu modellek
- utility-hatas tisztabb kimutatasa: a reduced par

## Elso Kovetkeztetes

Ebbol az elso nagyon kozeli probabol a legkorrektebb olvasat ez:

- pusztan a fa-szam novelese `400`-rol `500`-ra nem hozott javulast
- mind `no_utility`, mind `all_utility` esetben inkabb enyhe overfittinget erositett
- a utilitys verzio tovabbra is picit jobb maradt a no-utilitysnel, de a kulonbseg kicsi

## Javasolt Kovetkezo Lepesek

Logikus kovetkezo pontok lehetnek:

- `6 / 300 / 0.05`
- `6 / 400 / 0.02`
- `6 / 800 / 0.02`
- `4 / 400 / 0.05`

Ezekkel mar jobban szet lehet valasztani:

- a rovidebb boosting hatasat
- a lassabb tanulas hatasat
- a sekelyebb fa hatasat

## Vegso Osszegzes

Az elso 3-fo-parameteres kor azt mutatta, hogy a `6 / 500 / 0.05` beallitas sem a `no_utility`, sem az `all_utility` modellnel nem javitott a sajat 30%-os baseline-hoz kepest: a train score nott, de a valid es test metrikak enyhen romlottak, ami arra utal, hogy a plusz fa ezen a ponton mar inkabb tulilleszkedest erositett.
