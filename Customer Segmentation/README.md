# Customer Segmentation (K-Means)

Groups credit-card customers into behavioural segments from their purchase, payment and cash-advance activity, so marketing can target each group differently.

## Dataset
[Kaggle – Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata): `CC GENERAL.csv`, **8,950 customers × 18 columns**, about 6 months of usage per customer.

| Group | Columns |
|---|---|
| Balance and limit | `BALANCE`, `BALANCE_FREQUENCY`, `CREDIT_LIMIT` |
| Purchases | `PURCHASES`, `ONEOFF_PURCHASES`, `INSTALLMENTS_PURCHASES`, `PURCHASES_TRX` and their frequencies |
| Cash advance | `CASH_ADVANCE`, `CASH_ADVANCE_FREQUENCY`, `CASH_ADVANCE_TRX` |
| Payments | `PAYMENTS`, `MINIMUM_PAYMENTS`, `PRC_FULL_PAYMENT` |
| Other | `TENURE`, `CUST_ID` (identifier, dropped) |

Data quality: no duplicate rows; missing values only in `MINIMUM_PAYMENTS` (313) and `CREDIT_LIMIT` (1).

## Workflow
1. **Inspect:** shape, dtypes, duplicates, nulls, summary statistics.
2. **Preprocess:** fill the 2 columns with missing values, drop `CUST_ID`.
3. **EDA:** histograms, boxplots and a correlation heatmap for all 17 features.
4. **Scale:** `StandardScaler`.
5. **Choose k:** elbow method (WCSS for k = 1–19).
6. **Model:** K-Means (`k-means++`, k = 8, `random_state=42`).
7. **Interpret:** append cluster labels and compare the mean of every feature per cluster.

## Results
With k = 8 the clusters range from 82 to about 2,600 customers (silhouette ≈ 0.21 when re-run on the full data). The segments separate into recognisable types:
- a very small group of **big spenders** (average purchases above 16,000)
- **frequent buyers** with high purchase frequency (~0.9+)
- **high-balance, low-spend** customers (average balance 4,500+, purchase frequency ~0.2)
- **low-balance, low-activity** customers

## Limitations and improvements
- Missing values filled with the **mean**, which is skewed by outliers; the median is safer.
- No log transform, although most money columns are heavily right-skewed.
- k = 8 is fixed without checking silhouette or Davies-Bouldin; the elbow is not clear-cut.
- No sklearn `Pipeline` and no saved model, so new customers can't be scored directly.

## Run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
# put CC GENERAL.csv in data/
jupyter notebook Customer_Segmentation.ipynb
```
