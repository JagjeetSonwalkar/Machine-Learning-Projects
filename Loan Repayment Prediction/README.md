# Loan Repayment Prediction (Ensemble Learning)

Predicts whether a loan applicant is likely to **not fully repay** (`not.fully.paid`), so a bank can decide whether to approve the loan. Decision Tree, Bagging, AdaBoost, Random Forest and Gradient Boosting are compared.

## Dataset
`data/loan_data.csv` – 9,578 loans, 14 columns, no missing values (LendingClub-style data).

| Column | Meaning |
|---|---|
| `credit.policy` | 1 if the customer meets the credit underwriting criteria |
| `purpose` | Loan purpose (7 categories) |
| `int.rate` | Interest rate |
| `installment` | Monthly installment |
| `log.annual.inc` | Log of annual income |
| `dti` | Debt-to-income ratio |
| `fico` | FICO credit score |
| `days.with.cr.line`, `revol.bal`, `revol.util` | Credit-line age, revolving balance, utilisation |
| `inq.last.6mths`, `delinq.2yrs`, `pub.rec` | Recent inquiries, delinquencies, public records |
| **`not.fully.paid`** | **Target** (1 = not fully repaid) |

The target is imbalanced: about 84% repaid vs 16% not repaid.

## Workflow
1. Load data and check types and nulls.
2. Label-encode `purpose`.
3. EDA: FICO vs `credit.policy` and repayment, purpose counts, FICO vs interest rate, correlation heatmap.
4. 70/30 train–test split (`random_state=101`).
5. Models, tuned with 5-fold `StratifiedKFold`:
   - Decision Tree (GridSearch on `max_depth`)
   - Bagging (100 trees)
   - AdaBoost (Decision Tree and Random Forest base)
   - Random Forest (600 trees)
   - Gradient Boosting (learning rate 0.05)

## Results (test set, 2,874 rows)
| Model | Accuracy | Recall (not repaid) |
|---|---|---|
| Decision Tree (depth 2) | 84.6% | 0.00 |
| Bagging (CV mean) | 73.1% | – |
| Random Forest | ~84.7% | 0.02 |
| AdaBoost + Random Forest | ~84.8% | 0.02 |
| Gradient Boosting | ~84.4% | 0.02 |

**Key takeaway:** every model scores about 85%, which is exactly what you get by predicting "repaid" for everyone (2,431 / 2,874 = 84.6%). The models almost never catch the defaulters (recall of about 2%), so accuracy is misleading here. Random Forest is called "best" only by a tiny margin.

## Known limitations and next steps
- **Class imbalance is not handled.** Try `class_weight="balanced"`, SMOTE or threshold tuning.
- **Wrong metric.** `recall_weighted` equals accuracy. Use ROC-AUC, PR-AUC, or recall/F1 on the positive class.
- `purpose` is label-encoded although it has no order. One-hot encoding is better.
- Preprocessing is not in a `Pipeline`, and the scaler is fitted on the full data before cross-validation (mild leakage).
- No model is saved yet.

## Run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook Loan_Repayment_Prediction.ipynb
```
