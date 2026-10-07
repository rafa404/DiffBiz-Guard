# Datasets

No raw or prepared records are distributed. Obtain data from the original providers and comply with their terms. The JSON files under `splits/` contain non-sensitive preparation and split metadata only.

| Dataset | Role | Source | Preparation and split |
|---|---|---|---|
| Bank Marketing | Internal | [Bank Marketing Dataset for Term Deposit Prediction](https://www.kaggle.com/datasets/nisargpatel344/bank-marketing-dataset-for-term-deposit-prediction) | Complete 100,000-row table; `ClientID` and four leakage candidates excluded; 69,999/15,001/15,000 train/validation/test. |
| Bank Account Fraud | Internal | [Bank Account Fraud Dataset, NeurIPS 2022](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) | First 120,000 rows of `Variant V.csv`; stratified 84,000/18,000/18,000 split using seeds 2026 and 2027. |
| Large Customer Churn | Internal | [Customer Churn Prediction Dataset 1M](https://www.kaggle.com/datasets/isandeep06/customer-churn-prediction-dataset-1m) | Preserved 120,000-row extract; `customer_id` removed and `signup_date` expanded; 84,000/18,000/18,000 split. |
| Home Credit Default Risk | External transfer | [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) | `application_train.csv` only; independent 70/15/15 splits using seeds 2026, 2027, and 2028. |
| Airline Passenger Satisfaction | Breadth | [Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction) | Supplied source files concatenated and stratified to 120,000 rows; 84,000/18,000/18,000 split. |
| Hotel Reservations | Breadth | [Hotel Reservations Classification](https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset) | 10,275 exact duplicate cleaned rows removed; 26,000-row modelling table; 18,200/3,900/3,900 split. |
| DataCo Supply Chain | Breadth | [DataCo Smart Supply Chain](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) | Stratified sample to 120,000 rows; 84,000/18,000/18,000 split. |

## Lineage boundaries

- The preserved Large Customer Churn lineage begins at the audited 120,000-row extract. The earlier operator that reduced the upstream benchmark is unavailable and is not inferred.
- Source and split hashes are preserved where they were present in the final artifacts. Airline, Hotel, and DataCo hashes were not available and remain explicitly `null`.
- Home Credit competition rows are not redistributed. `SK_ID_CURR` was retained only for split tracking.
- The DataCo URL above is the corrected source used by this release.
