# FraudDetector
A two-stage anomaly detection system combining Isolation Forest (unsupervised outlier detection) with decision tree logic (human-interpretable rules) to identify fraudulent credit card transactions in real-time.

- Required Python Packages:
  ```python
  pandas
  numpy
  scikit-learn
  kagglehub
  ```
## Installation
Install Required Python Packages
```python
python -m pip install pandas numpy scikit-learn kagglehub
```
## Feature Extraction
Each transaction is represented as a multidimensional feature vector capturing patterns
### Features
1. Temporal Features: Cyclical encoding of hour and day of week using sine/cosine transformations to capture 24-hour and weekly spending patterns
2. Amount Features: Raw and log-normalized transaction amounts to handle skewed distributions
3. User-Relative Features: Z-score deviation from user's historical spending average; hours elapsed since last transaction
4. Merchant Risk: Binary flag for high-risk categories (online shopping)
### User Profiling
The system maintains a rolling user profile (lookback window = 100 transactions) that dynamically updates mean and standard deviation of spending amounts, enabling personalized anomaly detection that adapts to changes in user behavior.

## Design
#### Two Stage Approach
1. Isolation Forest provides probabilistic anomaly scoring (0.0–1.0 confidence) based on learned feature distributions
2. Decision Tree Logic applies business rules and risk thresholds, making the final verdict interpretable to stakeholders

This hybrid approach balances statistical rigor with explainability which critical for financial institutions where flagged transactions must justify intervention.

### Why Isolation Forest
- No fraud examples needed: Fraud detection datasets are typically imbalanced (fraud is rare). Isolation Forest learns what "normal" looks like from historical data without requiring labeled fraud cases
- Handles unusual spending patterns: Works well when users have different spending profiles (some spend $100/day, others $1,000/day). Doesn't assume everyone's behavior follows the same pattern

### Why Decision Tree
- Interpretability: Non-technical employees  can understand why a transaction was flagged
- Safety rails: Hard rules (e.g., $20k limit) enforce business constraints regardless of model confidence
- Velocity checks: Rapid successive transactions are often fraudulent, easy to detect with simple thresholds

