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
### Two Stage Approach
1. Isolation Forest provides probabilistic anomaly scoring (0.0–1.0 confidence) based on learned feature distributions
2. Decision Tree Logic applies business rules and risk thresholds, making the final verdict interpretable to stakeholders

This hybrid approach balances statistical rigor with explainability which critical for financial institutions where flagged transactions must justify intervention.

#### Why Isolation Forest
- No fraud examples needed: Fraud detection datasets are typically imbalanced (fraud is rare). Isolation Forest learns what "normal" looks like from historical data without requiring labeled fraud cases
- Handles unusual spending patterns: Works well when users have different spending profiles (some spend $100/day, others $1,000/day). Doesn't assume everyone's behavior follows the same pattern

#### Why Decision Tree
- Interpretability: Non-technical employees  can understand why a transaction was flagged
- Safety rails: Hard rules (e.g., $20k limit) enforce business constraints regardless of model confidence
- Velocity checks: Rapid successive transactions are often fraudulent, easy to detect with simple thresholds

## Real World Evolution Path
### 1. Improved Feature Engineering
For Production add more features including
1. Device & Network: Does the IP address match the cardholder's location? Are they using a new device?
2. Geographic Inconsistencies: Card used in New York at 2 PM, then Miami at 3 PM
3. Social Graph: Is the merchant a known fraud site? Does it share common customers with other fraud cases?

#### Why are these important
1. Fraudsters will often spend in different locations than the User's hometown and can use a different device
2. A $200 charge is fine at Target however the same $200 on a random cryptocurrency exchange at 3 AM is suspicious

#### Implementation
1. Partner with payment processor to access device IDs, IP geolocation, VPN detection
2. Integrate merchant blacklists (known scam sites, phishing URLs)
3. Build a graph database of merchant to fraud case relationships

### 2. Personalized Thresholds
Currently thresholds are the same for all users. Thresholds should vary
1. Segment users by risk profile: students (low spend), business owners (high spend), elderly (less likely using online purchases)
2. Merchant-specific rules: Stricter rules for high-risk merchants (casinos, wire transfers, cryptocurrency)
3. Time-based rules: Different thresholds for big holidays such as Christmas or Black Friday vs a regular day

#### Why these are important
1. $3000 for a buisness is nothing but for a student, this could be their entire account so it's important the transaction is stopped
2. Elderly people are generally not spending much online compared to the average middle aged person
3. Should not stop big transactions on big holidays as could result in users getting frustrated that their transaction is not going through

#### Implementation
1. Segment cardholder database into 10–20 risk cohorts (age, income, usage patterns, history)
2. Calibrate thresholds separately for each cohort using past dispute data
3. Adjust thresholds monthly based on business metrics (fraud loss, customer complaints, operational cost)

### 3. Real Time Processing
Currrently, trains once on historical data and processes transactions one at a time in batch. For Production
1. Subsecond decision latency: Transactions must be approved/declined before the customer leaves the register (< 200ms)
2. Continuous learning: Model should update hourly or daily as new fraud patterns emerge, not quarterly
3. High throughput: Should process 1,000s of transactions per second

#### Implementation
1. Deploy model to stream processing infrastructure such as AWS Kinesis instead of batch jobs
2. Pre-compute user profiles in a fast cache like Redis so lookups are instant and background jobs update cache every minute
3. Use model serving infrastructure  like TensorFlow Serving that keeps copies of current model in memory
4. Set up automated retraining: Daily scheduled jobs that train new models on yesterday's transactions





