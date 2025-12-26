"""
Sequential Transaction Fraud Detector
Uses Isolation Forest and then goes through Decision Tree for Final verdict
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


class TransactionAnomalyDetector:
    def __init__(self, contamination=0.01, lookback_window=100):
        self.contamination = contamination
        self.lookback_window = lookback_window

        #Unsupervised Isolation Forest: Searches for outlier data points
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            n_jobs=-1
        )
        self.scaler = StandardScaler()

        #Creates User Profiles to store Spending Habits
        self.user_profiles = defaultdict(lambda: {
            'transaction_history': [],
            'avg_amount': 0,
            'std_amount': 1,
            'transaction_count': 0,
            'last_transaction_time': None
        })
        self.is_trained = False

    def extract_features(self, transaction):
        features = {}

        #Temporal Features: Cos and sin useful for hours and days
        trans_time = pd.to_datetime(transaction['trans_date_trans_time'])
        features['hour'] = trans_time.hour
        features['hour_sin'] = np.sin(2 * np.pi * trans_time.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * trans_time.hour / 24)
        features['dow_sin'] = np.sin(2 * np.pi * trans_time.dayofweek / 7)
        features['dow_cos'] = np.cos(2 * np.pi * trans_time.dayofweek / 7)

        # Amount Features
        amount = float(transaction['amt'])
        features['amount'] = amount
        features['log_amount'] = np.log1p(amount)

        # Context/Risk Features
        risky_categories = ['shopping_net', 'misc_net', 'grocery_pos', 'gas_transport']
        features['merchant_risk'] = 1 if transaction['category'] in risky_categories else 0

        # User Profile Features
        user_id = transaction['cc_num']
        profile = self.user_profiles[user_id]
        # Calculates Z-Score
        if profile['transaction_count'] > 0:
            features['amount_deviation'] = (amount - profile['avg_amount']) / (profile['std_amount'] + 1e-6)

            if profile['last_transaction_time']:
                hours_since = (trans_time - profile['last_transaction_time']).total_seconds() / 3600
                features['hours_since_last'] = hours_since
            else:
                features['hours_since_last'] = 0
        else:
            features['amount_deviation'] = 0
            features['hours_since_last'] = 0

        return features

    def update_user_profile(self, transaction, features):
        user_id = transaction['cc_num']
        profile = self.user_profiles[user_id]

        profile['transaction_history'].append(features['amount'])
        if len(profile['transaction_history']) > self.lookback_window:
            profile['transaction_history'].pop(0)

        # Recalculation of User Statistics
        vals = profile['transaction_history']
        profile['avg_amount'] = np.mean(vals)
        profile['std_amount'] = np.std(vals) if len(vals) > 1 else 1
        profile['transaction_count'] += 1
        profile['last_transaction_time'] = pd.to_datetime(transaction['trans_date_trans_time'])

    def train(self, transactions_df):
        print(f"Training Isolation Forest on {len(transactions_df)} records...")
        # Views Historical data to figure out what normal transactions are

        feature_matrix = []
        for _, row in transactions_df.iterrows():
            feats = self.extract_features(row)
            #Building User Profile History
            self.update_user_profile(row, feats)
            feature_matrix.append([
                feats['hour_sin'], feats['hour_cos'],
                feats['log_amount'], feats['merchant_risk'],
                feats['amount_deviation']
            ])

        X = np.array(feature_matrix)
        X_scaled = self.scaler.fit_transform(X)
        self.isolation_forest.fit(X_scaled)
        self.is_trained = True
        print("Training complete.")

    def _apply_decision_tree_logic(self, if_score, features):
        # Part 2: The Verdict - Uses decision tress based on isolation forest models confidence calculation
        # Only will go through if confidence is above 0.50 unless it breaches hard $20,000 limit
        reason = []
        risk_level = "LOW"
        is_anomaly = False

        # High Model Confidence
        if if_score > 0.75:

            if features['amount'] > 1000:
                risk_level = "CRITICAL"
                reason.append("High Model Confidence + Amount > $1k")
                is_anomaly = True

            elif features['amount_deviation'] > 3:
                risk_level = "CRITICAL"
                reason.append("High Model Confidence + Unusual Spending for User")
                is_anomaly = True

            elif features['merchant_risk'] == 1:
                risk_level = "HIGH"
                reason.append("High Model Confidence + Risky Merchant")
                is_anomaly = True

            else:
                risk_level = "MEDIUM"
                reason.append("Possible Spam - High Model Confidence")
                is_anomaly = True

        elif if_score > 0.50:

            if features['hours_since_last'] < 0.1 and features['amount'] > 100:
                risk_level = "HIGH"
                reason.append("Rapid Velocity Transaction")
                is_anomaly = True

            elif (features['hour'] < 5) and (features['amount'] > 500):
                risk_level = "MEDIUM"
                reason.append("High Value Late Night")
                is_anomaly = True

            else:
                risk_level = "LOW"

        else:
            if features['amount'] > 20000:
                risk_level = "CRITICAL"
                reason.append("Hard Rule: Amount > $20k")
                is_anomaly = True

            else:
                risk_level = "LOW"

        return is_anomaly, risk_level, ", ".join(reason)

    def score_transaction(self, transaction):
        features = self.extract_features(transaction)

        feature_vector = np.array([[
            features['hour_sin'], features['hour_cos'],
            features['log_amount'], features['merchant_risk'],
            features['amount_deviation']
        ]])

        # Isolation Forest Stage
        if self.is_trained:
            feature_vector_scaled = self.scaler.transform(feature_vector)
            raw_score = self.isolation_forest.decision_function(feature_vector_scaled)[0]

            # Converts the raw model output into a probability score between 0.0 (Safe) and 1.0 (Fruad)
            if_confidence = 1 / (1 + np.exp(20 * raw_score))
        else:
            if_confidence = 0.0

        # Decision Tree Stage
        is_anomaly, risk_level, reason = self._apply_decision_tree_logic(if_confidence, features)

        result = {
            'id': transaction.get('trans_num', 'N/A'),
            'amount': features['amount'],
            'model_confidence': round(if_confidence, 4),
            'risk_level': risk_level,
            'is_anomaly': is_anomaly,
            'reason': reason
        }

        # Updates Profile for next transaction
        self.update_user_profile(transaction, features)
        return result

    def process_stream(self, stream_df):
        print(f"\nProcessing stream of {len(stream_df)} transactions...")
        anomalies = []
        # Loop looks at one transaction at a time
        for idx, (_, row) in enumerate(stream_df.iterrows()):
            res = self.score_transaction(row)
            if res['is_anomaly']:
                anomalies.append(res)
        return pd.DataFrame(anomalies)


def load_data(sample_size=10000):
    # Kaggle data loads in
    try:
        import kagglehub
        import os
        path = kagglehub.dataset_download("ealtman2019/credit-card-transactions")

        #Looks in download path for CSV file
        csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("No Kaggle path or CSV file found")

        #Loads CSV into Pandas dataframe
        df = pd.read_csv(os.path.join(path, csv_files[0]))

        #Combines Month,day, and time columns into Single string
        #Then converts String into Python datetime object
        df['trans_date_trans_time'] = pd.to_datetime(
            df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-' +
            df['Day'].astype(str) + ' ' + df['Time'].astype(str)
        )
        #Cleans Amount column removing $ and converting to float
        df['amt'] = df['Amount'].astype(str).str.replace('$', '', regex=False).astype(float)
        #Converts Kaggle columns to our data names
        df = df.rename(columns={'Card': 'cc_num', 'MCC': 'category', 'Is Fraud?': 'is_fraud'})

        if sample_size:
            df = df.head(sample_size)

        #Sorts data by time
        return df.sort_values('trans_date_trans_time')
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    df = load_data(sample_size=20000)
    if df is None: return

    #70% of data is history to train model
    split = int(len(df) * 0.7)
    train_df = df.iloc[:split]
    stream_df = df.iloc[split:]

    detector = TransactionAnomalyDetector(contamination=0.02)
    detector.train(train_df)
    results = detector.process_stream(stream_df)

    print("\n" + "=" * 50)
    print(f"ANOMALY REPORT ({len(results)} detected)")
    print("=" * 50)

    if not results.empty:
        print(results[['amount', 'model_confidence', 'risk_level', 'reason']].head(10).to_string())


if __name__ == "__main__":
    main()