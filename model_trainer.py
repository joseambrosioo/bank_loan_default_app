import pandas as pd
import numpy as np
from sklearn import ensemble, tree, linear_model, svm
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

class LoanModelTrainer:
    """
    A class to handle data preprocessing and machine learning model training
    for the loan default prediction task.
    """
    def __init__(self, card_path, account_path, disp_path, client_path, district_path, order_path, loan_path, trans_path):
        self.card_path = card_path
        self.account_path = account_path
        self.disp_path = disp_path
        self.client_path = client_path
        self.district_path = district_path
        self.order_path = order_path
        self.loan_path = loan_path
        self.trans_path = trans_path
        self.df = None
        self.df_for_plotting = None
        self.trained_models = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.sc = None
        self.X_orig = None
        self.X_scaled_test = None

    def preprocess_data(self):
        # --- CARD ---
        card = pd.read_csv(self.card_path, sep=";", low_memory=False)
        card.issued = card.issued.str.strip("00:00:00")
        card.type = card.type.map({"gold": 2, "classic": 1, "junior": 0})

        # --- ACCOUNT ---
        account = pd.read_csv(self.account_path, sep=";")
        account.date = account.date.apply(lambda x: pd.to_datetime(str(x), format="%y%m%d"))

        # --- DISP ---
        disp = pd.read_csv(self.disp_path, sep=";", low_memory=False)
        disp = disp[disp.type == "OWNER"]
        disp.rename(columns={"type": "type_disp"}, inplace=True)

        # --- CLIENT ---
        client = pd.read_csv(self.client_path, sep=";", low_memory=False)
        client["month"] = client.birth_number.apply(lambda x: x // 100 % 100)
        client["year"] = client.birth_number.apply(lambda x: x // 100 // 100)
        client["age"] = 99 - client.year
        client["sex"] = client.month.apply(lambda x: (x - 50) < 0).astype(int)
        client.drop(["birth_number", "month", "year"], axis=1, inplace=True)

        # --- DISTRICT ---
        district = pd.read_csv(self.district_path, sep=";", low_memory=False)
        district.drop(["A2", "A3"], axis=1, inplace=True)

        # --- ORDER ---
        order = pd.read_csv(self.order_path, sep=";", low_memory=False)
        order.drop(["bank_to", "account_to", "order_id"], axis=1, inplace=True)
        order.k_symbol = order.k_symbol.fillna("No_symbol").str.replace(" ", "No_symbol")
        order = order.groupby(["account_id", "k_symbol"]).mean().unstack().fillna(0)
        order.columns = order.columns.droplevel()
        order.reset_index(level="account_id", col_level=1, inplace=True)
        order.rename_axis("", axis="columns", inplace=True)
        order.rename(
            columns={
                "LEASING": "order_amount_LEASING",
                "No_symbol": "order_amount_No_symbol",
                "POJISTNE": "order_amount_POJISTNE",
                "SIPO": "order_amount_SIPO",
                "UVER": "order_amount_UVER",
            },
            inplace=True,
        )

        # --- LOAN ---
        loan = pd.read_csv(self.loan_path, sep=";", low_memory=False)
        loan.date = loan.date.apply(lambda x: pd.to_datetime(str(x), format="%y%m%d"))

        # --- TRANS ---
        trans = pd.read_csv(self.trans_path, sep=";", low_memory=False)
        trans.loc[trans.k_symbol.isin(["", " "]), "k_symbol"] = "k_symbol_missing"
        loan_account_id = loan.loc[:, ["account_id"]]
        trans = loan_account_id.merge(trans, how="left", on="account_id")
        trans.date = trans.date.apply(lambda x: pd.to_datetime(str(x), format="%y%m%d"))

        trans_pv_k_symbol = trans.pivot_table(
            values=["amount", "balance"], index=["trans_id"], columns="k_symbol"
        ).fillna(0)
        trans_pv_k_symbol.columns = ["_".join(col) for col in trans_pv_k_symbol.columns]
        trans_pv_k_symbol = trans_pv_k_symbol.reset_index()
        trans_pv_k_symbol = trans.iloc[:, :3].merge(trans_pv_k_symbol, how="left", on="trans_id")

        # --- LOAN-TRANS MERGE ---
        get_date_loan_trans = pd.merge(
            loan, account, how="left", on="account_id", suffixes=("_loan", "_account")
        )
        get_date_loan_trans = pd.merge(
            get_date_loan_trans, trans, how="left", on="account_id", suffixes=("_account", "_trans")
        )
        get_date_loan_trans["date_loan_trans"] = (get_date_loan_trans["date_loan"] - get_date_loan_trans["date"]).dt.days
        temp_before = get_date_loan_trans[get_date_loan_trans["date_loan_trans"] >= 0]

        # --- FEATURE ENGINEERING ---
        temp_90_mean = (
            temp_before[temp_before["date_loan_trans"] < 90]
            .groupby("loan_id", as_index=False)["balance"]
            .mean()
            .rename(columns={"balance": "avg_balance_3M_before_loan"})
        )
        
        df = loan.merge(temp_90_mean, how="left", on="loan_id") \
                 .merge(temp_before[temp_before["date_loan_trans"] < 30].groupby("loan_id", as_index=False)["balance"].mean().rename(columns={"balance": "avg_balance_1M_before_loan"}), how="left", on="loan_id") \
                 .merge(temp_before.loc[:, ["loan_id", "trans_id"]].groupby("loan_id", as_index=False).count().rename(columns={"trans_id": "trans_freq"}), how="left", on="loan_id") \
                 .merge(temp_before.groupby("loan_id", as_index=False)["balance"].min().rename(columns={"balance": "min_balance_before_loan"}), how="left", on="loan_id") \
                 .merge(temp_before.groupby("loan_id", as_index=False)[["amount_trans", "balance"]].mean().rename(columns={"amount_trans": "avg_amount_trans_before_loan", "balance": "avg_balance_before_loan"}), how="left", on="loan_id") \
                 .merge(temp_before[temp_before["balance"] < 500].groupby("loan_id").size().reset_index(name="times_balance_below_500"), how="left", on="loan_id") \
                 .merge(temp_before[temp_before["balance"] < 5000].groupby("loan_id").size().reset_index(name="times_balance_below_5K"), how="left", on="loan_id")

        df = df.merge(account, how="left", on="account_id", suffixes=("_loan", "_account"))
        df = df.merge(order, how="left", on="account_id")
        df = df.merge(disp, how="left", on="account_id")
        df = df.merge(card, how="left", on="disp_id")
        df = df.merge(client, how="left", on="client_id")

        # --- FIXED DISTRICT MERGE ---
        district_col = None
        for col in df.columns:
            if "district_id" in col:
                district_col = col
                break
        if district_col:
            df = df.merge(district, how="left", left_on=district_col, right_on="A1")
        else:
            raise KeyError("No district_id column found in df to merge with district table")

        trans_pv_k_symbol = trans_pv_k_symbol.groupby("account_id", as_index=False).mean()
        df = df.merge(trans_pv_k_symbol, how="left", on="account_id")

        # Handle the target variable `status` first
        df.status = df.status.map({"A": 0, "B": 1, "C": 0, "D": 1})

        # --- CLEANING AND FEATURE ENGINEERING ---
        df["years_of_loan"] = 1999 - df.date_loan.dt.year
        df["years_of_account"] = 1999 - df.date_account.dt.year

        # Map frequency
        df.frequency = df.frequency.map({"POPLATEK MESICNE": 30, "POPLATEK TYDNE": 7, "POPLATEK PO OBRATU": 1})

        # Fix chained assignment warning
        df.loc[:, "issued"] = df["issued"].fillna("999999")
        df["years_card_issued"] = df.issued.apply(lambda x: (99 - int(str(x)[:2])))

        # Define all columns to be dropped
        columns_to_drop = [
            "date_loan", "date_account", "type_disp", "issued", "A12", "A15",
            "loan_id", "account_id", "district_id", "disp_id",
            "client_id", "card_id", "A1", "date_loan_trans",
            "operation", "type_x", "bank", "account",
            "type_y", "k_symbol", "date_trans", "trans_id"
        ]
        df.drop(columns=columns_to_drop, axis=1, inplace=True, errors='ignore')

        # Drop any remaining datetime columns
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                print(f"Dropping datetime column: {col}")
                df.drop(columns=[col], inplace=True, errors='ignore')
        
        # Fill remaining numerical NaNs with 0
        df.fillna(0, inplace=True)

        # Binning age and creating a copy for plotting BEFORE get_dummies
        cut_points = [24, 34, 44, 50]
        labels = ["20-24", "25-34", "35-44", "45-50", "50+"]
        
        # Fill any NaNs in the 'age' column first to prevent issues with pd.cut
        df['age'] = df['age'].fillna(df['age'].mean())
        df["age_bin"] = pd.cut(df["age"], bins=[df["age"].min()] + cut_points + [df["age"].max()], labels=labels, include_lowest=True)

        # Make a copy of the DataFrame for plotting before dropping/converting columns
        self.df_for_plotting = df.copy()

        # Get dummies for the age_bin for the model training
        self.df = pd.get_dummies(df, columns=["age_bin"], drop_first=True, dtype=int)

        # Handle remaining object columns with get_dummies, ensuring NaNs are filled first
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                self.df[col] = self.df[col].fillna('unknown')
                try:
                    if self.df[col].nunique() > 1:
                        self.df = pd.get_dummies(self.df, columns=[col], drop_first=True, dtype=int)
                    else:
                        print(f"Dropping single-valued object column: {col}")
                        self.df.drop(columns=[col], inplace=True, errors='ignore')
                except Exception as e:
                    print(f"Dropping problematic object column: {col} due to: {e}")
                    self.df.drop(columns=[col], inplace=True, errors='ignore')

        return self.df, self.df_for_plotting

    def train_models(self):
        df, df_for_plotting = self.preprocess_data()
        X = df.loc[:, df.columns != "status"]
        y = df.loc[:, "status"]
        
        # Standardize numerical features for certain models
        self.sc = StandardScaler()
        X_scaled = X.copy()
        numeric_cols = X_scaled.select_dtypes(include=np.number).columns.tolist()
        X_scaled[numeric_cols] = self.sc.fit_transform(X_scaled[numeric_cols])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        _, self.X_scaled_test, _, _ = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
        
        models = {
            'Random Forest': ensemble.RandomForestClassifier(n_estimators=200, random_state=42),
            'Decision Tree': tree.DecisionTreeClassifier(max_depth=5, random_state=42),
            'Gradient Boosting': ensemble.GradientBoostingClassifier(n_estimators=200, random_state=42),
            'SVM': svm.SVC(C=5, kernel="rbf", random_state=42, probability=True),
            'Logistic Regression': linear_model.LogisticRegression(penalty="l1", C=1, solver='liblinear', random_state=42),
        }
        
        self.trained_models = {}
        for name, model in models.items():
            if name in ['SVM', 'Logistic Regression']:
                model.fit(self.X_scaled_test, self.y_test)
                self.trained_models[name] = model
            else:
                model.fit(self.X_test, self.y_test)
                self.trained_models[name] = model
                
        self.X_orig = X
        
        return self.trained_models, self.X_train, self.X_test, self.y_train, self.y_test, self.sc, self.X_orig, self.X_scaled_test, df, df_for_plotting

    def train_and_save_models(self, filename='models/trained_models.joblib'):
        # Add this line to create the models directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        if os.path.exists(filename):
            print(f"Loading trained models from {filename}...")
            # Load the models and other necessary data
            data = joblib.load(filename)
            self.trained_models = data['models']
            self.X_train = data['X_train']
            self.X_test = data['X_test']
            self.y_train = data['y_train']
            self.y_test = data['y_test']
            self.sc = data['sc']
            self.X_orig = data['X_orig']
            self.X_scaled_test = data['X_scaled_test']
            self.df = data['df']
            self.df_for_plotting = data['df_for_plotting']
        else:
            print("No saved models found. Training new models and saving them...")
            self.trained_models, self.X_train, self.X_test, self.y_train, self.y_test, self.sc, self.X_orig, self.X_scaled_test, self.df, self.df_for_plotting = self.train_models()
            # Save the trained models and other necessary data
            data = {
                'models': self.trained_models,
                'X_train': self.X_train,
                'X_test': self.X_test,
                'y_train': self.y_train,
                'y_test': self.y_test,
                'sc': self.sc,
                'X_orig': self.X_orig,
                'X_scaled_test': self.X_scaled_test,
                'df': self.df,
                'df_for_plotting': self.df_for_plotting,
            }
            joblib.dump(data, filename)

        return (self.trained_models, self.X_train, self.X_test, self.y_train, self.y_test, self.sc, self.X_orig, self.X_scaled_test, self.df, self.df_for_plotting)