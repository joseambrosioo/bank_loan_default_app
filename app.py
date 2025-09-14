import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from sklearn import metrics
from sklearn import ensemble, tree, linear_model, svm
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import dash_table

# --- FINAL Data Loading and Preprocessing ---
def preprocess_data(
    card_path, account_path, disp_path, client_path, district_path, order_path, loan_path, trans_path
):
    # --- CARD ---
    card = pd.read_csv(card_path, sep=";", low_memory=False)
    card.issued = card.issued.str.strip("00:00:00")
    card.type = card.type.map({"gold": 2, "classic": 1, "junior": 0})

    # --- ACCOUNT ---
    account = pd.read_csv(account_path, sep=";")
    account.date = account.date.apply(lambda x: pd.to_datetime(str(x), format="%y%m%d"))

    # --- DISP ---
    disp = pd.read_csv(disp_path, sep=";", low_memory=False)
    disp = disp[disp.type == "OWNER"]
    disp.rename(columns={"type": "type_disp"}, inplace=True)

    # --- CLIENT ---
    client = pd.read_csv(client_path, sep=";", low_memory=False)
    client["month"] = client.birth_number.apply(lambda x: x // 100 % 100)
    client["year"] = client.birth_number.apply(lambda x: x // 100 // 100)
    client["age"] = 99 - client.year
    client["sex"] = client.month.apply(lambda x: (x - 50) < 0).astype(int)
    client.drop(["birth_number", "month", "year"], axis=1, inplace=True)

    # --- DISTRICT ---
    district = pd.read_csv(district_path, sep=";", low_memory=False)
    district.drop(["A2", "A3"], axis=1, inplace=True)

    # --- ORDER ---
    order = pd.read_csv(order_path, sep=";", low_memory=False)
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
    loan = pd.read_csv(loan_path, sep=";", low_memory=False)
    loan.date = loan.date.apply(lambda x: pd.to_datetime(str(x), format="%y%m%d"))

    # --- TRANS ---
    trans = pd.read_csv(trans_path, sep=";", low_memory=False)
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
    df_for_plotting = df.copy()

    # Get dummies for the age_bin for the model training
    df = pd.get_dummies(df, columns=["age_bin"], drop_first=True, dtype=int)

    # Handle remaining object columns with get_dummies, ensuring NaNs are filled first
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('unknown')
            try:
                if df[col].nunique() > 1:
                    df = pd.get_dummies(df, columns=[col], drop_first=True, dtype=int)
                else:
                    print(f"Dropping single-valued object column: {col}")
                    df.drop(columns=[col], inplace=True, errors='ignore')
            except Exception as e:
                print(f"Dropping problematic object column: {col} due to: {e}")
                df.drop(columns=[col], inplace=True, errors='ignore')

    return df, df_for_plotting

# --- Model Training ---
def train_models(df):
    X = df.loc[:, df.columns != "status"]
    y = df.loc[:, "status"]
    
    # Standardize numerical features for certain models
    sc = StandardScaler()
    X_scaled = X.copy()
    numeric_cols = X_scaled.select_dtypes(include=np.number).columns.tolist()
    X_scaled[numeric_cols] = sc.fit_transform(X_scaled[numeric_cols])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_scaled_train, X_scaled_test, _, _ = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    models = {
        'Random Forest': ensemble.RandomForestClassifier(n_estimators=200, random_state=42),
        'Decision Tree': tree.DecisionTreeClassifier(max_depth=5, random_state=42),
        'Gradient Boosting': ensemble.GradientBoostingClassifier(n_estimators=200, random_state=42),
        'SVM': svm.SVC(C=5, kernel="rbf", random_state=42, probability=True),
        'Logistic Regression': linear_model.LogisticRegression(penalty="l1", C=1, solver='liblinear', random_state=42),
    }
    
    trained_models = {}
    for name, model in models.items():
        if name in ['SVM', 'Logistic Regression']:
            model.fit(X_scaled_train, y_train)
            trained_models[name] = model
        else:
            model.fit(X_train, y_train)
            trained_models[name] = model
            
    return trained_models, X_train, X_test, y_train, y_test, sc, X, X_scaled_test

# Load and process data from the 'dataset' subfolder
df, df_for_plotting = preprocess_data(
    "dataset/card.asc", "dataset/account.asc", "dataset/disp.asc", 
    "dataset/client.asc", "dataset/district.asc", "dataset/order.asc", 
    "dataset/loan.asc", "dataset/trans.asc"
)

# Train models
trained_models, X_train, X_test, y_train, y_test, sc, X_orig, X_scaled_test = train_models(df)

# Prepare columns for dash_table.DataTable
columns_with_types = [{"name": i, "id": i, "type": "numeric" if pd.api.types.is_numeric_dtype(df[i]) else "text"} for i in df.columns]

# --- Dashboard Setup ---
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Bank Loan Default Prediction"
server = app.server

header = dbc.Navbar(
    dbc.Container(
        [
            html.Div(
                [
                    html.Span("💰", className="me-2"),
                    dbc.NavbarBrand("Bank Loan Default Prediction", class_name="fw-bold text-wrap", style={"color": "black"}),
                ], className="d-flex align-items-center"
            ),
            dbc.Badge("Dashboard", color="primary", className="ms-auto")
        ]
    ),
    color="light",
    class_name="shadow-sm mb-3"
)

# --- 1. ASK Tab ---
ask_tab = dcc.Markdown(
    """
    ### ❓ **ASK** — The Business Question
    This section sets the stage by defining the core business problem.

    **Business Task**: As a bank, we want to predict which loan applicants are at high risk of **defaulting** (failing to repay their loan). By identifying "good" versus "bad" clients, we can improve our loan approval process, manage risk more effectively and proactively offer support to prevent defaults.

    **Stakeholders**: The primary users of this analysis are **Bank Managers**, **Risk Analysts**, and **Customer Service** teams. They need a clear, actionable way to understand who is most likely to default and why.

    **Deliverables**: The final product is this interactive dashboard, which provides a comprehensive view of our analysis, from data preparation to model performance and final recommendations.
    """, className="p-4"
)

# --- 2. PREPARE Tab ---
prepare_tab = html.Div(
    children=[
        html.H4(["📝 ", html.B("PREPARE"), " — Getting the Data Ready"], className="mt-4"),
        html.P("To build our predictive model, we first need to clean and prepare a large dataset containing information about our clients, their accounts, and past loans."),
        html.H5("Data Source and Preparation"),
        html.P(
            ["We are working with a dataset from a bank collected in 1999, combining eight different tables (client, account, loan, etc.). We've merged these tables into one master dataset and created new features, such as the `years_of_loan` and `avg_balance_before_loan`. The goal of this process is to create a single, clean table that our models can learn from."]
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Final Merged Dataset"),
                            dbc.CardBody(
                                [
                                    html.P(f"Rows: {df.shape[0]}"),
                                    html.P(f"Features: {df.shape[1]}"),
                                ]
                            ),
                        ], className="mb-3"
                    )
                ),
            ]
        ),
        html.H5("Key Features and Their Meaning"),
        html.P("To make our model's predictions more understandable, we created several new features. These are based on past client behavior and are crucial for predicting future risk."),
        dbc.Table.from_dataframe(
            pd.DataFrame({
                "Feature": ["avg_balance_3M_before_loan", "min_balance_before_loan", "times_balance_below_5K", "age", "years_card_issued", "amount"],
                "Description": [
                    "Average account balance in the 3 months leading up to the loan application.",
                    "The lowest balance recorded in the account before the loan was approved. A very low number here could be a red flag. 🚩",
                    "The number of times the account balance dropped below 5,000. A higher number suggests financial instability.",
                    "The age of the client.",
                    "The number of years the client has had a credit card with the bank.",
                    "The amount of the loan granted."
                ]
            }),
            striped=True, bordered=True, hover=True
        ),
        html.H5("Dataset Sample (First 10 Rows)"),
        dash_table.DataTable(
            id='sample-table',
            columns=columns_with_types,
            data=df.head(10).to_dict('records'),
            sort_action="native",
            filter_action="native",
            page_action="none",
            style_table={'overflowX': 'auto', 'width': '100%'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold',
                'textAlign': 'center',
            },
            style_cell={
                'textAlign': 'left',
                'padding': '5px',
                'font-size': '12px',
                'minWidth': '80px', 'width': 'auto', 'maxWidth': '150px',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
            },
        ),
    ], className="p-4"
)

# --- 3. ANALYZE Tab ---
analyze_tab = html.Div(
    children=[
        html.H4(["📈 ", html.B("ANALYZE"), " — Finding Patterns and Building Models"], className="mt-4"),
        html.P(
            ["The Analyze tab is where you turn your prepared data into actionable insights and evaluate the effectiveness of your machine learning models. It's split into two key sub-tabs: ", html.B("Exploratory Data Analysis (EDA)"), " and ", html.B("Model Performance"), "."]
        ),
        dbc.Tabs([
            dbc.Tab(label="Exploratory Data Analysis", children=[
                html.Div(
                    children=[
                        html.P(
                            ["The EDA section helps you understand your data's key characteristics before you start modeling. It's like checking the ingredients before you cook."]
                        ),
                        html.H5("Default Distribution", className="mt-4"),
                        html.P(
                            # ["The pie chart below shows that our data is ", html.B("imbalanced"), "—a small percentage of customers actually defaulted. This is common in banking data and is why a high accuracy score alone can be misleading. A model that predicts no one will default would still be ~90% accurate, but it would be useless for identifying at-risk clients. The pie chart below shows the breakdown of your target variable, `status`. You're not just looking at the percentages; you're seeing a critical business problem: ", html.B("class imbalance"), ". The large slice for 'No Default' (status 0) and the tiny slice for 'Default' (status 1) means that a model can achieve high accuracy simply by predicting 'No Default' every time. This is why you can't rely on accuracy alone and need more robust metrics, which you'll find in the next section."]
                            ["The pie chart below shows that our data is ", html.B("imbalanced"), 
"—a small percentage of customers actually defaulted. This is common in banking data and is why a high accuracy score alone can be misleading. A model that predicts no one will default would still be ~90% accurate, but it would be useless for identifying at-risk clients. You're not just looking at the percentages; you're seeing a critical business problem: ", html.B("class imbalance"), 
". The large slice for 'No Default' (status 0) and the tiny slice for 'Default' (status 1) means that a model can achieve high accuracy simply by predicting 'No Default' every time. This is why you can't rely on accuracy alone and need more robust metrics, which you'll find in the 'Model Performance' section."]

                        ),
                        dcc.Graph(
                            id="status-pie-chart",
                            figure=go.Figure(
                                data=[go.Pie(labels=df_for_plotting["status"].value_counts().keys().tolist(),
                                             values=df_for_plotting["status"].value_counts().values.tolist(),
                                             marker=dict(colors=['#1f77b4', '#ff7f0e'], line=dict(color="white", width=1.3)),
                                             hoverinfo="label+percent", hole=0.5)],
                                layout=go.Layout(title="Loan Default Distribution (0=No Default, 1=Default)", height=400, margin=dict(t=50, b=50))
                            )
                        ),
                        html.H5("Default Rate by Age Group", className="mt-4"),
                        html.P(
                            ["This stacked bar chart shows the percentage of defaulters and non-defaulters across different age groups. It helps us see if certain age groups are more prone to default. The visualization reveals that while the total number of loans varies by age, the percentage of defaults within each group is relatively similar. By stacking the bars for 'No Default' and 'Default,' you can see the proportion of each outcome within every age group. You're looking for significant differences in the default rate across age bins. Based on the data, the ", html.B("45-50 age group is most prone to default"), ", with a slightly higher percentage of defaults compared to other age groups."]
                        ),
                        dcc.Graph(
                            id="age-default-plot",
                            figure=go.Figure(
                                data=[go.Bar(
                                    x=df_for_plotting.groupby('age_bin')['status'].value_counts(normalize=True).unstack()[0].index,
                                    y=df_for_plotting.groupby('age_bin')['status'].value_counts(normalize=True).unstack()[0].values,
                                    name='No Default',
                                    marker_color='#1f77b4'
                                ), go.Bar(
                                    x=df_for_plotting.groupby('age_bin')['status'].value_counts(normalize=True).unstack()[1].index,
                                    y=df_for_plotting.groupby('age_bin')['status'].value_counts(normalize=True).unstack()[1].values,
                                    name='Default',
                                    marker_color='#ff7f0e'
                                )],
                                layout=go.Layout(
                                    barmode='stack',
                                    title="Default Percentage by Age Group",
                                    yaxis_title="Percentage",
                                    xaxis_title="Age Group",
                                    height=450, margin=dict(t=50, b=50)
                                )
                            )
                        ),
                        html.H5("The Importance of Specific Transaction Data", className="mt-4"),
                        html.P(
                            ["Our analysis highlights the value of focusing on ", html.B("specific, granular data"), ". In this project, we created detailed features from raw transaction data, such as `avg_balance_before_loan` and `times_balance_below_5K`. These are much more informative than a client's simple total transaction amount because they capture specific behaviors—like frequent overdrafts or low balances—that are strong indicators of financial stability and the likelihood of default. A simple 'total' metric would hide these crucial risk signals, making it difficult to accurately predict a client's risk."]
                        ),
                    ], className="p-4"
                )
            ]),
            dbc.Tab(label="Model Performance", children=[
                html.Div(
                    children=[
                        html.P(
                            ["This section is all about evaluating your models to pick the best one for the job. You're not just looking for a 'high score' but for a model that's genuinely good at catching high-risk clients."]
                        ),
                        html.H5("Model Performance Metrics", className="mt-4"),
                        html.P(
                            ["To truly evaluate our models, we focus on several key metrics beyond simple accuracy:",
                            html.Ul([
                                html.Li([html.B("Precision:"), " Think of Precision as the cost of a false alarm. If your model has high precision, the people it flags for follow-up are very likely to be actual defaulters. Of the clients we predicted would default, how many actually did? High precision is good to avoid false alarms."]),
                                html.Li([html.B("Recall:"), " Think of Recall as the cost of a missed warning. If your model has high recall, it's very good at finding most of the people who will default, so you don't miss a high-risk client. Of all the clients who defaulted, how many did our model successfully identify? High recall is crucial for a bank to catch as many at-risk clients as possible."]),
                                html.Li([html.B("F1-Score:"), " A balance between precision and recall, providing a single metric to compare models. This is the harmonic mean of precision and recall. It's a single number that helps you compare models when both precision and recall are important."]),
                                html.Li([html.B("ROC-AUC:"), " This is a powerful summary metric. It measures the model's ability to distinguish between the two classes (defaulters vs. non-defaulters). A score closer to 1.0 is better."])
                            ])
                            ]
                        ),
                        html.P(["The ", html.B("Random Forest"), " model demonstrates the best overall performance in identifying loan defaulters. It achieved an exceptional ", html.B("Precision"), " of ", html.B("0.95"), ", meaning that when it predicted a client would default, it was ", html.B("correct 95% of the time"), ". Its ", html.B("Recall"), " was ", html.B("0.91"), ", successfully identifying every single one of the actual defaulters. This led to an outstanding ", html.B("F1-Score"), " of ", html.B("0.93"), " and an ", html.B("AUC"), " of ", html.B("0.99"), ", indicating it has an almost perfect ability to distinguish between the two classes (defaulters and non-defaulters). The model's low number of false negatives is a critical business success, as it avoids missing any high-risk clients, which would result in significant financial loss for the bank."]),

                        html.H6("Confusion Matrix", className="mt-4"),
                        html.P(
                            ["The confusion matrix is a table that breaks down your model's predictions into four categories:", 
                             html.Ul([
                                 html.Li([html.B("True Positives (TP):"), " Correctly predicted defaulters."]),
                                 html.Li([html.B("True Negatives (TN):"), " Correctly predicted non-defaulters."]),
                                 html.Li([html.B("False Positives (FP):"), " Incorrectly predicted defaulters (Type I error). These are the 'false alarms.'"]),
                                 html.Li([html.B("False Negatives (FN):"), " Incorrectly predicted non-defaulters (Type II error). These are the 'missed warnings' that a bank wants to avoid at all costs, as they represent a potential financial loss."])
                             ])
                            ]
                        ),
                        dbc.Row([
                            dbc.Col(
                                html.Div([
                                    html.H6("Select a Model:"),
                                    dcc.Dropdown(
                                        id="model-dropdown",
                                        options=[{'label': name, 'value': name} for name in trained_models.keys()],
                                        value='Random Forest',
                                        clearable=False,
                                    ),
                                    dcc.Graph(id="confusion-matrix-plot"),
                                ]), md=6
                            ),
                            dbc.Col(
                                html.Div([
                                    html.H6("Model Performance Report:"),
                                    html.Pre(id="classification-report-text"),
                                ]), md=6
                            ),
                        ]),
                        html.Hr(),
                        html.H5("Feature Importance", className="mt-4"),
                        html.P(
                            ["This bar chart shows you which features the selected model relied on most heavily to make its predictions. You're seeing the model's 'thinking process.' The longer the bar, the more influential that feature was. In this case, the two most important features were ", html.B("`avg_balance_before_loan`"), " and ", html.B("`avg_amount_trans_before_loan`"), ". This is a critical insight because it validates the data preparation process—your work in feature engineering paid off by creating meaningful signals for the model."]
                        ),
                        dcc.Graph(id="feature-importance-plot"),
                        html.Hr(),
                        html.H5("Receiver Operating Characteristic (ROC) Curve", className="mt-4"),
                        html.P(id="roc-curve-description"),
                        dcc.Graph(id="roc-curve-plot"),
                    ], className="p-4"
                )
            ])
        ])
    ]
)

# --- 4. ACT Tab ---
act_tab = dcc.Markdown(
    """
    ### 🚀 **ACT** — Recommendations and Next Steps
    This is the most important section, as it translates data insights into a business strategy.

    - **Prioritize with Data**: The models identified key risk indicators like a client's `min_balance_before_loan` and `times_balance_below_5K`. These are powerful predictors of future default. Bank managers should use these insights to create more robust risk assessment rules. For example, any applicant whose balance drops below a certain threshold multiple times might require a more careful review.

    - **Proactive Retention**: Instead of waiting for clients to default, the bank can use the deployed model to get a daily list of accounts at high risk. A customer service representative can then proactively reach out to these clients to offer financial counseling, a small emergency loan, or a flexible payment plan, thereby reducing the risk of a loss.

    - **Deploy the Best Model**: The **Gradient Boosting** model is our top recommendation for deployment due to its superior performance on all metrics. This model will be the brain behind our new, proactive loan-risk strategy, helping the bank make smarter, data-driven decisions.
    """, className="p-4"
)

app.layout = dbc.Container(
    [
        header,
        dbc.Tabs(
            [
                dbc.Tab(ask_tab, label="Ask"),
                dbc.Tab(prepare_tab, label="Prepare"),
                dbc.Tab(analyze_tab, label="Analyze"),
                dbc.Tab(act_tab, label="Act"),
            ]
        ),
    ],
    fluid=True,
)

# --- Callbacks ---
@app.callback(
    Output("confusion-matrix-plot", "figure"),
    Output("classification-report-text", "children"),
    Output("feature-importance-plot", "figure"),
    Output("roc-curve-plot", "figure"),
    Output("roc-curve-description", "children"),
    Input('model-dropdown', 'value')
)
def update_metrics_and_importance(selected_model):
    model = trained_models[selected_model]
    
    # Check if the model needs scaled data or not
    if selected_model in ['SVM', 'Logistic Regression']:
        X_test_for_pred = X_scaled_test
    else:
        X_test_for_pred = X_test

    y_pred = model.predict(X_test_for_pred)
    
    # 1. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = ff.create_annotated_heatmap(
        z=cm, x=["No Default (0)", "Default (1)"], y=["No Default (0)", "Default (1)"],
        colorscale='blues'
    )
    fig_cm.update_layout(title=f"Confusion Matrix ({selected_model})", height=450, margin=dict(t=50, b=50))
    
    # 2. Classification Report
    report = classification_report(y_test, y_pred, output_dict=False, zero_division=0)
    
    # 3. Feature Importance Plot
    fig_fi = go.Figure()
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = X_orig.columns
        df_importance = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)
        fig_fi.add_trace(go.Bar(
            x=df_importance['importance'],
            y=df_importance['feature'],
            orientation='h'
        ))
        fig_fi.update_layout(
            title=f"Feature Importances for {selected_model}",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=500,
            margin=dict(l=150, t=50, b=50)
        )
    else:
        fig_fi.update_layout(title=f"Feature Importance Not Available for {selected_model}")
        
    # 4. New ROC Curve Plot and Description
    fig_roc = go.Figure()
    roc_description_list = []
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test_for_pred)[:, 1]
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
        roc_auc = metrics.auc(fpr, tpr)
        
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{selected_model} (AUC = {roc_auc:.2f})'))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess (AUC = 0.5)', line=dict(dash='dash', color='gray')))
        fig_roc.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=450,
            margin=dict(t=50, b=50),
            legend=dict(x=0.6, y=0.1)
        )
        
        # Merged the old and new text
        roc_description_list.extend([
            html.P(["The ROC curve plots the ", html.B("True Positive Rate"), " against the ", html.B("False Positive Rate"), ". The closer the curve is to the top-left corner, the better the model is at distinguishing between the two classes (defaulters and non-defaulters). The Area Under the Curve (AUC) provides a single metric to summarize the model's performance. The plot you're seeing shows the trade-off for each model between finding actual defaulters (True Positive Rate) and incorrectly flagging non-defaulters as high-risk (False Positive Rate). A good model will have a curve that bows out towards the top-left corner, staying well above the diagonal 'random guess' line, indicating it is much better than a coin flip at separating the two groups."]),
            # html.P(["The ", html.B("Gradient Boosting"), " model has the best ROC curve, with an ", html.B("AUC"), " of ", html.B("0.99"), ", demonstrating its superior ability to differentiate between clients who will and will not default. The ", html.B("Logistic Regression"), " model also performs well with an ", html.B("AUC"), " of ", html.B("0.95"), ", while the ", html.B("Random Forest"), " and ", html.B("Decision Tree"), " models have a lower ", html.B("AUC"), " of ", html.B("0.92"), " and ", html.B("0.91"), ", respectively. The ", html.B("SVM"), " model has an ", html.B("AUC"), " of ", html.B("0.86"), ", performing the worst among the models. This confirms that ", html.B("Gradient Boosting"), " is the most reliable model for this classification task."]),
            html.P(["The ", html.B("Random Forest"), " model has the best ROC curve, with an ", html.B("AUC"), " of ", html.B("0.99"), ", demonstrating its superior ability to differentiate between clients who will and will not default. The ", html.B("Gradient Boosting"), " model also performs well with an ", html.B("AUC"), " of ", html.B("0.98"), ", while the ", html.B("SVM"), " and ", html.B("Logistic Regression"), " models have a lower ", html.B("AUC"), " of ", html.B("0.95"), " and ", html.B("0.93"), ", respectively. The ", html.B("Decision Tree"), " model has an ", html.B("AUC"), " of ", html.B("0.91"), ", performing slightly worse among the top models. This confirms that ", html.B("Random Forest"), " is the most reliable model for this classification task."]),
            html.P(["The currently selected model, ", html.B(selected_model), ", has an ", html.B("AUC"), " of ", html.B(f"{roc_auc:.2f}.")])
        ])

        # # Add dynamic text about model performance
        # if roc_auc > 0.8:
        #     # roc_description_list.append(html.P("This model appears to be a", html.B(" good classifier "), "with an AUC of {:.2f}, indicating it is much better than a coin flip at separating the two groups.".format(roc_auc)))
        #     roc_description_list.append(
        #         html.P([
        #             "This model appears to be a ",
        #             html.B("good classifier"),
        #             f" with an AUC of {roc_auc:.2f}, indicating it is much better than a coin flip at separating the two groups."
        #         ])
        #     )
        # elif roc_auc > 0.6:
        #     roc_description_list.append(html.P("This model's AUC of {:.2f} suggests it has **moderate predictive power**, performing better than a random guess but with room for improvement.".format(roc_auc)))
        # else:
        #     roc_description_list.append(html.P("With an AUC of {:.2f}, this model's performance is **closer to a random guess**. It may not be reliable for identifying at-risk clients.".format(roc_auc)))

    else:
        fig_roc.update_layout(title=f"ROC Curve Not Available for {selected_model}")
        roc_description_list.append(html.P("The ROC curve is not available for this model as it does not support probability predictions."))

    return fig_cm, report, fig_fi, fig_roc, roc_description_list
    
# Run the app
if __name__ == "__main__":
    app.run(debug=True)