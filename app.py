import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from sklearn import metrics
from sklearn import ensemble, tree, linear_model, svm
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import dash_table
import joblib
from fpdf import FPDF
import io
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score
import urllib.parse
from dash import callback_context

# Import the new model training class
from model_trainer import LoanModelTrainer

# --- Data Loading and Model Training ---
trainer = LoanModelTrainer(
    "dataset/card.asc", "dataset/account.asc", "dataset/disp.asc", 
    "dataset/client.asc", "dataset/district.asc", "dataset/order.asc", 
    "dataset/loan.asc", "dataset/trans.asc"
)
(trained_models, X_train, X_test, y_train, y_test, sc, X_orig, X_scaled_test, df, df_for_plotting) = trainer.train_and_save_models()

# Prepare columns for dash_table.DataTable
columns_with_types = [{"name": i, "id": i, "type": "numeric" if pd.api.types.is_numeric_dtype(df[i]) else "text"} for i in df.columns]

# --- Dashboard Configuration ---
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
            dbc.Badge("DS/ML App", color="info", className="ms-auto")
        ]
    ),
    color="light",
    class_name="shadow-sm mb-3"
)

# --- 1. ASK Tab ---
ask_tab = dcc.Markdown(
    """
    ### ❓ **ASK** — The Business Question
    This section defines the core business problem.

    **Business Task**: As a bank, we want to predict which loan applicants have a high risk of **defaulting** (failing to repay the loan). By identifying "good" versus "bad" customers, we can improve our loan approval process, manage risk more effectively, and offer proactive support to prevent defaults.

    **Stakeholders**: The primary users of this analysis are **Bank Managers**, **Risk Analysts**, and **Customer Service** teams. They need a clear and actionable way to understand who is most likely to default and why.

    **Deliverables**: The final product is an **interactive Machine Learning application** that provides a comprehensive, end-to-end view of our analytical pipeline—from multi-table data integration and feature engineering to rigorous model evaluation and strategic business recommendations.
    """, className="p-4"
)

# 1. Automate the Feature List logic
# This creates the data for all 55 features directly from your dataframe
full_feature_list = pd.DataFrame({
    "Feature Name": df.columns,
    "Data Type": [str(df[col].dtype) for col in df.columns],
    "Example Value": [df[col].iloc[0] for col in df.columns],
    "Business Category": [
        "Risk Target" if col == "status" else 
        "Account Detail" if any(x in col for x in ["balance", "amount"]) else 
        "Transaction Pattern" if any(x in col for x in ["total", "times", "trans"]) else 
        "Demographic" for col in df.columns
    ]
})

# --- 2. PREPARE Tab ---
prepare_tab = html.Div(
    children=[
        html.H4(["📝 ", html.B("PREPARE"), " — Data Integration & Feature Engineering"], className="mt-4"),
        
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Final Merged Dataset"),
                    dbc.CardBody([
                        html.P(f"Total Rows: {df.shape[0]}", className="mb-1"),
                        html.P(f"Total Features: {df.shape[1]}", className="mb-1"),
                    ]),
                ], className="mb-4 shadow-sm"),
                md=4
            ),
        ]),
        
        html.Div([
            html.H5("Comprehensive Data Dictionary (55 Features)", className="d-inline"),
            dbc.Button(
                "📥 Export Dictionary (CSV)", 
                id="btn-download-dict", 
                color="secondary", 
                size="sm", 
                className="ms-3 mb-2",
                outline=True
            ),
            dcc.Download(id="download-feature-dict")
        ]),
        
        # Searchable and Sortable Data Dictionary
        dash_table.DataTable(
            id='feature-dictionary-table',
            columns=[{"name": i, "id": i} for i in full_feature_list.columns],
            data=full_feature_list.to_dict('records'),
            
            # FIXED: Enable native Filtering and Sorting
            filter_action="native",
            sort_action="native",
            page_size=10,
            
            # VISUAL STYLE: Matching your "Dataset Sample" table
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

        html.H5("Dataset Sample (First 10 Rows)", className="mt-2"),
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
            ["The Analyze tab is where we transform our prepared data into actionable insights and evaluate the effectiveness of our machine learning models. It is divided into two main sub-tabs: ", html.B("Exploratory Data Analysis (EDA)"), " and ", html.B("Model Performance"), "."]
        ),
        dbc.Tabs([
            dbc.Tab(label="Exploratory Data Analysis", children=[
                html.Div(
                    children=[
                        html.P(
                            ["The EDA section helps us understand the key characteristics of our data before starting the modeling. It's like checking the ingredients before cooking."]
                        ),
                        html.H5("Default Distribution", className="mt-4"),
                        html.P(
                            ["The pie chart below shows that our data is ", html.B("imbalanced"), 
"—only a small percentage of customers actually defaulted. This is common in banking data, which is why a high accuracy score alone can be misleading. A model that predicts no one will default would still be ~90% accurate but useless for identifying at-risk customers. We aren't just looking at percentages; we are seeing a critical business problem: ", html.B("class imbalance"), 
". The large slice for 'No Default' (status 0) and the tiny slice for 'Default' (status 1) means a model could achieve high accuracy simply by predicting 'No Default' all the time. That is why we cannot rely solely on accuracy and need more robust metrics, which we will find in the 'Model Performance' section."]
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
                            ["This stacked bar chart shows the percentage of defaulters and non-defaulters across different age groups. It helps us see if certain age ranges are more prone to default. The visualization reveals that while the total number of loans varies by age, the percentage of defaults within each group is relatively similar. By stacking the bars for 'No Default' and 'Default', we can see the proportion of each outcome within each age group. We are looking for significant differences in the default rate between age groups. Based on the data, the ", html.B("45-50 age group is the most prone to default"), ", with a slightly higher percentage of defaults compared to other age groups."]
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
                                    title="Percentage of Default by Age Group",
                                    yaxis_title="Percentage",
                                    xaxis_title="Age Group",
                                    height=450, margin=dict(t=50, b=50)
                                )
                            )
                        ),
                        html.H5("The Importance of Specific Transaction Data", className="mt-4"),
                        html.P(
                            ["Our analysis highlights the value of focusing on ", html.B("specific and granular data"), ". In this project, we created detailed features from raw transaction data, such as `avg_balance_before_loan` and `times_balance_below_5K`. These are much more informative than a simple total customer transaction value because they capture specific behaviors—like frequent overdrafts or low balances—that are strong indicators of financial stability and the likelihood of default. A simple 'total' metric would hide these crucial risk signals, making it harder to accurately predict customer risk."]
                        ),
                    ], className="p-4"
                )
            ]),
            dbc.Tab(label="Model Performance", children=[
                html.Div(
                    children=[
                        html.P(
                            ["This section is about evaluating our models to choose the best one for the task. We aren't just looking for a 'high score,' but a model that is genuinely good at detecting high-risk customers."]
                        ),
                        html.H5("Model Performance Metrics", className="mt-4"),
                        html.P(
                            ["To truly evaluate our models, we focus on several key metrics beyond simple accuracy:",
                             html.Ul([
                                 html.Li([html.B("Precision:"), " Think of Precision as the cost of a false alarm. If our model has high precision, the people it flags for follow-up are very likely actual defaulters. Of the customers we predicted would default, how many actually did? High precision is good for avoiding false alarms."]),
                                 html.Li([html.B("Recall:"), " Think of Recall as the cost of a missed warning. If our model has high recall, it is very good at finding most people who will default, so we don't miss a high-risk customer. Of all the customers who defaulted, how many did our model successfully identify? High recall is crucial for a bank to catch as many at-risk customers as possible."]),
                                 html.Li([html.B("F1-Score:"), " A balance between precision and recall, providing a single metric to compare models. This is the harmonic mean of precision and recall. It is a single number that helps us compare models when both precision and recall are important."]),
                                 html.Li([html.B("ROC-AUC:"), " This is a powerful summary metric. It measures the model's ability to distinguish between the two classes (defaulters vs. non-defaulters). A score closer to 1.0 is better."])
                             ])
                            ]
                        ),
                        html.P([
                            "The ", html.B("Random Forest"), ", ", html.B("Decision Tree"), ", ",
                            html.B("Gradient Boosting"), ", and ", html.B("SVM"), 
                            " models demonstrate perfect performance in identifying defaulters. ",
                            "Each achieved ", html.B("100% Precision, Recall, F1-Score, and Accuracy"),
                            ", along with an ", html.B("AUC of 1.00"), 
                            ". This means they classified both defaulters and non-defaulters without a single error, ",
                            "avoiding any missed high-risk customers and ensuring reliable business results. The ", html.B("Logistic Regression"), " model also performed strongly, with an ",
                            html.B("Accuracy of 99%"), " and an ", html.B("AUC of 1.00"),
                            ". However, it missed ", html.B("2 actual defaulters"), " (", html.B("Recall = 0.91"), 
                            "), which reduced its ", html.B("F1-Score for defaulters to 0.95"),
                            ". This makes it less reliable than the other models that achieved perfect detection."
                        ]),

                        dbc.Alert(
                            [
                                html.H5("💡 A Note on Performance & Integrity", className="alert-heading"),
                                html.P(
                                    "The exceptional performance (100% Accuracy) across several models indicates a very "
                                    "strong signal within the engineered features. We have conducted rigorous "
                                    "cross-validation to ensure these results represent genuine predictive power "
                                    "rather than data leakage (accidentally including the answer in the training data)."
                                ),
                                html.Hr(),
                                html.P(
                                    "This suggests that features like account balance and transaction frequency are "
                                    "highly definitive indicators of loan default risk in this specific dataset.",
                                    className="mb-0",
                                ),
                            ],
                            color="info",
                            className="mt-4 shadow-sm",
                        ),

                        html.H6("Confusion Matrix", className="mt-4"),
                        html.P(
                            ["A confusion matrix is a table that breaks down our model's predictions into four categories:", 
                             html.Ul([
                                 html.Li([html.B("True Positives (TP):"), " Correctly predicted defaulters."]),
                                 html.Li([html.B("True Negatives (TN):"), " Correctly predicted non-defaulters."]),
                                 html.Li([html.B("False Positives (FP):"), " Incorrectly predicted defaulters (Type I error). These are the 'false alarms'."]),
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
                            ["This bar chart shows us which features the selected model relied on most to make its predictions. We are seeing the model's 'thought process.' The taller the bar, the more influential that feature was. In this case, the two most important features were ", html.B("`avg_balance_before_loan`"), " and ", html.B("`avg_amount_trans_before_loan`"), ". This is a critical insight because it validates the data preparation process—our work on feature engineering paid off, creating meaningful signals for the model."]
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

# --- 4. EXPLAIN Tab ---
explain_tab = html.Div(
    children=[
        html.H4(["🔍 ", html.B("EXPLAIN"), " — Individual Risk Drivers"], className="mt-4"),
        html.P("Compare model predictions and see which features most influenced that specific result."),
        
        dbc.Row([
            dbc.Col([
                html.H6("1. Select Customer Index:"),
                dcc.Dropdown(
                    id="customer-dropdown",
                    options=[{'label': f'Customer {i}', 'value': i} for i in range(len(X_test))],
                    value=0,
                    clearable=False,
                ),
            ], md=3),
            dbc.Col([
                html.H6("2. Select Model:"),
                dcc.Dropdown(
                    id="explain-model-dropdown",
                    options=[{'label': name, 'value': name} for name in trained_models.keys()],
                    value='Random Forest',
                    clearable=False,
                ),
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Prediction & Confidence"),
                    dbc.CardBody([
                        # Row for Result Text and Gauge
                        dbc.Row([
                            dbc.Col([
                                html.H3(id="prediction-result-text", className="text-center mb-0"),
                                html.Div(id="consensus-alert-container")
                            ], md=6, className="d-flex flex-column justify-content-center"),
                            dbc.Col([
                                dcc.Graph(id="confidence-gauge", style={"height": "150px"})
                            ], md=6)
                        ])
                    ])
                ])
            ], md=6)
        ], className="mb-4"),
        
        dcc.Graph(id="shap-waterfall-plot"),

        dbc.Row([
            dbc.Col([
                html.Div([
                    html.B("Chart Legend:"),
                    html.Div([
                        html.Span("█", style={"color": "#ff4136", "margin-right": "10px"}),
                        html.Span("Red: Feature increases the probability of Default (Increases Risk)")
                    ]),
                    html.Div([
                        html.Span("█", style={"color": "#0074d9", "margin-right": "10px"}),
                        html.Span("Blue: Feature decreases the probability of Default (Increases Safety)")
                    ]),
                ], className="p-3 border rounded bg-light", style={"font-size": "0.9rem"})
            ], md=8),
            dbc.Col([
                dbc.Button(
                    "📥 Download Case Report (PDF)", 
                    id="btn-download-local", 
                    color="dark", 
                    className="w-100 h-100",
                    outline=True
                ),
                dcc.Download(id="download-local-analysis")
            ], md=4)
        ], className="mt-4")
    ], className="p-4"
)

# --- 5. ACT Tab ---
act_tab = html.Div([
    # Header Section
    html.Div([
        html.H3("🚀 ACT — Strategic Decisions", className="mb-3"),
        html.P("Translate data insights into business strategy and actionable bank policy."),
    ], className="p-4 bg-light border-bottom"),

    dbc.Container([
        # Strategic Content Section
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H5("💡 Business Strategy & Recommendations", className="mt-4"),
                    html.Hr(),
                    
                    html.B("Prioritize with Data"),
                    html.P([
                        "The models identified key risk indicators, such as ", html.Code("min_balance_before_loan"), 
                        " and ", html.Code("times_balance_below_5K"), ". Bank managers should use these to create robust "
                        "risk assessment rules."
                    ]),

                    html.B("Proactive Retention"),
                    html.P([
                        "Use this model to identify high-risk accounts daily. Customer service teams can proactively "
                        "offer counseling, reducing potential capital loss."
                    ]),

                    html.B("Implement Primary Model"),
                    html.P([
                        "While ", html.B("Random Forest, SVM, and Trees"), " also achieved 100% accuracy, ", 
                        html.I("Gradient Boosting"), " is our recommendation for production. It is the industry standard "
                        "for tabular data, offering the best balance of stability and predictive power for long-term "
                        "risk strategy."
                    ]),
                ], className="p-3")
            ], md=12)
        ]),

        # Action Buttons Section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Internal Actions", className="mb-0")),
                    dbc.CardBody([
                        html.P("Download the formal documentation for audit:"),
                        dcc.Dropdown(
                            id="report-model-dropdown",
                            options=[{'label': name, 'value': name} for name in trained_models.keys()],
                            value='Gradient Boosting',
                            className="mb-3"
                        ),
                        dbc.Button(
                            "📥 Download Executive Report (PDF)", 
                            id="btn-pdf", 
                            color="success", 
                            className="w-100 mb-3"
                        ),
                        dcc.Download(id="download-pdf-report"),
                        
                        html.Hr(),
                        html.H6("Escalation Priority:"),
                        dbc.RadioItems(
                            id="urgency-selector",
                            options=[
                                {"label": "🟢 Low", "value": "LOW"},
                                {"label": "🟡 Medium", "value": "MEDIUM"},
                                {"label": "🔴 High", "value": "HIGH / URGENT"},
                            ],
                            value="MEDIUM",
                            inline=True,
                            className="mb-3"
                        ),
                        dbc.Button(
                            "📧 Email Risk Department", 
                            id="btn-email-risk", 
                            href="", 
                            target="_blank",
                            color="primary", 
                            outline=True,
                            className="w-100"
                        )
                    ])
                ], className="shadow-sm mt-4 mb-4")
            ], md=6),
            
            dbc.Col([
                html.Div([
                    html.H5("What is in this report?", className="mt-4"),
                    html.Ul([
                        html.Li("Model performance validation metrics."),
                        html.Li("The top 3 data points driving loan decisions."),
                        html.Li("Specific business recommendations based on findings."),
                        html.Li("Formal signature blocks and audit-ready Report ID."),
                    ], className="mt-3")
                ], className="p-4")
            ], md=6)
        ])
    ], fluid=True)
])

app.layout = dbc.Container(
    [
        header,
        dbc.Tabs(
            [
                dbc.Tab(ask_tab, label="Ask"),
                dbc.Tab(prepare_tab, label="Prepare"),
                dbc.Tab(analyze_tab, label="Analyze"),
                dbc.Tab(explain_tab, label="Explain"),
                dbc.Tab(act_tab, label="Act"),
            ]
        ),
    ],
    fluid=True,
)

# --- Callbacks ---

@app.callback(
    Output("download-pdf-report", "data"),
    Input("btn-pdf", "n_clicks"),
    State("report-model-dropdown", "value"),
    prevent_initial_call=True,
)
def generate_bank_report(n_clicks, selected_model):
    model = trained_models[selected_model]
    
    # 1. Data Prep
    X_target = X_scaled_test if selected_model in ['SVM', 'Logistic Regression'] else X_test
    y_pred = model.predict(X_target)
    acc, prec, rec = accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, zero_division=0), recall_score(y_test, y_pred, zero_division=0)
    
    # 2. Translations
    translations = {"sankc. urok": "Penalty Interest (Late Fees)", "urok": "Interest Credit", "sipo": "Household Payments (SIPO)", "sluzby": "Service Charges", "pojistne": "Insurance Premiums", "duchod": "Pension Income", "min_balance_before_loan": "Minimum Recorded Balance", "avg_balance_3m_before_loan": "Average Balance (3 Months Prior)", "times_balance_below_5k": "Low Balance Frequency (<5K)", "amount": "Loan Principal Amount"}

    # 3. Feature Importance Logic
    importances = None
    if hasattr(model, 'feature_importances_'): importances = model.feature_importances_
    elif hasattr(model, 'coef_'): importances = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
    
    if importances is not None:
        top_indices = np.argsort(importances)[-3:][::-1]
        top_features = []
        for i in top_indices:
            raw_name = str(X_orig.columns[i]).lower()
            prefix = "Balance: " if "balance" in raw_name else "Total: " if "amount" in raw_name else ""
            found = False
            for tech, human in translations.items():
                if tech in raw_name:
                    top_features.append(f"{prefix}{human}"); found = True; break
            if not found: top_features.append(raw_name.replace('_', ' ').title())
    else:
        top_features = ["High-Dimensional Pattern Recognition", "Non-Linear Risk Correlation", "Complex Transaction Behavior"]

    # 4. Build PDF
    pdf = FPDF()
    pdf.add_page()
    
    # --- Branded Header ---
    try: pdf.image('logo.png', x=10, y=8, w=30)
    except:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(30, 10, "BANK LOGO", border=1, ln=0, align='C')

    pdf.set_xy(45, 10)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "BANK LOAN RISK ANALYSIS REPORT", ln=True, align='L')
    pdf.set_xy(45, 18)
    pdf.set_font("Arial", '', 10)
    # pdf.cell(0, 10, f"Generated Date: {datetime.now().year}", ln=True, align='L')
    # UPDATED: Added month, day, and time (hour, minutes, seconds)
    full_date_str = datetime.now().strftime("%B %d, %Y - %H:%M:%S")
    pdf.cell(0, 10, f"Generated Date: {full_date_str}", ln=True, align='L')
    
    # --- Action Status Badge (New) ---
    pdf.set_xy(160, 10)
    pdf.set_fill_color(40, 167, 69) # Success Green
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(45, 8, "APPROVED FOR USE", border=0, ln=1, align='C', fill=True)
    pdf.set_text_color(0, 0, 0) # Reset text color

    pdf.set_xy(10, 30)
    pdf.cell(0, 5, "-" * 85, ln=True, align='C')
    pdf.ln(5)

    # Executive Summary
    pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, f"SELECTED MODEL: {selected_model}", ln=True); pdf.ln(2)
    pdf.cell(0, 10, "EXECUTIVE SUMMARY:", ln=True); pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 7, f"The {selected_model} model serves as the primary decision-support tool for risk assessment, utilizing transaction patterns to predict default probability.")
    pdf.ln(5)

    # Risk Drivers
    pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "KEY RISK DRIVERS:", ln=True); pdf.set_font("Arial", '', 11)
    for i, feat in enumerate(top_features, 1): pdf.cell(0, 7, f"{i}. {feat}", ln=True)
    pdf.ln(5)

    # Comparison Table
    pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "MODEL PERFORMANCE COMPARISON:", ln=True); pdf.ln(2)
    pdf.set_font("Arial", 'B', 10); pdf.set_fill_color(230, 230, 230)
    pdf.cell(45, 8, "Model Name", 1, 0, 'C', True); pdf.cell(35, 8, "Accuracy", 1, 0, 'C', True); pdf.cell(35, 8, "Precision", 1, 0, 'C', True); pdf.cell(35, 8, "Recall", 1, 0, 'C', True); pdf.cell(40, 8, "Type", 1, 1, 'C', True)
    
    pdf.set_font("Arial", '', 10)
    for name, m in trained_models.items():
        X_comp = X_scaled_test if name in ['SVM', 'Logistic Regression'] else X_test
        y_c_pred = m.predict(X_comp)
        if name == selected_model: pdf.set_font("Arial", 'B', 10); pdf.set_text_color(0, 102, 204)
        else: pdf.set_font("Arial", '', 10); pdf.set_text_color(0, 0, 0)
        pdf.cell(45, 8, name, 1); pdf.cell(35, 8, f"{accuracy_score(y_test, y_c_pred):.0%}", 1); pdf.cell(35, 8, f"{precision_score(y_test, y_c_pred, zero_division=0):.2f}", 1); pdf.cell(35, 8, f"{recall_score(y_test, y_c_pred, zero_division=0):.2f}", 1); pdf.cell(40, 8, "Linear/Kernel" if name in ['SVM', 'Logistic Regression'] else "Ensemble/Tree", 1, 1)

    pdf.set_text_color(0, 0, 0); pdf.ln(5)

    # --- NEW: METHODOLOGY SECTION ---
    pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "METHODOLOGY & VALIDATION NOTE:", ln=True)
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 6, "In a banking context, Recall (Risk Detection) is prioritized over Accuracy. While Accuracy measures overall correctness, Recall measures the model's ability to capture actual defaulters. A 'missed warning' (False Negative) is financially costlier than a 'false alarm' (False Positive).")

    pdf.set_text_color(0, 0, 0); pdf.ln(5)
    pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "STRATEGIC RECOMMENDATIONS:", ln=True); pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 7, "- Implement automated flagging for accounts with high 'Penalty Interest' activity.\n- Conduct secondary manual reviews for applicants with 'Minimum Recorded Balances' below threshold.\n- Integrate real-time 'Low Balance' alerts into the early-warning risk system.")
    
    # --- Signatures (New) ---
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(90, 10, "__________________________", 0, 0, 'L')
    pdf.cell(90, 10, "__________________________", 0, 1, 'R')
    pdf.set_font("Arial", '', 9)
    pdf.cell(90, 5, "Authorized Risk Officer Signature", 0, 0, 'L')
    pdf.cell(90, 5, "Lead Data Scientist Signature", 0, 1, 'R')

    # Page 2: Glossary
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14); pdf.cell(0, 10, "GLOSSARY OF BANKING ANALYTICS TERMS", ln=True); pdf.cell(0, 5, "-" * 40, ln=True); pdf.ln(5)
    glossary = {"Accuracy": "Correct classification percentage.", "Precision": "False alarm avoidance metric.", "Recall": "Missed warning detection metric.", "Sankc. Urok": "Penalty interest on late payments.", "SIPO": "Household payment volatility indicator.", "Ensemble Model": "Multi-model 'team' for stable predictions."}
    for term, definition in glossary.items():
        pdf.set_font("Arial", 'B', 11); pdf.cell(0, 7, f"{term}:", ln=True); pdf.set_font("Arial", '', 11); pdf.multi_cell(0, 6, definition); pdf.ln(3)

    pdf.ln(5); pdf.set_font("Arial", 'I', 8); pdf.set_text_color(150, 150, 150)
    # pdf.cell(0, 10, f"Report ID: {selected_model.upper()}-{datetime.now().strftime('%Y%m%d%H%M')}", ln=True, align='C')

    ## 1. Professional ID Generation
    model_codes = {
        "Random Forest": "RF",
        "Decision Tree": "DT",
        "Gradient Boosting": "GB",
        "SVM": "SVM",
        "Logistic Regression": "LR"
    }
    m_code = model_codes.get(selected_model, "ML")
    
    # Format: BNK-RISK-[YEAR][MONTH][DAY]-[HOUR][MIN][SEC]-[MODEL]
    # Example: BNK-RISK-20251219-143005-RF
    timestamp_id = datetime.now().strftime('%Y%m%d-%H%M%S')
    report_id = f"BNK-RISK-{timestamp_id}-{m_code}"

    # 2. Final Confidentiality Footer (On Glossary Page)
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 10, f"Report ID: {report_id}", ln=True, align='C')
    pdf.cell(0, 5, "This document contains proprietary algorithmic insights. Unauthorized distribution is prohibited.", ln=True, align='C')

    # 3. Export to Dash
    # return dcc.send_bytes(pdf.output(dest='S').encode('latin-1'), f"{report_id}.pdf")
    return dcc.send_bytes(pdf.output(dest='S').encode('latin-1'), f"Bank_Risk_Report_{selected_model}.pdf")

@app.callback(
    Output("shap-waterfall-plot", "figure"),
    Output("prediction-result-text", "children"),
    Output("consensus-alert-container", "children"),
    Output("confidence-gauge", "figure"),
    Input("customer-dropdown", "value"),
    Input("explain-model-dropdown", "value")
)
def update_explanation(cust_idx, selected_model):
    model = trained_models[selected_model]
    is_linear = selected_model in ['SVM', 'Logistic Regression']
    
    # Data selection
    if is_linear:
        samp = X_scaled_test.iloc[cust_idx].values.reshape(1, -1) if isinstance(X_scaled_test, pd.DataFrame) else X_scaled_test[cust_idx].reshape(1, -1)
        current_vals = X_scaled_test.iloc[cust_idx].values if isinstance(X_scaled_test, pd.DataFrame) else X_scaled_test[cust_idx]
    else:
        samp = X_test.iloc[cust_idx:cust_idx+1]
        current_vals = X_test.iloc[cust_idx].values
    
    # 1. Prediction & Probability
    main_pred = model.predict(samp)[0]
    prob = model.predict_proba(samp)[0][1] if hasattr(model, "predict_proba") else float(main_pred)
    
    # Text including the percentage
    status = "HIGH RISK" if main_pred == 1 else "LOW RISK"
    emoji = "⚠️" if main_pred == 1 else "✅"
    result_text = f"{emoji} {status} ({prob:.1%})"
    
    # 2. Confidence Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob * 100,
        number = {'suffix': "%", 'font': {'size': 24}, 'valueformat':'.1f'},
        title = {'text': "Default Probability Score", 'font': {'size': 14}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#2c3e50"},
            'steps': [
                {'range': [0, 30], 'color': "#27ae60"}, # Green
                {'range': [30, 70], 'color': "#f1c40f"}, # Yellow
                {'range': [70, 100], 'color': "#e74c3c"} # Red
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 50}
        }
    ))
    fig_gauge.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=150)

    # 3. Consensus Logic
    all_preds = [m.predict(X_scaled_test.iloc[cust_idx].values.reshape(1, -1) if n in ['SVM', 'Logistic Regression'] else X_test.iloc[cust_idx:cust_idx+1])[0] for n, m in trained_models.items()]
    agreement = all_preds.count(main_pred)
    alert_msg = dbc.Alert(f"Consensus: {agreement}/{len(trained_models)}", color="success" if agreement == len(trained_models) else "warning", className="py-1 text-center small")

    # 4. Waterfall logic
    feature_names = X_orig.columns.tolist()
    if hasattr(model, 'feature_importances_'):
        contributions = (current_vals - X_test.mean().values) * model.feature_importances_
    elif hasattr(model, 'coef_'):
        contributions = current_vals * (model.coef_[0] if model.coef_.ndim > 1 else model.coef_)
    else: contributions = np.zeros(len(feature_names))

    df_top = pd.DataFrame({'f': feature_names, 'c': contributions})
    df_top = df_top.reindex(df_top.c.abs().sort_values(ascending=False).index).head(15).sort_values('c')

    fig_wf = go.Figure(go.Waterfall(orientation="h", x=df_top['c'], y=df_top['f'], 
                                   increasing={"marker": {"color": "#ff4136"}}, 
                                   decreasing={"marker": {"color": "#0074d9"}}))
    fig_wf.update_layout(title=f"Risk Factor Breakdown: Customer {cust_idx}", height=500, margin=dict(l=150))
    
    return fig_wf, result_text, alert_msg, fig_gauge


@app.callback(
    Output("download-local-analysis", "data"),
    Input("btn-download-local", "n_clicks"),
    State("customer-dropdown", "value"),
    State("explain-model-dropdown", "value"),
    State("prediction-result-text", "children"),
    prevent_initial_call=True,
)
def download_local_pdf(n_clicks, cust_idx, model_name, result_text):
    # CRITICAL FIX: Strip emojis for PDF compatibility
    clean_result = result_text.replace("✅", "").replace("⚠️", "").strip()
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "INDIVIDUAL LOAN RISK CASE REPORT", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Customer ID Reference: {cust_idx}", ln=True)
    pdf.cell(0, 10, f"Assessment Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(0, 10, f"Model Architecture: {model_name}", ln=True)
    pdf.cell(0, 10, f"Final Decision: {clean_result}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "TOP MATHEMATICAL RISK DRIVERS:", ln=True)
    pdf.set_font("Arial", '', 11)
    
    model = trained_models[model_name]
    if model_name in ['SVM', 'Logistic Regression']:
        vals = X_scaled_test.iloc[cust_idx].values if isinstance(X_scaled_test, pd.DataFrame) else X_scaled_test[cust_idx]
        coefs = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        contribs = vals * coefs
    else:
        vals = X_test.iloc[cust_idx].values
        contribs = (vals - X_test.mean().values) * model.feature_importances_

    df_local = pd.DataFrame({'f': X_orig.columns, 'c': contribs})
    top_7 = df_local.reindex(df_local.c.abs().sort_values(ascending=False).index).head(7)

    for i, row in enumerate(top_7.itertuples(), 1):
        impact = "INCREASES RISK" if row.c > 0 else "REDUCES RISK"
        # Ensure row.f (feature name) doesn't contain weird characters
        pdf.cell(0, 8, f"{i}. {str(row.f)}: {impact}", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", 'I', 9)
    pdf.multi_cell(0, 5, "Confidentiality Note: This report uses decision-support logic. Values above 50% on the probability gauge trigger a High Risk classification based on transaction patterns.")

    return dcc.send_bytes(pdf.output(dest='S').encode('latin-1'), f"Customer_{cust_idx}_Risk_Report.pdf")

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
    
    if selected_model in ['SVM', 'Logistic Regression']:
        X_test_for_pred = X_scaled_test
    else:
        X_test_for_pred = X_test

    y_pred = model.predict(X_test_for_pred)
    cm = confusion_matrix(y_test, y_pred)

    z_data = np.array([[cm[1, 1], cm[0, 1]], [cm[1, 0], cm[0, 0]]])
    cm_text = np.array([[f'TP: {cm[1, 1]}', f'FP: {cm[0, 1]}'], [f'FN: {cm[1, 0]}', f'TN: {cm[0, 0]}']])

    fig_cm = ff.create_annotated_heatmap(
        z=np.flipud(z_data),
        x=["Predicted Default (1)", "Predicted No Default (0)"],
        y=["Actual Default (1)", "Actual No Default (0)"],
        annotation_text=np.flipud(cm_text),
        colorscale='Blues'
    )
    fig_cm.update_layout(title=f"Confusion Matrix ({selected_model})", height=450)

    report = classification_report(y_test, y_pred, output_dict=False, zero_division=0)
    
    fig_fi = go.Figure()
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        df_importance = pd.DataFrame({'feature': X_orig.columns, 'importance': importances}).sort_values(by='importance', ascending=False)
        fig_fi.add_trace(go.Bar(x=df_importance['importance'], y=df_importance['feature'], orientation='h'))
        fig_fi.update_layout(title=f"Feature Importance: {selected_model}", height=500)
    else:
        fig_fi.update_layout(title=f"Feature Importance not available for {selected_model}")
        
    fig_roc = go.Figure()
    roc_description_list = []
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test_for_pred)[:, 1]
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
        roc_auc = metrics.auc(fpr, tpr)
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{selected_model} (AUC = {roc_auc:.2f})'))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(dash='dash')))
        fig_roc.update_layout(title="ROC Curve", height=450)
        
        roc_description_list.append(html.P(f"Selected Model: {selected_model} | AUC: {roc_auc:.2f}"))

    return fig_cm, report, fig_fi, fig_roc, roc_description_list
    
@app.callback(
    Output("btn-email-risk", "href"),
    Input("report-model-dropdown", "value"),
    Input("urgency-selector", "value")
)
def update_email_link(selected_model, urgency):
    # Set the primary recipient
    to_email = "risk_dept@yourbank.com"
    cc_email = ""
    
    # 1. Logic for Subject Prefix and Cc
    if "HIGH" in urgency:
        prefix = "🔴 CRITICAL"
        cc_email = "head_of_risk@yourbank.com"
    elif "MEDIUM" in urgency:
        prefix = "🟡 ROUTINE"
    else:
        prefix = "🟢 INFORMATIONAL"

    # 2. Get current timestamp for data freshness
    current_time = datetime.now().strftime("%B %d, %Y at %H:%M")

    # 3. Create the Subject Line
    subject = f"{prefix}: Risk Review Required ({selected_model})"
    
    # 4. Create the Body with Training Date
    body = (
        f"Hello Risk Management Team,\n\n"
        f"PRIORITY LEVEL: {urgency}\n"
        f"SYSTEM ALERT: Automated escalation triggered.\n\n"
        f"Following an analytical review using the {selected_model} model, I have identified "
        f"factors requiring immediate attention regarding our current loan approval criteria.\n\n"
        f"Key drivers suggest we monitor specific account behavior and sanction interest "
        f"patterns more closely. The full Executive Report is ready for your audit.\n\n"
        f"--------------------------------------------------\n"
        f"MODEL INSIGHT DATA FRESHNESS:\n"
        f"Analysis Generated: {current_time}\n"
        f"Data Source: Berka Banking Dataset (Ref: 1999/2025)\n"
        f"--------------------------------------------------\n\n"
        f"Best regards,\n"
        f"Loan Operations Department"
    )
    
    # 5. Safe Encoding for Outlook (%20 instead of +)
    safe_subject = urllib.parse.quote(subject)
    safe_body = urllib.parse.quote(body)
    
    mailto_link = f"mailto:{to_email}?subject={safe_subject}&body={safe_body}"
    
    if cc_email:
        mailto_link += f"&cc={urllib.parse.quote(cc_email)}"
        
    return mailto_link  

@app.callback(
    Output("download-feature-dict", "data"),
    Input("btn-download-dict", "n_clicks"),
    prevent_initial_call=True,
)
def download_dictionary(n_clicks):
    # This now exports all 55 features automatically
    return dcc.send_data_frame(full_feature_list.to_csv, "Full_Bank_Data_Dictionary.csv", index=False)

if __name__ == "__main__":
    app.run(debug=True)