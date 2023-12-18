import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    PrecisionRecallDisplay,
)
import constants
import warnings
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap

figure_colors_cmap = sns.color_palette("viridis", as_cmap=True)


def check_if_superset(
    super: pd.DataFrame,
    sub: pd.DataFrame,
    columns: list,
    super_name: str,
    sub_name: str,
) -> None:
    """Check if 'super' is a superset of 'sub' when compared by 'columns'"""
    super_ids = super[columns]
    sub_ids = sub[columns]
    test_superset_df = sub_ids.isin(super_ids)
    is_superset = test_superset_df.all().all()
    if is_superset:
        print(f"{super_name} table is a superset of {sub_name}")


def check_df_nulls(df: pd.DataFrame) -> None:
    if df.isna().sum() != 0:
        print(f"{df.isna().sum()} nulls found")
    else:
        print("All ok")


def aggregate_bur_bal(df: pd.DataFrame) -> pd.DataFrame:
    """Perform aggregations for the Bureau Balance table"""
    df_clean = df.replace({"X": np.nan, "C": -1}).copy()
    df_clean["STATUS"] = df_clean["STATUS"].astype(np.float32)

    last_statuses = df_clean[df_clean["MONTHS_BALANCE"] == 0]

    df_drop_C = df_clean[df_clean["STATUS"] != -1]
    aggregates = df_drop_C.groupby("SK_ID_BUREAU").agg(
        {"STATUS": ["mean", "std", "max"]}
    )
    aggregates = aggregates.droplevel(0, axis=1)

    merged_df = last_statuses.merge(aggregates, on="SK_ID_BUREAU", how="outer").drop(
        columns="MONTHS_BALANCE"
    )
    merged_df.columns = [
        "SK_ID_BUREAU",
        "STATUS_last",
        "STATUS_mean",
        "STATUS_std",
        "STATUS_max",
    ]
    return merged_df


def add_labels(ax=None, *args, **kwargs) -> None:
    """Adds labels to the matplotlib bar plot figure."""
    if ax is None:
        ax = plt.gca()
    for container in ax.containers:
        ax.bar_label(container, *args, **kwargs)


def encode_for_aggregation(df: pd.DataFrame, column: str, encoder) -> pd.DataFrame:
    """Encode "column" in "df" with the given encoder."""
    column_encoded = encoder.fit_transform(df[[column]])
    column_encoded.columns = column_encoded.columns.str.replace(" ", "_")
    return df.join(column_encoded)


def clean_agg_columns(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Clean-up the multi-indexed columns after an aggregation."""
    df_copy = df.copy()
    multi_columns = df_copy.columns
    text_to_map = table_name + "_{0[0]}_{0[1]}"
    single_columns = multi_columns.map(text_to_map.format)
    df_copy.columns = single_columns
    return df_copy


def aggregate_bur(df: pd.DataFrame) -> pd.DataFrame:
    """Perform aggregations for the Bureau table,
    in which Bureau Balances has already been merged.
    """
    df_copy = df.copy()

    # Handle CREDIT_TYPE
    oh_encoder = OneHotEncoder(max_categories=7, sparse_output=False).set_output(
        transform="pandas"
    )
    df_copy = encode_for_aggregation(df_copy, "CREDIT_TYPE", oh_encoder)

    # Handle CREDIT_ACTIVE
    df_copy = encode_for_aggregation(df_copy, "CREDIT_ACTIVE", oh_encoder)

    # Handle flags
    df_copy["Has_overdue"] = df_copy["CREDIT_DAY_OVERDUE"] > 0
    df_copy["Has_prolong"] = df_copy["CNT_CREDIT_PROLONG"] > 0

    # Handle ENDDATE
    credit_enddate_positive = df_copy[["SK_ID_CURR", "DAYS_CREDIT_ENDDATE"]].copy()
    credit_enddate_positive = credit_enddate_positive[
        credit_enddate_positive["DAYS_CREDIT_ENDDATE"] > 0
    ]
    credit_enddate_positive_agg = credit_enddate_positive.groupby("SK_ID_CURR").agg(
        {"DAYS_CREDIT_ENDDATE": ["mean", "max"]}
    )

    aggregates = df_copy.groupby("SK_ID_CURR").agg(
        {
            "CREDIT_CURRENCY": ["nunique"],
            "DAYS_CREDIT": ["min", "mean"],
            "AMT_CREDIT_SUM": ["mean", "max"],
            "AMT_CREDIT_SUM_DEBT": ["mean", "max"],
            "AMT_CREDIT_SUM_LIMIT": ["mean", "max"],
            "AMT_CREDIT_SUM_OVERDUE": ["mean", "max"],
            "CREDIT_TYPE": ["nunique"],
            "AMT_ANNUITY": ["mean", "max"],
            "STATUS_mean": ["mean"],
            "STATUS_std": ["mean"],
            "STATUS_max": ["mean"],
            "STATUS_last": ["mean"],
            "CREDIT_ACTIVE_Closed": ["sum"],
            "CREDIT_ACTIVE_Active": ["sum"],
            "CREDIT_ACTIVE_Sold": ["sum"],
            "CREDIT_ACTIVE_Bad_debt": ["sum"],
            "Has_overdue": ["sum"],
            "Has_prolong": ["sum"],
            "CREDIT_TYPE_Consumer_credit": ["sum"],
            "CREDIT_TYPE_Credit_card": ["sum"],
            "CREDIT_TYPE_Mortgage": ["sum"],
            "CREDIT_TYPE_Car_loan": ["sum"],
            "CREDIT_TYPE_Microloan": ["sum"],
            "CREDIT_TYPE_Loan_for_business_development": ["sum"],
            "CREDIT_TYPE_infrequent_sklearn": ["sum"],
        }
    )
    aggregates = aggregates.join(credit_enddate_positive_agg, how="outer")
    aggregates = clean_agg_columns(aggregates, "bur")
    aggregates = aggregates.rename(
        {
            "bur_CREDIT_TYPE_Loan_for_business_development_sum": "bur_CREDIT_TYPE_business_sum",
            "bur_CREDIT_TYPE_infrequent_sklearn_sum": "bur_CREDIT_TYPE_infrequent_sum",
        },
        axis=1,
    )
    return aggregates


def aggregate_cash(df: pd.DataFrame) -> pd.DataFrame:
    """Perform aggregations for the pos_cash_balance table."""
    df_copy = df.copy()
    df_copy["Has_overdue"] = df_copy["SK_DPD"] > 0
    df_copy["Has_overdue_def"] = df_copy["SK_DPD_DEF"] > 0
    aggregates = df_copy.groupby("SK_ID_CURR").agg(
        {
            "SK_ID_PREV": ["nunique", "count"],
            "CNT_INSTALMENT": ["max"],
            "SK_DPD": ["mean", "max"],
            "Has_overdue": ["sum"],
            "SK_DPD_DEF": ["mean", "max"],
            "Has_overdue_def": ["sum"],
        }
    )
    PREFIX = "cash"
    aggregates = clean_agg_columns(aggregates, PREFIX)
    active_statuses = df_copy[
        (df_copy["MONTHS_BALANCE"] == -1)
        & (df_copy["NAME_CONTRACT_STATUS"] == "Active")
    ]
    active_count = (
        active_statuses.groupby("SK_ID_CURR")["SK_ID_PREV"]
        .count()
        .rename(f"{PREFIX}_Status_Active_count")
    )
    aggregates = aggregates.join(active_count)
    return aggregates


def aggregate_inst(df: pd.DataFrame) -> pd.DataFrame:
    """Perform aggregations for the installments_payments table."""
    df_copy = df.copy()
    df_copy["Days_till_deadline"] = (
        df_copy["DAYS_INSTALMENT"] - df_copy["DAYS_ENTRY_PAYMENT"]
    )
    df_copy["Amt_deficit"] = df_copy["AMT_INSTALMENT"] - df_copy["AMT_PAYMENT"]
    aggregates = df_copy.groupby("SK_ID_CURR").agg(
        {
            "Days_till_deadline": ["mean", "max", "min"],
            "Amt_deficit": ["mean", "max"],
            "AMT_INSTALMENT": ["mean", "max"],
        }
    )
    aggregates = clean_agg_columns(aggregates, "inst")
    return aggregates


def aggregate_cred(df):
    """Perform aggregations for the credit_card_balance table."""
    df_copy = df.copy()
    df_copy["Has_overdue"] = df_copy["SK_DPD"] > 0
    df_copy["Has_overdue_def"] = df_copy["SK_DPD_DEF"] > 0
    aggregates = df_copy.groupby("SK_ID_CURR").agg(
        {
            "SK_ID_PREV": ["nunique", "count"],
            "SK_DPD": ["mean", "max"],
            "Has_overdue": ["sum"],
            "SK_DPD_DEF": ["mean", "max"],
            "Has_overdue_def": ["sum"],
            "AMT_BALANCE": ["median", "max"],
            "AMT_CREDIT_LIMIT_ACTUAL": ["median", "max"],
            "AMT_DRAWINGS_ATM_CURRENT": ["min", "max", "mean"],
            "AMT_DRAWINGS_CURRENT": ["min", "max", "mean"],
            "AMT_DRAWINGS_OTHER_CURRENT": ["mean", "max"],
            "AMT_DRAWINGS_POS_CURRENT": ["mean", "max"],
            "AMT_INST_MIN_REGULARITY": ["mean", "max"],
            "AMT_PAYMENT_CURRENT": ["mean", "max"],
            "AMT_PAYMENT_TOTAL_CURRENT": ["mean", "max"],
            "AMT_RECEIVABLE_PRINCIPAL": ["min", "max", "mean"],
            "AMT_RECIVABLE": ["min", "max", "mean"],
            "AMT_TOTAL_RECEIVABLE": ["min", "max", "mean"],
            "CNT_DRAWINGS_ATM_CURRENT": ["mean", "max"],
            "CNT_DRAWINGS_CURRENT": ["mean", "max"],
            "CNT_DRAWINGS_OTHER_CURRENT": ["mean", "max"],
            "CNT_DRAWINGS_POS_CURRENT": ["mean", "max"],
        }
    )
    PREFIX = "cred"
    aggregates = clean_agg_columns(aggregates, PREFIX)
    active_statuses = df_copy[
        (df_copy["MONTHS_BALANCE"] == -1)
        & (df_copy["NAME_CONTRACT_STATUS"] == "Active")
    ]
    active_count = (
        active_statuses.groupby("SK_ID_CURR")["SK_ID_PREV"]
        .count()
        .rename(f"{PREFIX}_Status_Active_count")
    )
    aggregates = aggregates.join(active_count)
    return aggregates


def aggregate_prev(df):
    """Perform aggregations for the previous_application table."""
    df_copy = df.copy()
    oh_encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
    df_copy = encode_for_aggregation(df_copy, "NAME_CONTRACT_TYPE", oh_encoder)
    df_copy = encode_for_aggregation(df_copy, "NAME_CONTRACT_STATUS", oh_encoder)
    df_copy = encode_for_aggregation(df_copy, "NAME_PORTFOLIO", oh_encoder)
    df_copy = encode_for_aggregation(df_copy, "NAME_PRODUCT_TYPE", oh_encoder)
    df_copy = encode_for_aggregation(df_copy, "NAME_YIELD_GROUP", oh_encoder)
    df_copy["Credit_diff"] = df_copy["AMT_APPLICATION"] - df_copy["AMT_CREDIT"]
    aggregates = df_copy.groupby("SK_ID_CURR").agg(
        {
            "NAME_CONTRACT_TYPE_Consumer_loans": ["sum"],
            "NAME_CONTRACT_TYPE_Cash_loans": ["sum"],
            "NAME_CONTRACT_TYPE_Revolving_loans": ["sum"],
            "AMT_ANNUITY": ["mean", "max"],
            "AMT_APPLICATION": ["mean", "max"],
            "Credit_diff": ["mean", "max"],
            "AMT_DOWN_PAYMENT": ["mean", "max", "min"],
            "AMT_GOODS_PRICE": ["mean", "max"],
            "NFLAG_LAST_APPL_IN_DAY": ["mean"],
            "RATE_DOWN_PAYMENT": ["mean", "max", "min"],
            "RATE_INTEREST_PRIMARY": ["mean", "max", "min"],
            "RATE_INTEREST_PRIVILEGED": ["mean", "max", "min"],
            "NAME_CONTRACT_STATUS_Approved": ["sum"],
            "NAME_CONTRACT_STATUS_Refused": ["sum"],
            "NAME_CONTRACT_STATUS_Canceled": ["sum"],
            "NAME_CONTRACT_STATUS_Unused_offer": ["sum"],
            "DAYS_DECISION": ["mean", "count"],
            "NAME_GOODS_CATEGORY": ["nunique"],
            "NAME_PORTFOLIO_POS": ["sum"],
            "NAME_PORTFOLIO_Cash": ["sum"],
            "NAME_PORTFOLIO_Cards": ["sum"],
            "NAME_PRODUCT_TYPE_x-sell": ["sum"],
            "NAME_PRODUCT_TYPE_walk-in": ["sum"],
            "CNT_PAYMENT": ["mean"],
            "NAME_YIELD_GROUP_middle": ["sum"],
            "NAME_YIELD_GROUP_low_action": ["sum"],
            "NAME_YIELD_GROUP_high": ["sum"],
            "NAME_YIELD_GROUP_low_normal": ["sum"],
            "NFLAG_INSURED_ON_APPROVAL": ["mean"],
        }
    )
    PREFIX = "prev"
    aggregates = clean_agg_columns(aggregates, PREFIX)
    return aggregates


def norm_plot(
    df: pd.DataFrame,
    x: str,
    hue: str,
    ax: list[plt.Axes],
    *args,
    **kwargs,
):
    """Create a figure of two plots. First plot shows normalised counts and the second is just a countplot."""
    hue_percent = f"{hue}, percent"
    df_counts = df.groupby(x)[hue].value_counts(normalize=True)
    df_counts = df_counts.mul(100).rename(hue_percent).reset_index()

    sns.barplot(
        y=df_counts[x],
        x=df_counts[hue_percent],
        hue=df_counts[hue],
        ax=ax[0],
        orient="h",
        *args,
        **kwargs,
    )
    mean_positive_proportion = df_counts[df_counts[hue] == 1][hue_percent].mean()
    ax[0].axvline(
        mean_positive_proportion,
        color="red",
        alpha=0.5,
        ls="--",
        label="mean of 1",
    )
    ax[0].legend()

    sns.countplot(
        y=df[x],
        stat="percent",
        ax=ax[1],
        *args,
        **kwargs,
    )
    sns.despine(ax=ax[0])
    sns.despine(ax=ax[1])
    add_labels(ax=ax[1], fmt="%1.1f%%")
    ax[0].set_xlabel("Proportion in Specific Group")
    ax[0].set_ylabel("")
    ax[1].set_xlabel("Percent of the Whole Dataset")


def preprocess_test_train_TARGET(df: pd.DataFrame, preprocessor) -> tuple[pd.DataFrame]:
    """Split the dataframe into X/y and train/test, transform X with the preprocessor."""
    df_train, df_valid = train_test_split(
        df, test_size=0.2, stratify=df["TARGET"], random_state=42
    )
    X_train = df_train.drop(columns="TARGET")
    y_train = df_train["TARGET"]
    X_valid = df_valid.drop(columns="TARGET")
    y_valid = df_valid["TARGET"]

    X_train_tf = preprocessor.fit_transform(X_train, y_train)
    X_valid_tf = preprocessor.transform(X_valid)

    return X_train_tf, y_train, X_valid_tf, y_valid


def report_classification_metrics(y_truth: pd.DataFrame, y_pred: pd.DataFrame) -> None:
    """Print classification metrics for the given truth and prediction lists"""
    conf_mx = confusion_matrix(y_truth, y_pred)
    disp_rf = ConfusionMatrixDisplay(confusion_matrix=conf_mx)
    disp_rf.plot(cmap=figure_colors_cmap)
    print(classification_report(y_truth, y_pred, zero_division=0))


def report_xgb_metrics(model, X: pd.DataFrame, y: pd.DataFrame, name: str):
    """Print XGB classification metrics: log loss and PRC"""
    print(f"log loss: {model.best_score:1.3f}")
    disp = PrecisionRecallDisplay.from_estimator(
        model, X, y, name=name, plot_chance_level=True
    )
    _ = disp.ax_.set_title(f"{name} Precision-Recall Curve")
    plt.legend(loc=1)
    plt.grid()
    sns.despine()


def get_shap_importance_df(X: pd.DataFrame, shap_values):
    """Create a shap feature importance dataframe"""
    feature_names = X.columns
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    importance = np.abs(shap_df.values).mean(0)

    shap_importance_df = pd.DataFrame(
        list(zip(feature_names, importance)),
        columns=["col_name", "shap_feature_importance"],
    )
    shap_importance_df.sort_values(
        by=["shap_feature_importance"], ascending=False, inplace=True
    )
    shap_importance_df = shap_importance_df.reset_index(drop=True)
    shap_importance_df["shap_rank"] = shap_importance_df.index

    return shap_importance_df


def shap_sequentially_select(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_valid: pd.DataFrame,
    reduce_to: int = 20,
    learning_rate: float = 0.2,
    early_stopping_rounds: int = 20,
    sample_frac: float = 0.3,
) -> pd.Series:
    """Sequentially select features based on shap values"""
    warnings.filterwarnings("ignore", category=Warning)
    selected_cols = X_train.columns
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=500,
        learning_rate=learning_rate,
        early_stopping_rounds=early_stopping_rounds,
        seed=42,
        random_state=42,
    )
    while len(selected_cols) > reduce_to:
        X_train_selected = X_train[selected_cols]
        X_valid_selected = X_valid[selected_cols]
        model.fit(
            X_train_selected,
            y_train,
            eval_set=[(X_valid_selected, y_valid)],
            verbose=0,
        )
        X_valid_sample = X_valid_selected.sample(frac=sample_frac, random_state=42)
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_valid_sample)
        shap_importance_df = get_shap_importance_df(X_valid_sample, shap_vals)
        selected_cols = shap_importance_df["col_name"].iloc[:-1]
    return selected_cols


def aggregate_cash_TARGET_20(df: pd.DataFrame) -> pd.DataFrame:
    """Produce selected aggregations of pos_cash_balance table
    for TARGET 20 feature model prediction."""
    df_copy = df.copy()
    aggregates = df_copy.groupby("SK_ID_CURR").agg(
        {
            "SK_ID_PREV": ["count"],
            "CNT_INSTALMENT": ["max"],
        }
    )
    PREFIX = "cash"
    aggregates = clean_agg_columns(aggregates, PREFIX)
    return aggregates


def aggregate_inst_TARGET_20(df: pd.DataFrame) -> pd.DataFrame:
    """Produce selected aggregations of installments_payments table
    for TARGET 20 feature model prediction."""
    df_copy = df.copy()
    df_copy["Days_till_deadline"] = (
        df_copy["DAYS_INSTALMENT"] - df_copy["DAYS_ENTRY_PAYMENT"]
    )
    df_copy["Amt_deficit"] = df_copy["AMT_INSTALMENT"] - df_copy["AMT_PAYMENT"]
    aggregates = df_copy.groupby("SK_ID_CURR").agg(
        {
            "Days_till_deadline": ["min"],
            "Amt_deficit": ["mean"],
        }
    )
    PREFIX = "inst"
    aggregates = clean_agg_columns(aggregates, PREFIX)
    return aggregates


def aggregate_bur_TARGET_20(df: pd.DataFrame) -> pd.DataFrame:
    """Produce selected aggregations of bureau table
    for TARGET 20 feature model prediction."""
    df_copy = df.copy()
    aggregates = df_copy.groupby("SK_ID_CURR").agg(
        {
            "DAYS_CREDIT": ["mean"],
        }
    )
    PREFIX = "bur"
    aggregates = clean_agg_columns(aggregates, PREFIX)
    return aggregates


def aggregate_prev_TARGET_20(df: pd.DataFrame) -> pd.DataFrame:
    """Produce selected aggregations of previous_application table
    for TARGET 20 feature model prediction."""
    df_copy = df.copy()
    oh_encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
    df_copy = encode_for_aggregation(df_copy, "NAME_CONTRACT_STATUS", oh_encoder)
    df_copy["Credit_diff"] = df_copy["AMT_APPLICATION"] - df_copy["AMT_CREDIT"]
    aggregates = df_copy.groupby("SK_ID_CURR").agg(
        {
            "AMT_ANNUITY": ["mean"],
            "NAME_CONTRACT_STATUS_Refused": ["sum"],
            "Credit_diff": ["max"],
        }
    )
    PREFIX = "prev"
    aggregates = clean_agg_columns(aggregates, PREFIX)
    return aggregates


def prepare_data_TARGET_20(
    df_cash: pd.DataFrame,
    df_inst: pd.DataFrame,
    df_bur: pd.DataFrame,
    df_prev_app: pd.DataFrame,
    df_app: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare the simplified dataset for TARGET prediction"""
    agg_cash = aggregate_cash_TARGET_20(df_cash)
    agg_inst = aggregate_inst_TARGET_20(df_inst)
    agg_bur = aggregate_bur_TARGET_20(df_bur)
    agg_prev = aggregate_prev_TARGET_20(df_prev_app)
    df_app_selected = df_app.set_index("SK_ID_CURR")[constants.APP_COLS_TARGET_20]
    df_merged = (
        df_app_selected.join(agg_cash).join(agg_inst).join(agg_bur).join(agg_prev)
    )
    return df_merged


def calculate_profit_TARGET(y_truth, y_pred):
    # Define Income matrix
    I00 = 0.2
    I11 = 0
    I01 = -0.8
    I10 = -0.2
    I = np.array([(I00, I10), (I01, I11)])

    conf_mx = confusion_matrix(y_truth, y_pred)
    profit = np.sum(conf_mx * I)
    return profit
