import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

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


def check_df_nulls(df):
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


def encode_for_aggregation(df, column, encoder):
    """Encode "column" in "df" with the given encoder."""
    column_encoded = encoder.fit_transform(df[[column]])
    column_encoded.columns = column_encoded.columns.str.replace(" ", "_")
    return df.join(column_encoded)


def clean_agg_columns(df, table_name: str):
    """Clean-up the multi-indexed columns after an aggregation."""
    df_copy = df.copy()
    multi_columns = df_copy.columns
    text_to_map = table_name + "_{0[0]}_{0[1]}"
    single_columns = multi_columns.map(text_to_map.format)
    df_copy.columns = single_columns
    return df_copy


def aggregate_bur(df):
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
    aggregates = clean_agg_columns(aggregates, "BUR")
    aggregates = aggregates.rename(
        {
            "BUR_CREDIT_TYPE_Loan_for_business_development_sum": "BUR_CREDIT_TYPE_business_sum",
            "BUR_CREDIT_TYPE_infrequent_sklearn_sum": "BUR_CREDIT_TYPE_infrequent_sum",
        },
        axis=1,
    )
    return aggregates


def aggregate_cash(df):
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
    PREFIX = "CASH"
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


def aggregate_inst(df):
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
    aggregates = clean_agg_columns(aggregates, "INST")
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
    PREFIX = "CRED"
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


def plot_metrics(y_valid, y_pred) -> None:
    conf_mx = confusion_matrix(y_valid, y_pred)
    disp_rf = ConfusionMatrixDisplay(confusion_matrix=conf_mx)
    disp_rf.plot(cmap=figure_colors_cmap)
    print(classification_report(y_valid, y_pred, zero_division=0))
