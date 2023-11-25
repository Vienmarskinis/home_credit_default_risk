import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
    df_clean = df.replace({"X": np.nan, "C": -1})
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
