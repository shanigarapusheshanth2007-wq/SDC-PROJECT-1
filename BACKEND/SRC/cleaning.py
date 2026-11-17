
import pandas as pd


def clean_term(df):
    df["term"] = (
        df["term"]
        .astype("string")
        .str.extract(r"(\d+)")
        .astype(float)
    )
    return df


def clean_emp_length(df):
    df["emp_length"] = (
        df["emp_length"]
        .astype("string")
        .str.extract(r"(\d+)")
        .astype(float)
    )
    return df


def handle_missing_values(df):

    num_cols = [
        "loan_amnt", "int_rate", "installment", "annual_inc",
        "dti", "delinq_2yrs", "fico_range_low", "fico_range_high",
        "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
        "term", "emp_length"
    ]

    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df


def clean_dataset(df):
    df = clean_term(df)
    df = clean_emp_length(df)
    df = handle_missing_values(df)

    return df


def preprocess_dataset(df):

    df = clean_dataset(df)

    df = pd.get_dummies(
        df,
        columns=[
            "grade",
            "home_ownership",
            "verification_status",
            "purpose",
            "emp_length"
        ],
        drop_first=True
    )

    return df
