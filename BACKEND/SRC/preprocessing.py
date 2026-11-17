import pandas as pd
from CONFIGS.paths import DATA_CONFIG, RAW_DATA_PATH_RISK, LOADING_CONFIG



def load_dataset():
    data_path = RAW_DATA_PATH_RISK
    if data_path.suffix.lower() != ".csv":
        raise ValueError("The provided dataset must be a CSV file.")
    chunk_size = LOADING_CONFIG.get("chunk_size")
    if chunk_size is None:
        raise ValueError("chunk_size is missing in data_config.json")
    
    print("loading data set chuks wise with chunksize:",chunk_size)
    chunks = pd.read_csv(
        data_path,
        chunksize=chunk_size
    )
    df = pd.concat(chunks, ignore_index=True)

    columns = [
        "loan_amnt", "term", "int_rate", "installment", "grade",
        "emp_length", "home_ownership", "annual_inc",
        "verification_status", "purpose", "dti", "delinq_2yrs",
        "fico_range_low", "fico_range_high", "open_acc", "pub_rec",
        "revol_bal", "revol_util", "total_acc", "loan_status"
    ]
    missing_columns = [
        col for col in columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )
    df = df[columns]

    return df
