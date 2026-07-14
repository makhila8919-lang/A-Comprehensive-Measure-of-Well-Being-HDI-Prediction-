import pandas as pd


def load_dataset(path):

    """
    Load dataset from CSV file.
    """

    return pd.read_csv(path)


def clean_dataset(df):

    """
    Clean missing values and duplicate rows.
    """

    df = df.copy()

    df.drop_duplicates(inplace=True)

    df.fillna(df.median(numeric_only=True), inplace=True)

    return df


def select_features(df):

    """
    Select required features.
    """

    features = [

        "Life expectancy at birth",

        "Expected years of schooling",

        "Mean years of schooling",

        "Gross national income (GNI) per capita"

    ]

    return df[features]


def create_target(df):

    """
    Create HDI Category
    """

    def category(value):

        if value >= 0.800:
            return "Very High"

        elif value >= 0.700:
            return "High"

        elif value >= 0.550:
            return "Medium"

        return "Low"

    df["HDI Category"] = df["HDI value"].apply(category)

    return df
