import os
import pandas as pd
import hopsworks
import matplotlib.pyplot as plt
import seaborn as sns


FG_NAME = "aqi_features"
FG_VERSION = 1


def get_project():
    return hopsworks.login(
        host=os.getenv("HOPSWORKS_HOST"),
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    )


def main():
    print("Connecting to Hopsworks...")
    project = get_project()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(FG_NAME, version=FG_VERSION)

    print("Reading feature group...")
    df = fg.read()

    print("\nDataset Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nHead:")
    print(df.head())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nData Types:")
    print(df.dtypes)

    print("\nBasic Statistics:")
    print(df.describe())

    # Parse timestamp
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce", utc=True)

    # AQI distribution
    plt.figure()
    sns.countplot(x="aqi_target", data=df)
    plt.title("AQI Target Distribution (1-5)")
    plt.show()

    # Correlation matrix
    numeric_df = df.select_dtypes(include=["number"])
    corr = numeric_df.corr()

    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    plt.title("Feature Correlation Matrix")
    plt.show()

    # Time trend of AQI
    df = df.sort_values("event_time")
    plt.figure()
    plt.plot(df["event_time"], df["aqi_target"])
    plt.title("AQI Over Time")
    plt.xlabel("Time")
    plt.ylabel("AQI (1-5)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    print("\nEDA Complete.")


if __name__ == "__main__":
    main()
