from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
DATASET_PATH = Path(__file__).with_name("Dataset.csv")
MIN_CLASS_SAMPLES = 10


def prepare_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build numeric features and the rating target from the dataset."""
    data = data.copy()
    class_counts = data["rating"].value_counts()
    data = data[data["rating"].isin(class_counts[class_counts >= MIN_CLASS_SAMPLES].index)].copy()

    data["is_movie"] = data["type"].eq("Movie").astype(int)
    data["duration_value"] = pd.to_numeric(
        data["duration"].str.extract(r"(\d+)", expand=False), errors="coerce"
    )
    date_added = pd.to_datetime(data["date_added"], errors="coerce")
    data["year_added"] = date_added.dt.year.fillna(date_added.dt.year.median())
    data["month_added"] = date_added.dt.month.fillna(1)
    data["has_director"] = data["director"].fillna("Unknown").str.lower().ne("unknown").astype(int)

    main_country = data["country"].fillna("Unknown").str.split(",").str[0].str.strip()
    top_countries = main_country.value_counts().head(10).index
    country_features = pd.get_dummies(
        main_country.where(main_country.isin(top_countries), "Other"),
        prefix="country",
        dtype=int,
    )
    genre_features = data["listed_in"].fillna("").str.get_dummies(sep=", ")
    genre_features = genre_features.loc[:, genre_features.sum().nlargest(25).index]

    numeric_features = data[
        ["is_movie", "duration_value", "release_year", "year_added", "month_added", "has_director"]
    ]
    features = pd.concat([numeric_features, country_features, genre_features], axis=1).astype(float)
    features = features.fillna(features.median())
    return features, data["rating"]


def train_and_evaluate(features: pd.DataFrame, target: pd.Series) -> None:
    """Train baseline and tuned models, then report the best tuned model."""
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    decision_tree = DecisionTreeClassifier(random_state=RANDOM_STATE).fit(x_train, y_train)
    random_forest = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1).fit(x_train, y_train)
    results = {
        "Decision Tree": accuracy_score(y_test, decision_tree.predict(x_test)),
        "Random Forest": accuracy_score(y_test, random_forest.predict(x_test)),
    }

    tuned_models = {
        "Decision Tree (Tuned)": GridSearchCV(
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                "max_depth": [5, 10, 15, 20],
                "min_samples_split": [2, 10, 20],
                "min_samples_leaf": [1, 5, 10],
            },
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
        ).fit(x_train, y_train),
        "Random Forest (Tuned)": GridSearchCV(
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "n_estimators": [100, 200],
                "max_depth": [10, 20, None],
                "min_samples_leaf": [1, 3],
            },
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
        ).fit(x_train, y_train),
    }

    for name, search in tuned_models.items():
        print(f"\nBest {name} parameters: {search.best_params_}")
        results[name] = accuracy_score(y_test, search.predict(x_test))

    print("\nModel comparison (test accuracy):")
    print(pd.Series(results).sort_values(ascending=False).round(4).to_string())

    best_name = max(tuned_models, key=lambda name: results[name])
    best_model = tuned_models[best_name].best_estimator_
    predictions = best_model.predict(x_test)
    print(f"\nBest model: {best_name}")
    print(classification_report(y_test, predictions, zero_division=0))

    figure, axis = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, ax=axis, xticks_rotation=45, cmap="Blues"
    )
    axis.set_title(f"Confusion matrix: {best_name}")
    figure.tight_layout()
    plt.show()


def main() -> None:
    """Load the dataset, display its rating distribution, and run classification."""
    data = pd.read_csv(DATASET_PATH)
    print(f"Dataset shape: {data.shape}")
    print("\nRating distribution:\n", data["rating"].value_counts())

    class_counts = data["rating"].value_counts()
    filtered_data = data[data["rating"].isin(class_counts[class_counts >= MIN_CLASS_SAMPLES].index)]
    filtered_data["rating"].value_counts().plot(
        kind="bar", title="Rating categories", figsize=(8, 4)
    )
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    features, target = prepare_features(data)
    train_and_evaluate(features, target)


if __name__ == "__main__":
    main()
