import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Production CLI tool for comparing classification models on a CSV dataset."
    )

    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to the input CSV dataset."
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Directory where results and plots will be saved. Default: ./output"
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds. Default: 5"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the dataset and print configuration without training models."
    )

    return parser.parse_args()


def load_data(data_path: str) -> pd.DataFrame:
    path = Path(data_path)

    if not path.exists():
        logging.error("Data file not found: %s", data_path)
        sys.exit(1)

    try:
        df = pd.read_csv(path)
        logging.info("Loaded data from %s with shape %s", data_path, df.shape)
        return df
    except Exception as exc:
        logging.error("Failed to load data: %s", exc)
        sys.exit(1)


def validate_data(df: pd.DataFrame) -> str:
    if df.empty:
        logging.error("The dataset is empty.")
        sys.exit(1)

    if df.shape[1] < 2:
        logging.error("Dataset must contain at least one feature column and one target column.")
        sys.exit(1)

    if "target" in df.columns:
        target_col = "target"
    else:
        target_col = df.columns[-1]
        logging.warning("No 'target' column found. Using the last column as target: %s", target_col)

    if df[target_col].isna().all():
        logging.error("Target column contains only missing values.")
        sys.exit(1)

    if df[target_col].nunique(dropna=True) < 2:
        logging.error("Target column must have at least two classes.")
        sys.exit(1)

    logging.info("Validation passed successfully.")
    logging.info("Target column: %s", target_col)
    logging.info("Class distribution:\n%s", df[target_col].value_counts(dropna=False))

    return target_col


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


def get_models(random_seed: int) -> dict:
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=random_seed),
        "DecisionTree": DecisionTreeClassifier(random_state=random_seed),
        "RandomForest": RandomForestClassifier(random_state=random_seed)
    }


def train_and_evaluate(
    df: pd.DataFrame,
    target_col: str,
    n_folds: int,
    random_seed: int
):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    preprocessor = build_preprocessor(X)
    models = get_models(random_seed)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    results = []
    detailed_scores = []

    for model_name, model in models.items():
        logging.info("Evaluating model: %s", model_name)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring="accuracy"
        )

        results.append({
            "model": model_name,
            "mean_accuracy": scores.mean(),
            "std_accuracy": scores.std()
        })

        for fold_idx, score in enumerate(scores, start=1):
            detailed_scores.append({
                "model": model_name,
                "fold": fold_idx,
                "accuracy": score
            })

        logging.info(
            "%s -> mean accuracy: %.4f | std: %.4f",
            model_name,
            scores.mean(),
            scores.std()
        )

    results_df = pd.DataFrame(results).sort_values(by="mean_accuracy", ascending=False)
    detailed_scores_df = pd.DataFrame(detailed_scores)

    return results_df, detailed_scores_df


def save_results(results_df: pd.DataFrame, detailed_scores_df: pd.DataFrame, output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_path = output_path / "metrics.csv"
    cv_scores_path = output_path / "cv_scores.csv"
    summary_path = output_path / "summary.txt"
    plot_path = output_path / "model_comparison.png"

    results_df.to_csv(metrics_path, index=False)
    detailed_scores_df.to_csv(cv_scores_path, index=False)

    best_model = results_df.iloc[0]

    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("Model Comparison Summary\n")
        file.write("========================\n\n")
        file.write(results_df.to_string(index=False))
        file.write("\n\n")
        file.write(
            f"Best model: {best_model['model']} "
            f"(Mean Accuracy: {best_model['mean_accuracy']:.4f}, "
            f"Std Accuracy: {best_model['std_accuracy']:.4f})\n"
        )

    plt.figure(figsize=(8, 5))
    plt.bar(results_df["model"], results_df["mean_accuracy"])
    plt.xlabel("Model")
    plt.ylabel("Mean Accuracy")
    plt.title("Model Comparison")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    logging.info("Saved metrics to %s", metrics_path)
    logging.info("Saved fold scores to %s", cv_scores_path)
    logging.info("Saved summary to %s", summary_path)
    logging.info("Saved plot to %s", plot_path)


def main() -> None:
    setup_logging()
    args = parse_args()

    logging.info("Pipeline configuration:")
    logging.info("Data path: %s", args.data_path)
    logging.info("Output directory: %s", args.output_dir)
    logging.info("Number of folds: %d", args.n_folds)
    logging.info("Random seed: %d", args.random_seed)
    logging.info("Dry run: %s", args.dry_run)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data_path)
    target_col = validate_data(df)

    if args.dry_run:
        logging.info("Dry run completed successfully. No models were trained.")
        return

    results_df, detailed_scores_df = train_and_evaluate(
        df=df,
        target_col=target_col,
        n_folds=args.n_folds,
        random_seed=args.random_seed
    )

    save_results(results_df, detailed_scores_df, args.output_dir)
    logging.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()