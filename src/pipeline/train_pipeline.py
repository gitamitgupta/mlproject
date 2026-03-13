import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class TrainPipelineConfig:
	project_root: Path = Path(__file__).resolve().parents[2]
	train_data_path: Path = project_root / "artifacts" / "train.csv"
	test_data_path: Path = project_root / "artifacts" / "test.csv"
	model_output_path: Path = project_root / "artifacts" / "model.pkl"


class TrainPipeline:
    COLUMN_RENAME_MAP = {
        "race/ethnicity": "race_ethnicity",
        "parental level of education": "parental_level_of_education",
        "test preparation course": "test_preparation_course",
        "math score": "math_score",
        "reading score": "reading_score",
        "writing score": "writing_score",
    }

    def __init__(self):
        self.config = TrainPipelineConfig()
        self.target_column = "math_score"
        self.numerical_features = ["reading_score", "writing_score"]
        self.categorical_features = [
            "gender",
            "race_ethnicity",
            "parental_level_of_education",
            "lunch",
            "test_preparation_course",
        ]

    def _build_preprocessor(self) -> ColumnTransformer:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, self.numerical_features),
                ("cat", categorical_pipeline, self.categorical_features),
            ]
        )
        return preprocessor

    def _build_model_pipeline(self) -> Pipeline:
        preprocessor = self._build_preprocessor()
        model = RandomForestRegressor(n_estimators=300, random_state=42)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        return pipeline

    def run(self) -> float:
        try:
            train_df = pd.read_csv(self.config.train_data_path)
            test_df = pd.read_csv(self.config.test_data_path)
            train_df = train_df.rename(columns=self.COLUMN_RENAME_MAP)
            test_df = test_df.rename(columns=self.COLUMN_RENAME_MAP)
            logging.info("Loaded train and test datasets for training")

            x_train = train_df.drop(columns=[self.target_column])
            y_train = train_df[self.target_column]
            x_test = test_df.drop(columns=[self.target_column])
            y_test = test_df[self.target_column]

            model_pipeline = self._build_model_pipeline()
            model_pipeline.fit(x_train, y_train)
            logging.info("Model pipeline training completed")

            predictions = model_pipeline.predict(x_test)
            score = r2_score(y_test, predictions)
            logging.info(f"Model R2 score on test set: {score:.4f}")

            save_object(str(self.config.model_output_path), model_pipeline)
            logging.info(f"Saved trained model pipeline to {self.config.model_output_path}")
            return float(score)

        except Exception as e:
            raise CustomException(e, sys)


def run_training_pipeline() -> float:
    trainer = TrainPipeline()
    return trainer.run()


if __name__ == "__main__":
    metric = run_training_pipeline()
    print(f"Training completed. Test R2: {metric:.4f}")
