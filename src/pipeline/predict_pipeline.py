import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.exception import CustomException


@dataclass
class PredictPipelineConfig:
    project_root: Path = Path(__file__).resolve().parents[2]
    model_path: Path = project_root / "artifacts" / "model.pkl"
    column_rename_map: dict[str, str] = field(
        default_factory=lambda: {
            "race/ethnicity": "race_ethnicity",
            "parental level of education": "parental_level_of_education",
            "test preparation course": "test_preparation_course",
            "math score": "math_score",
            "reading score": "reading_score",
            "writing score": "writing_score",
        }
    )


def _load_pickle(path: Path) -> Any:
    import pickle

    with open(path, "rb") as file_obj:
        return pickle.load(file_obj)


class PredictPipeline:
	def __init__(self):
		self.config = PredictPipelineConfig()

	def predict(self, features: pd.DataFrame):
		try:
			if not self.config.model_path.exists():
				raise FileNotFoundError(
					f"Trained model not found at {self.config.model_path}. Run training first."
				)
			features = features.rename(columns=self.config.column_rename_map)
			model = _load_pickle(self.config.model_path)
			predictions = model.predict(features)
			return predictions
		except Exception as e:
			raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: float,
        writing_score: float,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
