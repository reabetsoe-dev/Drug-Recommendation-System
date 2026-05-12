"""
Shared configuration for frontend, backend, and AI service.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path = Path(__file__).resolve().parent.parent

    backend_host: str = "127.0.0.1"
    backend_port: int = 8000
    ai_service_host: str = "127.0.0.1"
    ai_service_port: int = 5000

    request_timeout_seconds: int = 60

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def backend_base_url(self) -> str:
        return f"http://{self.backend_host}:{self.backend_port}"

    @property
    def ai_service_base_url(self) -> str:
        return f"http://{self.ai_service_host}:{self.ai_service_port}"

    @property
    def model_path(self) -> Path:
        return self.models_dir / "model.pkl"

    @property
    def vectorizer_path(self) -> Path:
        return self.models_dir / "vectorizer.pkl"

    @property
    def label_encoder_path(self) -> Path:
        return self.models_dir / "label_encoder.pkl"

    @property
    def train_dataset_path(self) -> Path:
        return self.data_dir / "drugsComTrain_raw.csv"

    @property
    def test_dataset_path(self) -> Path:
        return self.data_dir / "drugsComTest_raw.csv"

    @property
    def nltk_data_dir(self) -> Path:
        return self.project_root / "nltk_data"

    @property
    def mplconfig_dir(self) -> Path:
        return self.project_root / ".mplconfig"


settings = Settings()
