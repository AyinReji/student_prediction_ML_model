from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class RulesConfig(BaseModel):
    low_series_avg: float = Field(default=18.0, description="Threshold for low average series score out of 40")
    decline_points: float = Field(default=4.0, description="Points decline from series 1 to series 2")
    min_attendance: float = Field(default=75.0, description="Minimum attendance percentage")
    min_study_hours: float = Field(default=2.0, description="Minimum study hours per day")
    max_social_hours: float = Field(default=4.0, description="Maximum social media hours per day")


class PriorityWeightsConfig(BaseModel):
    risk: float = Field(default=0.7, ge=0.0, le=1.0)
    decline: float = Field(default=0.2, ge=0.0, le=1.0)
    flags: float = Field(default=0.1, ge=0.0, le=1.0)


class RiskBadgesConfig(BaseModel):
    high: float = Field(default=0.6, ge=0.0, le=1.0)
    medium: float = Field(default=0.3, ge=0.0, le=1.0)


class SplitConfig(BaseModel):
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    seed: int = Field(default=42)


class Settings(BaseModel):
    target_score: float = Field(default=40.0)
    risk_margin: float = Field(default=3.0)
    capacity: int = Field(default=25, ge=1)
    interval_level: float = Field(default=0.90, gt=0.0, lt=1.0)
    rules: RulesConfig = Field(default_factory=RulesConfig)
    priority_weights: PriorityWeightsConfig = Field(default_factory=PriorityWeightsConfig)
    risk_badges: RiskBadgesConfig = Field(default_factory=RiskBadgesConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    cv_folds: int = Field(default=5, ge=2)


DEFAULT_CONFIG_PATH = Path("config/settings.yaml")


def load_settings(path: str | Path = DEFAULT_CONFIG_PATH) -> Settings:
    path = Path(path)
    if not path.exists():
        settings = Settings()
        save_settings(settings, path)
        return settings

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Settings(**data)


def save_settings(settings: Settings, path: str | Path = DEFAULT_CONFIG_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(settings.model_dump(), f, sort_keys=False)
