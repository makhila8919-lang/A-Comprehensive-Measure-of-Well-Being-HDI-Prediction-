import os

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

SECRET_KEY = os.environ.get("SECRET_KEY", "HDI_PROJECT_SECRET_KEY")

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "Human_Development_Data.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "hdi_model.pkl"
)

ENCODER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "label_encoder.pkl"
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)
