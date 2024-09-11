import os

TRAIN_DATASET = os.path.join("project", "data/train.csv")
TEST_DATASET = os.path.join("project", "data/test.csv")
DEV_DATASET = os.path.join("project", "data/validation.csv")

ROOT = os.path.join("./")
OPENAI_CREDS = os.path.join("./", "creds/openai.json")

HATESPEECH_COL = "hatespeech"
COUNTERSPEECH_COL = "counterspeech"
INTENT_COL = "csType"
ID_COL = "id"
HATESPEECH_EXP_COLS = [
    "hatespeechOffensiveness",
    "targetGroup",
    "speakerIntent",
    "relevantPowerDynamics",
    "hatespeechImplication",
    "targetGroupEmotionalReaction",
    "targetGroupCognitiveReaction",
]

TEST_SIZE = 0.3
RANDOM_STATE = 1998
MODEL_NAME = "flant-t5-xxl"
MODEL_TYPE = "t5"


TRAINING_CONFIG = {
    'num_epochs': 30,
    'learning_rate': 1e-3,
    'batch_size': 16,
    'max_length': 1024,
    'model_name': 'google/flan-t5-small'
}