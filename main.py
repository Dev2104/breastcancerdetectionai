"""
Main entry point for running ML pipeline (training + saving model).
Run this if you want to retrain the model manually.
"""

from src.train import main as train_main


def main():
    print("🚀 Starting training pipeline...\n")
    train_main()
    print("\n✅ Training pipeline completed successfully!")


if __name__ == "__main__":
    main()