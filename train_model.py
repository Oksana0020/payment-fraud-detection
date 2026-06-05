from app.model import train_and_save_model


if __name__ == "__main__":
    bundle = train_and_save_model()
    print(
        {
            "status": "trained",
            "model_version": bundle.model_version,
            "roc_auc": round(bundle.roc_auc, 4),
        }
    )
