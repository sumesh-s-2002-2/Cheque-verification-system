import torch
import mlflow
from loguru import logger
from evaluation.contrastive_loss import ContrastiveLoss
from src.evaluation.evaluate_model import evaluate
from src.evaluation.evaluate_result import evaluate_classification,find_best_threshold
from src.evaluation.metrics import log_confusion_matrix
from src.utils.config_loader import model_cfg,training_cfg


def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    epochs,
    lr,
    margin,
    save_path,
    patience=5,
    min_delta=1e-4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for param in model.cnn.parameters():
        param.requires_grad = False
    model.cnn.eval()
    logger.info("CNN backbone frozen")

    criterion = ContrastiveLoss(margin=margin)

    optimizer = torch.optim.AdamW(
        model.fc.parameters(),
        lr=float(lr)
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.fc.train()
        total_train_loss = 0.0

        logger.info(f"Epoch {epoch+1}/{epochs}")

        for img1, img2, label in train_loader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            emb1, emb2 = model(img1, img2)
            loss = criterion(emb1, emb2, label)
            loss.backward()
            logger.info(f"Loss per iteration : {loss}")
            optimizer.step()

            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)

        val_loss = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device
        )

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

        logger.info(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss - float(min_delta):
            best_val_loss = val_loss
            epochs_without_improvement = 0

            torch.save(model.state_dict(), save_path)
            mlflow.log_artifact(save_path, artifact_path="best_model")
            logger.success("Best model updated")

        else:
            epochs_without_improvement += 1
            logger.warning(
                f"No improvement for {epochs_without_improvement}/{patience} epochs"
            )

            if epochs_without_improvement >= patience:
                logger.warning("Early stopping triggered")
                break

    logger.info("Loading best validation model for testing")
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.to(device)

    test_loss = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device
    )

    mlflow.log_metric("test_loss", test_loss)
    logger.success(f" Final Test Loss: {test_loss:.4f}")

    logger.info(f"Finding the best threshold for margin : {margin}....")
    best_threshold, best_f1 = find_best_threshold(model, val_loader, device, training_cfg.model.similarity_threshold)
    logger.info(f"Best threshold  : {best_threshold}, best F1: {best_f1}")

    metrics = evaluate_classification(
        model=model,
        dataloader=test_loader,
        device=device,
        threshold=best_threshold
    )

    mlflow.log_metric("test_accuracy", metrics["accuracy"])
    mlflow.log_metric("test_precision", metrics["precision"])
    mlflow.log_metric("test_recall", metrics["recall"])
    mlflow.log_metric("test_f1_score", metrics["f1"])
 
    cm = metrics["confusion_matrix"]
    logger.info(f"Confusion Matrix:\n{cm}")
    log_confusion_matrix(cm)