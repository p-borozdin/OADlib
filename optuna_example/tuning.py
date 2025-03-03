""" The module provides an example of how to tune hyperparameters
of LSTM-based models using `optuna` python liibrary.
The LSTMFullyConnected model is used for the demonstration.
"""
import os
import gc
import sys
import errno

import optuna
from optuna.pruners import HyperbandPruner
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics.regression import MeanAbsoluteError

# Adding OADlib directory to path
OADLIB_DIR = f"{os.path.abspath('')}/../"
sys.path.append(OADLIB_DIR)
sys.path.append(f'{OADLIB_DIR}/../')

from OADlib.logging.logger_manager import LoggerManager
from OADlib.dataset.oad_dataset import OADDataset
from OADlib.models.lstm_fully_connected import LSTMFullyConnected

# Specifying the visible GPU and torch.device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Creating logger to log intermediate training results
# The log file will be created in the same directory
LOG_FILENAME = "tuning.log"
logger_manager = LoggerManager()
logger_manager.bind(open(LOG_FILENAME, "w", encoding='utf8'))

logger = logger_manager.get_logger()
logger.info(f"*** Procces started with id {os.getpid()} ***")
DEVICE_NAME = ("CPU" if device == torch.device('cpu')
               else "CUDA " + os.environ['CUDA_VISIBLE_DEVICES'])
logger.info(f"Device: {DEVICE_NAME}\n")

# Specifying the directory to save training results
TRAIN_DIR = "training/"

# Specifying the tensorflow summary writer's directory
WRITER_DIR = f"{TRAIN_DIR}/models/trials"

# Specifying the directory with dataset to use
# We will use dataset for 1010 ppm CH4
CONC = "CH4_1010_ppm"
DATASET_DIR = f"{OADLIB_DIR}/data/datasets/{CONC}"

# Defining the metric and loss used for training
mae_metric_object = MeanAbsoluteError().to(device)
SUMMARY_METRIC_NAME = "MAE metric"
loss_fn = nn.MSELoss()
SUMMARY_LOSS_NAME = "MSE loss"

# Declaring the best MAE reached on validation
global_best_mae_valid = torch.inf

EPOCHS = 200  # Total number of training epochs
TRIALS = 400  # Total number of optuna trials
PATIENCE = 20  # Patience for optuna's pruner

# Specifying short model's name for serialization
MODEL_SHORT_NAME = "LSTM_FC"  # Stands for "LSTM Fully-Connected"

# Fully-connected layers' activations to tune
activations_matching = {
    "ReLU": nn.ReLU(),
    "Sigmoid": nn.Sigmoid(),
    "SiLU": nn.SiLU(),
    "Hardshrink": nn.Hardshrink(),
    "Tanh": nn.Tanh(),
    "LogSigmoid": nn.LogSigmoid()
    }


def restore(y: torch.Tensor) -> torch.Tensor:
    """ Restores normalized target values of resonance frequency
    multiplying it by a factor of `1000`
    (see `OADlib/data/preprocess.ipynb` section `3. Data Normalization`
    for more details)

    Args:
        y (torch.Tensor): normalized values

    Returns:
        torch.Tensor: restored (initial) values
    """
    factor = 1000
    return y * factor


def objective(trial: optuna.Trial) -> float:
    """ The objective function for optuna's tuning.

    Args:
        trial (optuna.Trial): optuna's trial

    Raises:
        optuna.TrialPruned: the trial was pruned

    Returns:
        float: best MAE metric reached on validation
            during the trial's training process
    """
    global global_best_mae_valid

    # Best MAE metrics achieved during the training process
    best_epoch_train_mae = torch.inf
    best_epoch_valid_mae = torch.inf
    best_epoch_test_mae = torch.inf
    best_epoch = -1

    # Specifying hyperparameters to tune
    seq_len = trial.suggest_int("seq_len", 3, 8)
    hidden_size = trial.suggest_int("hidden_size", 2, 7)
    lr = trial.suggest_float("learning_rate", 1E-3, 1E-1, log=True)
    factor = trial.suggest_float("scheduler_factor", 0.2, 0.9)
    batch_size = int(2 ** trial.suggest_int("batch_size", 2, 4))
    activation_func_name = trial.suggest_categorical(
        "activation",
        list(activations_matching.keys())
        )
    hidden_layer_size = trial.suggest_int("hidden_layer_size", 2, 3)

    hyperparameter_string = '\n\t' + '\n\t'.join([': '.join(
        [param,
         str(trial.params[param])]
         ) for param in trial.params])
    logger.info('-' * 50 +
                f"\nTrial # {trial.number} started with hyperparameters: " +
                hyperparameter_string)

    # Loading datasets
    dataset_file = f"{DATASET_DIR}/dataset_{seq_len}_points.pth"
    predictors = ["Temp", "dT/dt"]
    target = "Fres"
    train_set = OADDataset(
        path_to_file=dataset_file,
        predictors=predictors,
        target=target,
        mode='train',
        device=device
    )
    valid_set = OADDataset(
        path_to_file=dataset_file,
        predictors=predictors,
        target=target,
        mode='valid',
        device=device
    )
    test_set = OADDataset(
        path_to_file=dataset_file,
        predictors=predictors,
        target=target,
        mode='test',
        device=device
    )

    # Creating torch DataLoaders
    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4
        )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4
        )
    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4
        )

    train_len = len(train_set)
    valid_len = len(valid_set)
    test_len = len(test_set)

    # Creating summary writers
    train_writer = SummaryWriter(f'{WRITER_DIR}/{trial.number}/train')
    valid_writer = SummaryWriter(f'{WRITER_DIR}/{trial.number}/valid')
    test_writer = SummaryWriter(f'{WRITER_DIR}/{trial.number}/test')

    train_writer.add_text("Hyperparameters", hyperparameter_string)

    model = LSTMFullyConnected(
        input_size=len(predictors),
        hidden_size=hidden_size,
        hidden_layer_size=hidden_layer_size,
        activation_func=activations_matching[activation_func_name],
        device=device
    )
    model.type(torch.float32)
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
        )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=factor,
        patience=5,
        verbose=True
        )

    for epoch in range(EPOCHS):

        # Training
        model.train()
        tot_train_loss = 0
        for (x, y) in train_dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = torch.flatten(model(x))
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

            tot_train_loss += loss.item() * x.size(0)
            mae_metric_object.update(restore(y_pred), restore(y))

            del x, y, y_pred, loss
            torch.cuda.empty_cache()
            gc.collect()

        train_loss = tot_train_loss / train_len
        train_mae = float(mae_metric_object.compute())
        mae_metric_object.reset()

        train_writer.add_scalar(SUMMARY_LOSS_NAME, train_loss, epoch)
        train_writer.add_scalar(SUMMARY_METRIC_NAME, train_mae, epoch)

        del train_loss, tot_train_loss
        torch.cuda.empty_cache()
        gc.collect()

        # Evaluation
        model.eval()
        with torch.no_grad():

            # Validation
            tot_valid_loss = 0
            for (x, y) in valid_dataloader:
                x, y = x.to(device), y.to(device)
                y_pred = torch.flatten(model(x))
                loss = loss_fn(y_pred, y)

                tot_valid_loss += loss.item() * x.size(0)
                mae_metric_object.update(restore(y_pred), restore(y))

                del x, y, y_pred, loss
                torch.cuda.empty_cache()
                gc.collect()

            valid_loss = tot_valid_loss / valid_len
            valid_mae = float(mae_metric_object.compute())
            mae_metric_object.reset()

            valid_writer.add_scalar(SUMMARY_LOSS_NAME, valid_loss, epoch)
            valid_writer.add_scalar(SUMMARY_METRIC_NAME, valid_mae, epoch)

            del tot_valid_loss
            torch.cuda.empty_cache()
            gc.collect()

            # Test
            tot_test_loss = 0
            for (x, y) in test_dataloader:
                x, y = x.to(device), y.to(device)
                y_pred = torch.flatten(model(x))
                loss = loss_fn(y_pred, y)

                tot_test_loss += loss.item() * x.size(0)
                mae_metric_object.update(restore(y_pred), restore(y))

                del x, y, y_pred, loss
                torch.cuda.empty_cache()
                gc.collect()

            test_loss = tot_test_loss / test_len
            test_mae = float(mae_metric_object.compute())
            mae_metric_object.reset()

            test_writer.add_scalar(SUMMARY_LOSS_NAME, test_loss, epoch)
            test_writer.add_scalar(SUMMARY_METRIC_NAME, test_mae, epoch)

            del test_loss, tot_test_loss
            torch.cuda.empty_cache()
            gc.collect()

            # Updating best epoch metrics
            if best_epoch_valid_mae > valid_mae:
                best_epoch_train_mae = train_mae
                best_epoch_valid_mae = valid_mae
                best_epoch_test_mae = test_mae
                best_epoch = epoch

            # Saving checkpoint
            checkpoint_name = MODEL_SHORT_NAME + f"_{trial.number}trial" + \
                f"_{epoch}ep_MAE-{round(train_mae, 2)}" + \
                f"_{round(valid_mae, 2)}_{round(test_mae, 2)}"
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_loss': valid_loss,
                    'train_mae': train_mae,
                    'valid_mae': valid_mae,
                    'test_mae': test_mae
                    },
                f'{WRITER_DIR}/{trial.number}/{checkpoint_name}.pth'
                )

        train_writer.add_scalar(
            "Learning rate",
            optimizer.param_groups[0]['lr'],
            epoch)
        scheduler.step(valid_loss)

        # Pruning
        if epoch >= PATIENCE:
            trial.report(float(best_epoch_valid_mae), epoch)

            if trial.should_prune():
                logger.info("Trial was pruned with best epoch valid MAE: " +
                            f"{round(float(best_epoch_valid_mae), 2)}")
                train_writer.close()
                valid_writer.close()
                test_writer.close()

                del train_writer, valid_writer, test_writer
                del model, train_dataloader, valid_dataloader, test_dataloader
                torch.cuda.empty_cache()
                gc.collect()

                raise optuna.TrialPruned()

    train_writer.close()
    valid_writer.close()
    test_writer.close()

    del train_writer, valid_writer, test_writer
    del train_dataloader, valid_dataloader, test_dataloader
    torch.cuda.empty_cache()
    gc.collect()

    if global_best_mae_valid > best_epoch_valid_mae:
        global_best_mae_valid = best_epoch_valid_mae
        model_name = MODEL_SHORT_NAME + f"_{trial.number}trial" + \
            f"_{best_epoch}ep_MAE-{round(float(best_epoch_train_mae), 2)}" + \
            f"_{round(float(best_epoch_valid_mae), 2)}" + \
            f"_{round(float(best_epoch_test_mae), 2)}"
        torch.save(model, f"{TRAIN_DIR}/models/{model_name}.pt")

    return float(best_epoch_valid_mae)


try:
    os.mkdir(TRAIN_DIR)
except OSError as err:
    if err.errno != errno.EEXIST:
        raise

try:
    STUDY_NAME = f'{MODEL_SHORT_NAME}_{CONC}'
    storage_name = f'sqlite:///{TRAIN_DIR}/{STUDY_NAME}.db'
    study = optuna.create_study(
        direction='minimize',
        pruner=HyperbandPruner(min_resource=PATIENCE),
        storage=storage_name,
        study_name=STUDY_NAME,
        load_if_exists=True
        )
    study.optimize(objective, TRIALS)
except RuntimeError:
    logger.exception("Exception was cought:")

logger.info("\n*** Process finished ***")
logger_manager.close_file()
