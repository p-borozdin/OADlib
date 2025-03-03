The script `optuna_examples/tuning.py` contains the demonstrative code of how one can organize the hyperparameter tuning process for a model with specific architecture with the means of `optuna` library.
For demonstrative purposes, the `LSTMFullyConnected` model is used in the example, but the main steps of the tuning process can be easily generalized to any other models. 
The model was trained on the dataset for 1010 ppm CH<sub>4</sub> (see `OADlib/data/` for more details).

```text
The structure of the optuna examples directory:
.
├── README.md
└── tuning.py  \\ The example of hyperparameters tuning with optuna library for LSTMFullyConnected model
```
To run the script on your machine, it is recommended to change the execution directory to `OADlib/optuna_examples/` and run the script from here. After the script has started to execute,
the next files/folders will be created in the execution directory (i.e. in the `OADlib/optuna_examples/`):
* `tuning.log` - the file to save logs during the tuning process. The logs include:
  * The process's id
  * The device the process is executed on (CPU or GPU)
  * The trial's number and corresponding values of hyperparameters
  * The exception's stack trace (if an exception was cought)
  * The completion of the process
  
  The code from `optuna_examples/tuning.py` that specifies the log file location and creates logger from logger manager is:
  ```python
  LOG_FILENAME = "tuning.log"
  logger_manager = LoggerManager()
  logger_manager.bind(open(LOG_FILENAME, "w", encoding='utf8'))
  
  logger = logger_manager.get_logger()
  ```
  You can edit this piece of code in order to specify other names and/or locations of your log files.

* `training/` - the directory to save the training results (e.g model's checkpoints, an optuna database and so on...). The directory will contain:
  * `{optuna_database_name}.db` - the database that optuna uses to save the results of hyperparameters tuning. The name of the database (`{optuna_database_name}`) is specified in lines:
  ```python
  # Specifying the directory to save training results
  TRAIN_DIR = "training/"
  ```
  ```python
  # Specifying the directory with dataset to use
  # We will use dataset for 1010 ppm CH4
  CONC = "CH4_1010_ppm"
  ```
  ```python
  # Specifying short model's name for serialization
  MODEL_SHORT_NAME = "LSTM_FC"  # Stands for "LSTM Fully-Connected"
  ```
  ```python
  STUDY_NAME = f'{MODEL_SHORT_NAME}_{CONC}'
  storage_name = f'sqlite:///{TRAIN_DIR}/{STUDY_NAME}.db'
  ```
  
  * `models/` - the directory where the summary of all the trials as well as the best models will be saved. The structure of the directory is the next:
    * `trials/` - the subdirectory used by tensorboard's `SummaryWriter` to save the summary about training process for each optuna's trial. The directory's name, location and contents are specified
    by the next pieces of code:
    ```python
    # Specifying the directory to save training results
    TRAIN_DIR = "training/"
    
    # Specifying the tensorflow summary writer's directory
    WRITER_DIR = f"{TRAIN_DIR}/models/trials"
    ```
    ```python
    # Creating summary writers
    train_writer = SummaryWriter(f'{WRITER_DIR}/{trial.number}/train')
    valid_writer = SummaryWriter(f'{WRITER_DIR}/{trial.number}/valid')
    test_writer = SummaryWriter(f'{WRITER_DIR}/{trial.number}/test')
    ```

    In addition, the model's checkpoints for each epoch in each trial are saved in that directory using the next code:
    ```python
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
    ```

    * `{model_name}.pt` - the current best model. When the current best result (mean absolute error (MAE) on validation set in the given demonstrative script) is exceeded by a model, the model is saved in a *.pt format.
    The code dealing with it is:
    ```python
    model_name = MODEL_SHORT_NAME + f"_{trial.number}trial" + \
      f"_{best_epoch}ep_MAE-{round(float(best_epoch_train_mae), 2)}" + \
      f"_{round(float(best_epoch_valid_mae), 2)}" + \
      f"_{round(float(best_epoch_test_mae), 2)}"
    torch.save(model, f"{TRAIN_DIR}/models/{model_name}.pt")
    ```
    The `model_name` contains information about the trial and epoch at which the model performed best, as well as the best metrics (i.e. MAE) that were demonstrated by the model at that particular point.

The code from `OADlib/optuna_example/tuning.py` can be copied and modified in order to fit the requirements of your particular model, dataset or set of hyperparameters.
