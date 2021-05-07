
import wandb
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score
from sklearn.model_selection import train_test_split
import logging
from simpletransformers.classification import (
    ClassificationModel,
    ClassificationArgs
)


prefix = 'data/'
train_df = pd.read_csv(prefix + 'train.csv')

train_df = pd.DataFrame({
    'text': train_df['text'].replace(r'\n', ' ', regex=True),
    'labels':train_df['humor_rating']
})

train_df['labels'] = (train_df['labels'] == 1.0).astype(int)

train_df, eval_df = train_test_split(train_df, test_size=0.2)


layer_parameters = {f"layer_{i}-{i + 6}": {"min": 0.0, "max": 5e-5} for i in range(0, 24, 6)}

sweep_config = {
    "name": "ratings-sweep-4",
    "method": "bayes",
    "metric": {"name": "rmse", "goal": "minimize"},

    "parameters": {
        "num_train_epochs": {"min": 6, "max": 40},
        "learning_rate": { "min": 0, "max": 1e-4},
        "params_classifier.dense.weight": {"min": 0, "max": 1e-3},
        "params_classifier.dense.bias": {"min": 0, "max": 1e-3},
        "params_classifier.out_proj.weight": {"min": 0, "max": 1e-3},
        "params_classifier.out_proj.bias": {"min": 0, "max": 1e-3},
        **layer_parameters,
    },
    "early_terminate": {"type": "hyperband", "min_iter": 3,},
}

sweep_id = wandb.sweep(sweep_config, project="humor_rt")


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

model_args = ClassificationArgs()
model_args.evaluate_during_training = True
model_args.evaluate_during_training_silent = False
model_args.evaluate_during_training_steps = 1000
model_args.eval_batch_size = 4
model_args.learning_rate = 1e-7
model_args.use_cached_eval_features = True
model_args.manual_seed = 4
model_args.max_seq_length = 256
model_args.multiprocessing_chunksize = 5000
model_args.no_cache = True
model_args.no_save = True
model_args.regression = True
#model_args.num_labels = 1
model_args.num_train_epochs = 10
model_args.overwrite_output_dir = True
model_args.reprocess_input_data = True
model_args.train_batch_size = 4
model_args.gradient_accumulation_steps = 2
model_args.train_custom_parameters_only = False
model_args.wandb_project = "humor_rt"

def train():
    # Initialize a new wandb run
    wandb.init()

   
   
    #print(wandb.config)
    # Get sweep hyperparameters
    args = {
        key: value
        for (key, value) in wandb.config.items()
        if key != "_wandb"
    }
    #print(args)

    # Extracting the hyperparameter values
    cleaned_args = {}
    layer_params = []
    param_groups = []
    for key, value in args.items():
        if key.startswith("layer_"):
            # These are layer parameters
            layer_keys = key.split("_")[-1]

            # Get the start and end layers
            start_layer = int(layer_keys.split("-")[0])
            end_layer = int(layer_keys.split("-")[-1])

            # Add each layer and its value to the list of layer parameters
            for layer_key in range(start_layer, end_layer):
                layer_params.append(
                    {"layer": layer_key, "lr": value,}
                )
        elif key.startswith("params_"):
            # These are parameter groups (classifier)
            params_key = key.split("_")[-1]
            param_groups.append(
                {
                    "params": [params_key],
                    "lr": value,
                    "weight_decay": model_args.weight_decay
                    if "bias" not in params_key
                    else 0.0,
                }
            )
        else:
            # Other hyperparameters (single value)
            cleaned_args[key] = value
    cleaned_args["custom_layer_parameters"] = layer_params
    cleaned_args["custom_parameter_groups"] = param_groups

    #print(cleaned_args)

    model_args.update_from_dict(cleaned_args)

    # Create a TransformerModel
    model = ClassificationModel(
        "roberta", "roberta-large", num_labels=1, use_cuda=True, args=model_args)

    # Train the model
    model.train_model(
        train_df,
        eval_df=eval_df,
        f1=lambda truth, predictions: mean_squared_error(
            truth, [round(p) for p in predictions], false
        ),
    )


    # Sync wandb
    wandb.join()


wandb.agent(sweep_id, train)