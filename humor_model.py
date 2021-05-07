import wandb
import torch
import gc
import prettyprinter
from prettyprinter import pprint
from statistics import mean


from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
import logging
from simpletransformers.classification import (
    ClassificationModel,
    ClassificationArgs
)


prettyprinter.install_extras(include=["dataclasses",], warn_on_error=True)


#train_df['labels'] = (train_df['labels'] == 1.0).astype(int)

train_df, eval_df = train_test_split(train_df, test_size=0.2)



sweep_result = pd.read_csv(prefix + 'sweep.csv')
best_params = sweep_result.to_dict()


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

model_args = ClassificationArgs()
model_args.eval_batch_size = 4
model_args.evaluate_during_training = True
model_args.evaluate_during_training_silent = False
model_args.evaluate_during_training_steps = 1000
model_args.learning_rate = 3.84e-5
model_args.manual_seed = 4
model_args.max_seq_length = 256
model_args.multiprocessing_chunksize = 5000
model_args.no_cache = True
#model_args.no_save = True
model_args.num_train_epochs = 10
model_args.overwrite_output_dir = True
model_args.reprocess_input_data = True
model_args.train_batch_size = 4
#model_args.regression = True
model_args.gradient_accumulation_steps = 2
model_args.train_custom_parameters_only = False
model_args.save_eval_checkpoints = False
model_args.save_model_every_epoch = False
model_args.labels_list = [1, 0]
model_args.output_dir = "tuned_output"
model_args.best_model_dir = "tuned_output/best_model"
model_args.wandb_project = "post_valuation"
model_args.wandb_kwargs = {"name": "post-eval"}

layer_params = []
param_groups = []
cleaned_args = {}

for key, value in best_params.items():
    if key.startswith("layer_"):
        layer_keys = key.split("_")[-1]
        start_layer = int(layer_keys.split("-")[0])
        end_layer = int(layer_keys.split("-")[-1])
        for layer_key in range(start_layer, end_layer):
            layer_params.append(
                {"layer": layer_key, "lr": value[0],}
            )
    elif key.startswith("params_"):
        params_key = key.split("_")[-1]
        param_groups.append(
            {
                "params": [params_key],
                "lr": value[0],
                "weight_decay": model_args.weight_decay
                if "bias" not in params_key
                else 0.0,
            }
        )
    elif key == "num_train_epochs":
        cleaned_args[key] = value[0]
    elif key == "learning_rate":
        cleaned_args[key] = value[0]
        

cleaned_args["custom_layer_parameters"] = layer_params
cleaned_args["custom_parameter_groups"] = param_groups
model_args.update_from_dict(cleaned_args)

pprint(model_args)



# Create a TransformerModel
model = ClassificationModel(
        "roberta", "roberta-large", use_cuda=True, args=model_args)

# Train the model
model.train_model(
    train_df,
    eval_df=eval_df,
    accuracy=lambda truth, predictions: accuracy_score(
            truth, [round(p) for p in predictions]
    ),
)