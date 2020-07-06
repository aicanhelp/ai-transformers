from tqdm import tqdm
from transformers import PreTrainedModel

from .trainer_base import *
from .trainer_dataloaders import TaskTrainerDataLoaders
from .trainer_metrics import TaskTrainerMetrics


@configclass
class TrainerEvaluatorConfig():
    pass


class TaskTrainerPredictor():
    def __init__(self, trainer_env: TrainerEnv, model: PreTrainedModel,
                 data_loaders: TaskTrainerDataLoaders):
        self._env = trainer_env
        self._model = model
        self._data_loaders = data_loaders

    def predict(self, test_dataset: Dataset) -> TaskPredictionOutput:
        test_dataloader = self._data_loaders.get_test_dataloader(test_dataset)
        return self.prediction_loop(test_dataloader, description="Prediction")

    def _prepare_model(self):
        # multi-gpu eval
        if self._env.n_gpu > 1 and not isinstance(self._model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(self._model)
        else:
            model = self._model
        model.to(self._env.device)
        return model

    def _get_batch_size(self, dataloader: DataLoader):
        if self._env.is_tpu_available():
            return dataloader._loader._loader.batch_size
        else:
            return dataloader.batch_size

    def prediction_loop(
            self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
    ) -> TaskPredictionOutput:
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else prediction_loss_only
        model = self._prepare_model()
        batch_size = self._get_batch_size(dataloader)

        log.info("***** Running %s *****", description)
        log.info("  Num examples = %d", self._env.num_examples(dataloader))
        log.info("  Batch size = %d", batch_size)

        eval_losses: List[float] = []
        preds: np.ndarray = None
        label_ids: np.ndarray = None
        guids = []

        model.eval()

        for inputs in tqdm(dataloader, desc=description):

            has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels"])

            for k, v in inputs.items():
                if k != 'guid':
                    inputs[k] = v.to(self._env.device)
                else:
                    guids.extend(inputs.pop('guid'))

            with torch.no_grad():
                outputs = model(**inputs)
                if has_labels:
                    step_eval_loss, logits = outputs[:2]
                    eval_losses += [step_eval_loss.mean().item()]
                else:
                    logits = outputs[0]

            if not prediction_loss_only:
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                if inputs.get("labels") is not None:
                    if label_ids is None:
                        label_ids = inputs["labels"].detach().cpu().numpy()
                    else:
                        label_ids = np.append(label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        if self._env.is_tpu_available() and preds is not None and label_ids is not None:
            # tpu-comment: Get all predictions and labels from all worker shards of eval dataset
            preds, label_ids = self._env.get_tpu_eval(preds, label_ids)

        return TaskPredictionOutput(guids=guids, predictions=preds, label_ids=label_ids, eval_losses=eval_losses)


class TaskTrainerEvaluator():
    def __init__(self, trainer_env: TrainerEnv,
                 config: TrainerEvaluatorConfig,
                 predictor: TaskTrainerPredictor,
                 compute_metrics: TaskTrainerMetrics = None):
        self._env = trainer_env
        self.config = config
        self._predictor = predictor
        self._compute_metrics = compute_metrics

    def evaluate(self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None):
        p = self._predictor.prediction_loop(dataloader, description, prediction_loss_only)
        if self._compute_metrics: self._compute_metrics.eval_metrics(p)
        return p
