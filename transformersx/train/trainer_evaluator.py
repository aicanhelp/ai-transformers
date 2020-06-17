from .trainer_base import *
from .trainer_dataloaders import TaskTrainerDataLoaders


class TaskTrainerPredictor(TrainerBase):
    def __init__(self, args: TaskTrainingArguments, model: PreTrainedModel,
                 data_loaders: TaskTrainerDataLoaders):
        super().__init__(args)
        self._model = model
        self._data_loaders = data_loaders

    def predict(self, test_dataset: Dataset) -> TaskPredictionOutput:
        test_dataloader = self._data_loaders.get_test_dataloader(test_dataset)
        return self.prediction_loop(test_dataloader, description="Prediction")

    def _prepare_model(self):
        # multi-gpu eval
        if self._args.n_gpu > 1 and not isinstance(self._model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(self._model)
        else:
            model = self._model
        model.to(self._args.device)
        return model

    def _get_batch_size(self, dataloader: DataLoader):
        if self.is_tpu_available():
            return dataloader._loader._loader.batch_size
        else:
            return dataloader.batch_size

    def prediction_loop(
            self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None, max_steps=-1
    ) -> TaskPredictionOutput:
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else prediction_loss_only
        model = self._prepare_model()
        batch_size = self._get_batch_size(dataloader)

        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)

        eval_losses: List[float] = []
        preds: np.ndarray = None
        label_ids: np.ndarray = None
        guids = []

        model.eval()

        limit_step = 0
        for inputs in tqdm(dataloader, desc=description):
            if max_steps == 0 or (0 < max_steps < limit_step):
                break
            limit_step = limit_step + 1

            has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels"])

            for k, v in inputs.items():
                if k != 'guid':
                    inputs[k] = v.to(self._args.device)
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

        if self.is_tpu_available() and preds is not None and label_ids is not None:
            # tpu-comment: Get all predictions and labels from all worker shards of eval dataset
            preds = self.xm.mesh_reduce("eval_preds", preds, np.concatenate)
            label_ids = self.xm.mesh_reduce("eval_out_label_ids", label_ids, np.concatenate)

        return TaskPredictionOutput(guids=guids, predictions=preds, label_ids=label_ids, eval_losses=eval_losses)


class TaskTrainerEvaluator(TrainerBase):
    def __init__(self, args: TaskTrainingArguments,
                 predictor: TaskTrainerPredictor,
                 compute_metrics: Optional[Callable[[TaskEvalPrediction], Dict]] = None):
        super().__init__(args)
        self._predictor = predictor
        self._compute_metrics = compute_metrics

    def evaluate(self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None,
                 max_steps=-1):
        p = self._predictor.prediction_loop(dataloader, description, prediction_loss_only, max_steps)

        if self._compute_metrics is not None and p.predictions is not None and p.label_ids is not None:
            metrics = self._compute_metrics(TaskEvalPrediction(predictions=p.predictions, label_ids=p.label_ids))
        else:
            metrics = {}
        if len(p.eval_losses) > 0:
            metrics["eval_loss"] = np.mean(p.eval_losses)

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)
        p.metrics = metrics

        output = json.dumps({**metrics})
        print(output)

        if self._args.tpu_metrics_debug:
            self.xm.master_print(self.met.metrics_report())

        return p

    @staticmethod
    def build(args: TaskTrainingArguments, model: PreTrainedModel,
              data_loaders: TaskTrainerDataLoaders,
              compute_metrics: Optional[Callable[[TaskEvalPrediction], Dict]] = None):
        return TaskTrainerEvaluator(args, TaskTrainerPredictor(args, model, data_loaders), compute_metrics)
