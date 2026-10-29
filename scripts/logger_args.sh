# Logger arguments of the launchers (sourced by pretrain.sh and finetune.sh).
#
#   logger_args <run name> [<csv version>]   fills the array LOGGER_ARGS
#
# Metrics always go to logs/<run name>/version_<n>/metrics.csv. With WANDB=1 they are
# also sent to Weights & Biases (project WANDB_PROJECT, default ba-gnn-pretraining;
# team WANDB_ENTITY, default chebai). The W&B run id is stored in logs/<run name>/wandb_id, so a resumed
# pretraining and the test after a fine-tuning continue the same W&B run.

logger_args() {
  local name="$1" version="${2:-}"
  if [ "${WANDB:-0}" != "1" ]; then
    LOGGER_ARGS=(--trainer.logger.init_args.name="$name")
    if [ -n "$version" ]; then LOGGER_ARGS+=(--trainer.logger.init_args.version="$version"); fi
    return
  fi
  local id_file="logs/$name/wandb_id"
  if [ ! -f "$id_file" ]; then
    mkdir -p "logs/$name"
    echo "${name:0:48}_$(date +%s)" > "$id_file"
  fi
  local csv="\"save_dir\":\"logs\",\"name\":\"$name\""
  if [ -n "$version" ]; then csv="$csv,\"version\":\"$version\""; fi
  local wandb="\"save_dir\":\"logs\",\"project\":\"${WANDB_PROJECT:-ba-gnn-pretraining}\""
  wandb="$wandb,\"name\":\"$name\",\"id\":\"$(cat "$id_file")\",\"log_model\":false"
  wandb="$wandb,\"entity\":\"${WANDB_ENTITY:-chebai}\""
  # the CSV logger comes first: it decides where the checkpoints are written
  LOGGER_ARGS=(--trainer.logger="[{\"class_path\":\"lightning.pytorch.loggers.CSVLogger\",\"init_args\":{$csv}},{\"class_path\":\"lightning.pytorch.loggers.WandbLogger\",\"init_args\":{$wandb}}]")
}
