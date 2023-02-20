from multiprocessing import Queue

from ray.air.integrations.wandb import WandbLoggerCallback, _clean_log


class OurWandbLogger(WandbLoggerCallback):
    def __init__(self, exp_name, *args, **kwargs):
        super(OurWandbLogger, self).__init__(*args, **kwargs)
        self.exp_name = exp_name

    def log_trial_start(self, trial: "Trial"):
        config = trial.config.copy()

        config.pop("callbacks", None)  # Remove callbacks

        exclude_results = self._exclude_results.copy()

        # Additional excludes
        exclude_results += self.excludes

        # Log config keys on each result?
        if not self.log_config:
            exclude_results += ["config"]

        # Fill trial ID and name
        trial_id = trial.trial_id if trial else None
        trial_name = str(trial) if trial else None

        # Project name for Wandb
        wandb_project = self.project

        # Grouping
        wandb_group = self.group or trial.experiment_dir_name if trial else None

        # remove unpickleable items!
        config = _clean_log(config)

        # ========== Our modification! ==========
        run_name = "{}_{}".format(self.exp_name, trial_id)

        wandb_init_kwargs = dict(
            id=trial_id,
            name=run_name,  # Our modification!
            resume=False,
            reinit=True,
            allow_val_change=True,
            group=wandb_group,
            project=wandb_project,
            config=config,
        )
        # ========== Our modification ends! ==========

        wandb_init_kwargs.update(self.kwargs)

        self._trial_queues[trial] = Queue()
        self._trial_processes[trial] = self._logger_process_cls(
            logdir=trial.logdir,
            queue=self._trial_queues[trial],
            exclude=exclude_results,
            to_config=self._config_results,
            **wandb_init_kwargs,
        )
        self._trial_processes[trial].start()
