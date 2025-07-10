# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import datetime as dt
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import draccus
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from lerobot.common import envs
from lerobot.common.optim import OptimizerConfig
from lerobot.common.optim.schedulers import LRSchedulerConfig
from lerobot.common.utils.hub import HubMixin
from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig # <--- この行を追加

TRAIN_CONFIG_NAME = "train_config.json"


@dataclass
class TrainPipelineConfig(HubMixin):
    dataset: DatasetConfig
    env: envs.EnvConfig | None = None
    policy: PreTrainedConfig | None = None
    # Set `dir` to where you would like to save all of the run outputs. If you run another training session
    # with the same value for `dir` its contents will be overwritten unless you set `resume` to true.
    output_dir: Path | None = None
    job_name: str | None = None
    # Set `resume` to true to resume a previous run. In order for this to work, you will need to make sure
    # `dir` is the directory of an existing run with at least one checkpoint in it.
    # Note that when resuming a run, the default behavior is to use the configuration from the checkpoint,
    # regardless of what's provided with the training command at the time of resumption.
    resume: bool = False
    # `seed` is used for training (eg: model initialization, dataset shuffling)
    # AND for the evaluation environments.
    seed: int | None = 1000
    # Number of workers for the dataloader.
    num_workers: int = 4
    batch_size: int = 8
    steps: int = 100_000
    eval_freq: int = 20_000
    log_freq: int = 200
    save_checkpoint: bool = True
    # Checkpoint is saved every `save_freq` training iterations and after the last training step.
    save_freq: int = 20_000
    use_policy_training_preset: bool = True
    optimizer: OptimizerConfig | None = None
    scheduler: LRSchedulerConfig | None = None
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    # ここに、CLIから受け取るための新しいフィールドを追加します。
    # このフィールド名は、コマンドラインで指定する引数名になります。
    # 例: --cli_policy_observation_state_filter "pos,current" のように使います。
    # これはリストなので、draccusが適切にパースできるように文字列として受け取り、後でリストに変換します。
    cli_policy_observation_state_filter: str | None = field(
        default=None,
        metadata={"help": "Comma-separated list of observation state filters (e.g., 'pos,current'). This overrides policy.observation_state_filter."}
    )

    def __post_init__(self):
        self.checkpoint_path = None
        
        # ここで、CLIから受け取った値を実際のpolicy設定に伝播させます。

        print(f"DEBUG: __post_init__ entered. cli_policy_observation_state_filter: {self.cli_policy_observation_state_filter}")
        print(f"DEBUG: Initial self.policy type: {type(self.policy)}")
        print(f"DEBUG: Initial self.policy.observation_state_filter (if exists): {getattr(self.policy, 'observation_state_filter', 'NOT_FOUND')}")


        if self.cli_policy_observation_state_filter is not None:
            filter_list = [
                f.strip() for f in self.cli_policy_observation_state_filter.split(',')
            ]
            
            # ここが重要です。self.policyがSmolVLAConfigのインスタンスである必要があります。
            # もしこの時点でPolicyConfigのままなら、SmolVLAConfigにキャスト（変換）するか、
            # 別の方法でSmolVLAConfigのインスタンスにアクセスする必要があります。

            # 最も一般的なケースでは、policy_typeに基づいて適切なPolicyConfigのサブクラスに
            # draccusが自動的にデコードしているはずなので、ここでは型チェックをします。
            
            # 伝播前に、policyオブジェクトがSmolVLAConfigのインスタンスであるかを確認
            if isinstance(self.policy, SmolVLAConfig):
                print(f"DEBUG: self.policy is SmolVLAConfig. Setting observation_state_filter to {filter_list}")
                self.policy.observation_state_filter = filter_list
            else:
                print(f"DEBUG: WARNING: self.policy is NOT SmolVLAConfig. It is {type(self.policy)}. Cannot set observation_state_filter directly.")
                # ここで、もしcliからSmolVLAConfigのフィールドを設定しようとしているのに
                # self.policyがPolicyConfig（または別のポリシータイプ）の場合、
                # おそらくcli_policy_observation_state_filterを受け取るだけでなく、
                # policy.typeもCLIから指定し、そのtypeに基づいてpolicyオブジェクトが適切に
                # インスタンス化されることをdraccusに依存する必要があります。
                # 例: --policy.type=smolvla
                # もし--policy.typeも設定しているのにこのエラーが出るなら、draccusのパース順序の問題です。

        print(f"DEBUG: Final self.policy.observation_state_filter (if exists): {getattr(self.policy, 'observation_state_filter', 'NOT_FOUND')}")


    def validate(self):
        # HACK: We parse again the cli args here to get the pretrained paths if there was some.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            # Only load the policy config
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        elif self.resume:
            # The entire train config is already loaded, we just need to get the checkpoint dir
            config_path = parser.parse_arg("config_path")
            if not config_path:
                raise ValueError(
                    f"A config_path is expected when resuming a run. Please specify path to {TRAIN_CONFIG_NAME}"
                )
            if not Path(config_path).resolve().exists():
                raise NotADirectoryError(
                    f"{config_path=} is expected to be a local path. "
                    "Resuming from the hub is not supported for now."
                )
            policy_path = Path(config_path).parent
            self.policy.pretrained_path = policy_path
            self.checkpoint_path = policy_path.parent

        if not self.job_name:
            if self.env is None:
                self.job_name = f"{self.policy.type}"
            else:
                self.job_name = f"{self.env.type}_{self.policy.type}"

        if not self.resume and isinstance(self.output_dir, Path) and self.output_dir.is_dir():
            raise FileExistsError(
                f"Output directory {self.output_dir} already exists and resume is {self.resume}. "
                f"Please change your output directory so that {self.output_dir} is not overwritten."
            )
        elif not self.output_dir:
            now = dt.datetime.now()
            train_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/train") / train_dir

        if isinstance(self.dataset.repo_id, list):
            raise NotImplementedError("LeRobotMultiDataset is not currently implemented.")

        if not self.use_policy_training_preset and (self.optimizer is None or self.scheduler is None):
            raise ValueError("Optimizer and Scheduler must be set when the policy presets are not used.")
        elif self.use_policy_training_preset and not self.resume:
            self.optimizer = self.policy.get_optimizer_preset()
            self.scheduler = self.policy.get_scheduler_preset()

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]

    def to_dict(self) -> dict:
        return draccus.encode(self)

    def _save_pretrained(self, save_directory: Path) -> None:
        with open(save_directory / TRAIN_CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(self, f, indent=4)

    @classmethod
    def from_pretrained(
        cls: Type["TrainPipelineConfig"],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **kwargs,
    ) -> "TrainPipelineConfig":
        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            if TRAIN_CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, TRAIN_CONFIG_NAME)
            else:
                print(f"{TRAIN_CONFIG_NAME} not found in {Path(model_id).resolve()}")
        elif Path(model_id).is_file():
            config_file = model_id
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=TRAIN_CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{TRAIN_CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                ) from e

        cli_args = kwargs.pop("cli_args", [])
        with draccus.config_type("json"):
            return draccus.parse(cls, config_file, args=cli_args)


@dataclass(kw_only=True)
class TrainRLServerPipelineConfig(TrainPipelineConfig):
    dataset: DatasetConfig | None = None  # NOTE: In RL, we don't need an offline dataset
