from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainingSettings(BaseSettings):
    # Models (4-bit MLX checkpoints, no gated access required)
    base_model: str = Field(default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    teacher_model: str = Field(default="mlx-community/Qwen2.5-32B-Instruct-4bit")

    # Paths
    data_dir: str = Field(default="./data")
    distilled_dir: str = Field(default="./data/distilled")
    adapter_output_dir: str = Field(default="./models/adapters/slimorca")
    distilled_adapter_dir: str = Field(default="./models/adapters/distilled")
    fused_output_dir: str = Field(default="./models/fused_model")
    results_dir: str = Field(default="./results")

    # LoRA hyperparameters
    lora_rank: int = Field(default=16)
    lora_alpha: float = Field(default=32.0)
    lora_dropout: float = Field(default=0.05)
    lora_layers: int = Field(default=16, description="Number of transformer layers to adapt")

    # Optimisation
    batch_size: int = Field(default=4)
    learning_rate: float = Field(default=1e-4)
    iters: int = Field(default=1000)
    val_batches: int = Field(default=25)
    steps_per_eval: int = Field(default=100)
    save_every: int = Field(default=200)
    max_seq_length: int = Field(default=2048)
    grad_checkpoint: bool = Field(default=True)

    # Server
    server_model: str = Field(default="./models/fused_model")
    server_host: str = Field(default="127.0.0.1")
    server_port: int = Field(default=8000)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def lora_scale(self) -> float:
        """mlx-lm multiplies the adapter output directly by `scale`, i.e. alpha / rank."""
        return self.lora_alpha / self.lora_rank


settings = TrainingSettings()
