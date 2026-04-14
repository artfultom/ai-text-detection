import os
import time
from abc import ABC, abstractmethod

from llama_cpp import Llama
from omegaconf import DictConfig
from openai import OpenAI


class BaseClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, cfg: DictConfig) -> str: ...


class LocalClient(BaseClient):
    def __init__(self, cfg: DictConfig):
        self._llm = Llama(
            model_path=cfg.model.path,
            n_ctx=cfg.model.n_ctx,
            n_gpu_layers=cfg.model.n_gpu_layers,
            n_batch=cfg.model.n_batch,
            n_threads=cfg.model.n_threads,
            use_mlock=True,
            use_mmap=True,
            metal=True,
            verbose=False,
            seed=cfg.run.seed,
        )

    def generate(self, prompt: str, cfg: DictConfig) -> str:
        g = cfg.generation
        output = self._llm(
            prompt,
            max_tokens=self._llm.n_ctx() // 2,
            temperature=g.temperature,
            top_p=g.top_p,
            top_k=g.top_k,
            repeat_penalty=g.repeat_penalty,
        )

        return output["choices"][0]["text"]


class DeepSeekClient(BaseClient):
    def __init__(self, cfg: DictConfig):
        api_key = os.getenv(cfg.model.api_key_env)
        if not api_key:
            raise EnvironmentError(f"Env var '{cfg.model.api_key_env}' not set")

        self._client = OpenAI(api_key=api_key, base_url=cfg.model.base_url)
        self._system_prompt = cfg.model.system_prompt
        self._pause = cfg.model.pause_seconds

    def generate(self, prompt: str, cfg: DictConfig) -> str:
        response = self._client.chat.completions.create(
            model=cfg.model.name,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )
        time.sleep(self._pause)

        return response.choices[0].message.content


class OpenAIClient(BaseClient):
    def __init__(self, cfg: DictConfig):
        api_key = os.getenv(cfg.model.api_key_env)
        if not api_key:
            raise EnvironmentError(f"Env var '{cfg.model.api_key_env}' not set")

        self._client = OpenAI(api_key=api_key)
        self._pause = cfg.model.pause_seconds

    def generate(self, prompt: str, cfg: DictConfig) -> str:
        response = self._client.responses.create(
            model=cfg.model.name,
            input=prompt,
        )
        time.sleep(self._pause)

        return response.output_text


PROVIDERS: dict[str, type[BaseClient]] = {
    "local": LocalClient,
    "deepseek": DeepSeekClient,
    "openai": OpenAIClient,
}


def build_client(cfg: DictConfig) -> BaseClient:
    provider = cfg.model.provider
    cls = PROVIDERS.get(provider)
    if cls is None:
        raise ValueError(f"Unknown provider '{provider}'. Available: {list(PROVIDERS)}")

    return cls(cfg)
