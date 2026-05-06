"""Experiment: takes a YAML config, wires everything via registries, runs end-to-end."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np

from llm_kg.config import RootConfig, load_config
from llm_kg.data import DATASET_REGISTRY
from llm_kg.evaluation import EVALUATOR_REGISTRY, Prediction
from llm_kg.logging_ import LOGGER_REGISTRY
from llm_kg.methods import METHOD_REGISTRY
from llm_kg.methods.base import MethodConfig
from llm_kg.pipeline.stage import PipelineContext
from llm_kg.providers import EMBEDDING_REGISTRY, LLM_REGISTRY
from llm_kg.providers.cache import CachedEmbeddingProvider, CachedLLMProvider
from llm_kg.storage import GRAPH_REGISTRY, KV_REGISTRY, VECTOR_REGISTRY


class Experiment:
    """One end-to-end experiment run.

    Construction loads and validates the config but does not allocate any
    plug-ins. Call `run()` to actually instantiate everything and execute the
    pipeline.
    """

    def __init__(self, config_path: Path | str) -> None:
        self.config_path = Path(config_path)
        self.config: RootConfig = load_config(self.config_path)

    async def run(self) -> dict[str, float]:
        cfg = self.config
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

        # --- providers ---
        llm_cls = LLM_REGISTRY.get(cfg.llm.name)
        llm = llm_cls(**cfg.llm.params)
        if cfg.llm.cache:
            llm = CachedLLMProvider(
                inner=llm,
                model=str(cfg.llm.params.get("model", "default")),
                provider_name=cfg.llm.name,
                cache_dir=cfg.cache_dir,
            )

        embedder_cls = EMBEDDING_REGISTRY.get(cfg.embedder.name)
        embedder = embedder_cls(**cfg.embedder.params)
        if cfg.embedder.cache:
            embedder = CachedEmbeddingProvider(
                inner=embedder,
                model=str(cfg.embedder.params.get("model", "default")),
                provider_name=cfg.embedder.name,
                cache_dir=cfg.cache_dir,
            )

        # --- storage ---
        vec_store = VECTOR_REGISTRY.get(cfg.storage.vector.name)(**cfg.storage.vector.params)
        graph_store = GRAPH_REGISTRY.get(cfg.storage.graph.name)(**cfg.storage.graph.params)
        kv_store = KV_REGISTRY.get(cfg.storage.kv.name)(**cfg.storage.kv.params)

        # --- logger ---
        logger_kwargs: dict[str, Any] = {}
        if cfg.logger.project is not None:
            logger_kwargs["project"] = cfg.logger.project
        if cfg.logger.run_name is not None:
            logger_kwargs["run_name"] = cfg.logger.run_name
        logger = LOGGER_REGISTRY.get(cfg.logger.name)(**logger_kwargs)
        run_name = cfg.logger.run_name or f"{cfg.method.name}-{cfg.dataset.name}"
        logger.init(cfg.model_dump(mode="json"), run_name=run_name)

        # --- dataset & method ---
        dataset = DATASET_REGISTRY.get(cfg.dataset.name)(**cfg.dataset.params)
        method = METHOD_REGISTRY.get(cfg.method.name)()

        ctx = PipelineContext(
            llm=llm,
            embedder=embedder,
            vector_store=vec_store,
            graph_store=graph_store,
            kv_store=kv_store,
            logger=logger,
            trace={},
        )
        indexing, query = method.build(MethodConfig(**cfg.method.model_dump()), ctx)

        # --- index ---
        corpus = dataset.corpus()
        await indexing.run(list(corpus.documents()), ctx)

        # --- query each example ---
        predictions: list[Prediction] = []
        for ex in dataset.examples():
            ctx.query = ex.question  # so the Generator can read the question
            answer = await query.run(ex.question, ctx)
            predictions.append(
                Prediction(
                    qid=ex.id,
                    question=ex.question,
                    answer=answer if isinstance(answer, str) else str(answer),
                    gold=ex.answers if ex.answers else ex.long_answer,
                )
            )

        # --- evaluate & log ---
        evaluator = EVALUATOR_REGISTRY.get(cfg.evaluator.name)(**cfg.evaluator.params)
        metrics = await evaluator.score(predictions)

        logger.log_metrics(metrics)
        logger.log_predictions(predictions)
        logger.finish()

        return metrics
