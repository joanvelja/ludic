"""PVG (Prover-Verifier Game) training orchestration for Ludic."""

from ludic.pvg.algorithm import (
    BradleyTerryWithIRMLoss,
    make_bradley_terry_with_irm,
)
from ludic.pvg.credit import (
    PVGCreditAssigner,
    SignalExtractor,
    attach_verifier_scores,
)
from ludic.pvg.config import (
    DataSplitConfig,
    InferencePlacementConfig,
    PVGGameConfig,
    PVGRoundConfig,
    VerifierInitConfig,
)
from ludic.pvg.data import (
    EqualMixture,
    ExponentialDecayMixture,
    LatestOnlyMixture,
    MixtureStrategy,
    PreferencePair,
    PreferencePairBuilder,
    RolloutRecord,
    RoundDataStore,
    SlidingWindowMixture,
    build_pairs_from_dataset,
    get_mixture_strategy,
)
from ludic.pvg.orchestrator import (
    ConfigMismatchError,
    InvalidTransitionError,
    OrchestratorState,
    PhaseCheckpoint,
    PVGOrchestrator,
    PVGState,
    RoundCheckpoints,
)
from ludic.pvg.prover_env import (
    PVGEnvConfig,
    PVGProverEnv,
    create_pvg_env_factory,
    extract_raw_signals,
    wrap_env_for_pvg,
)
from ludic.pvg.metrics import (
    CollapseAlert,
    Distribution,
    GoodhartingAlert,
    ProblemMetrics,
    PVGMetrics,
    PVGMetricsLogger,
)
from ludic.pvg.rewards import (
    CGCReward,
    CompositeReward,
    GatedMultiplicativeReward,
    RewardStrategy,
    SRCReward,
)
from ludic.pvg.scoring import (
    MockRewardModelClient,
    RewardModelClient,
    VerifierScorer,
    compute_cache_key,
)
from ludic.pvg.minting import (
    MintingConfig,
    MintingResult,
    clone_honest_rollouts_for_round,
    mint_sneaky_data,
    mint_honest_from_dataset,
    create_few_shot_prompt,
)
from ludic.pvg.prover_trainer import (
    ProverTrainingConfig,
    train_prover_phase,
)
from ludic.pvg.verifier_trainer import (
    PreferenceBatchSource,
    VerifierTrainingConfig,
    reinitialize_verifier_head,
    train_verifier_phase,
)

__all__ = [
    # Algorithm
    "BradleyTerryWithIRMLoss",
    "make_bradley_terry_with_irm",
    # Config
    "DataSplitConfig",
    "InferencePlacementConfig",
    "PVGGameConfig",
    "PVGRoundConfig",
    "VerifierInitConfig",
    # Credit
    "PVGCreditAssigner",
    "SignalExtractor",
    "attach_verifier_scores",
    # Data
    "EqualMixture",
    "ExponentialDecayMixture",
    "LatestOnlyMixture",
    "MixtureStrategy",
    "PreferencePair",
    "PreferencePairBuilder",
    "RolloutRecord",
    "RoundDataStore",
    "SlidingWindowMixture",
    "build_pairs_from_dataset",
    "get_mixture_strategy",
    # Metrics
    "CollapseAlert",
    "Distribution",
    "GoodhartingAlert",
    "ProblemMetrics",
    "PVGMetrics",
    "PVGMetricsLogger",
    # Orchestrator
    "ConfigMismatchError",
    "InvalidTransitionError",
    "OrchestratorState",
    "PhaseCheckpoint",
    "PVGOrchestrator",
    "PVGState",
    "RoundCheckpoints",
    # Prover Env
    "PVGEnvConfig",
    "PVGProverEnv",
    "create_pvg_env_factory",
    "extract_raw_signals",
    "wrap_env_for_pvg",
    # Rewards
    "CGCReward",
    "CompositeReward",
    "GatedMultiplicativeReward",
    "RewardStrategy",
    "SRCReward",
    # Scoring
    "MockRewardModelClient",
    "RewardModelClient",
    "VerifierScorer",
    "compute_cache_key",
    # Minting
    "MintingConfig",
    "MintingResult",
    "clone_honest_rollouts_for_round",
    "create_few_shot_prompt",
    "mint_honest_from_dataset",
    "mint_sneaky_data",
    # Prover Training
    "ProverTrainingConfig",
    "train_prover_phase",
    # Verifier Training
    "PreferenceBatchSource",
    "VerifierTrainingConfig",
    "reinitialize_verifier_head",
    "train_verifier_phase",
]
