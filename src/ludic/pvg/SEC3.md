SEC 3: PVG Training Orchestration Plan                                                                
                                                                                                       
 Overview                                                                                              
                                                                                                       
 Implement the multi-round Prover-Verifier Game training loop for Ludic. This orchestrates:            
 - Verifier training: Bradley-Terry on (honest, sneaky) preference pairs with IRM regularization       
 - Sneaky prover training: Policy gradient with composite reward (test pass + certified + similarity + 
  verifier score)                                                                                      
 - Round lifecycle: Data minting → verifier training → prover training → checkpoint → repeat           
                                                                                                       
 ---                                                                                                   
 Key Design Decisions (from interview)                                                                 
 ┌────────────────┬────────────────────────────────────────────────────────────────┐                   
 │     Aspect     │                            Decision                            │                   
 ├────────────────┼────────────────────────────────────────────────────────────────┤                   
 │ Honest prover  │ Fixed (dataset samples)                                        │                   
 ├────────────────┼────────────────────────────────────────────────────────────────┤                   
 │ Verifier arch  │ Reward model (LLM + scalar head)                               │                   
 ├────────────────┼────────────────────────────────────────────────────────────────┤                   
 │ Verifier reset │ IRM-inspired (V-REx + balanced batching)                       │                   
 ├────────────────┼────────────────────────────────────────────────────────────────┤                   
 │ Sequential obs │ Configurable (sneaky can observe honest or not)                │                   
 ├────────────────┼────────────────────────────────────────────────────────────────┤                   
 │ Reward         │ Env provides components, credit assigner composes              │                   
 ├────────────────┼────────────────────────────────────────────────────────────────┤                   
 │ Pair strategy  │ Same problem ID (strict)                                       │                   
 ├────────────────┼────────────────────────────────────────────────────────────────┤                   
 │ GPU placement  │ Verifier + prover on inference GPU; prover trains on remaining │                   
 ├────────────────┼────────────────────────────────────────────────────────────────┤                   
 │ Stopping       │ Paper's criterion + fixed max steps                            │                   
 ├────────────────┼────────────────────────────────────────────────────────────────┤                   
 │ Round 0        │ Base model few-shot generations                                │                   
 ├────────────────┼────────────────────────────────────────────────────────────────┤                   
 │ Data split     │ Strict (D_V and D_π disjoint)                                  │                   
 ├────────────────┼────────────────────────────────────────────────────────────────┤                   
 │ Failed sneaky  │ Exclude from verifier training                                 │                   
 ├────────────────┼────────────────────────────────────────────────────────────────┤                   
 │ Score timing   │ Batch after N generations                                      │                   
 ├────────────────┼────────────────────────────────────────────────────────────────┤                   
 │ IRM            │ V-REx + balanced batching, configurable                        │                   
 └────────────────┴────────────────────────────────────────────────────────────────┘                   
 ---                                                                                                   
 New Abstractions                                                                                      
                                                                                                       
 1. PVGRoundConfig (dataclass)                                                                         
                                                                                                       
 src/ludic/pvg/config.py                                                                               
 Configuration for a single round:                                                                     
 - round_id: int                                                                                       
 - verifier_steps: int (max training steps for verifier)                                               
 - prover_steps: int (max training steps for prover)                                                   
 - sneaky_incorrect_threshold: float = 0.95 (stopping criterion)                                       
 - score_parity_threshold: float = 0.1 (stopping criterion)                                            
 - data_mixture_strategy: str = "equal_per_round"                                                      
 - irm_mode: Literal["vrex", "balanced", "both", "none"]                                               
 - sequential_observation: bool = False                                                                
                                                                                                       
 2. PVGGameConfig (dataclass)                                                                          
                                                                                                       
 src/ludic/pvg/config.py                                                                               
 Top-level game configuration:                                                                         
 - num_rounds: int                                                                                     
 - round_configs: List[PVGRoundConfig] (or single default)                                             
 - verifier_model_path: str                                                                            
 - prover_model_path: str                                                                              
 - data_split: DataSplitConfig (D_V and D_π)                                                           
 - output_dir: Path                                                                                    
 - checkpoints_per_round: int = 1                                                                      
                                                                                                       
 3. RewardStrategy (Protocol)                                                                          
                                                                                                       
 src/ludic/pvg/rewards.py                                                                              
 Composable reward computation:                                                                        
 class RewardStrategy(Protocol):                                                                       
     def compute(                                                                                      
         self,                                                                                         
         test_pass_rate: float,                                                                        
         sneaky_certified: bool,                                                                       
         similarity_score: float,                                                                      
         verifier_score: float,                                                                        
         role: str,  # "sneaky" or "honest"                                                            
     ) -> float:                                                                                       
         ...                                                                                           
                                                                                                       
 Built-in strategies:                                                                                  
 - SRCReward: Paper's Signed Relative Convincingness                                                   
 - CGCReward: Paper's Correctness-Gated Convincingness                                                 
 - CompositeReward: Weighted combination with gates                                                    
 - GatedMultiplicativeReward: (test * certified) * (1 - α*sim) * V(z)                                  
                                                                                                       
 4. IRMRegularizer (Protocol)                                                                          
                                                                                                       
 src/ludic/pvg/irm.py                                                                                  
 IRM loss augmentation:                                                                                
 class IRMRegularizer(Protocol):                                                                       
     def compute_penalty(                                                                              
         self,                                                                                         
         losses_per_env: Dict[str, Tensor],  # {env_id: loss}                                          
         model: nn.Module,                                                                             
     ) -> Tensor:                                                                                      
         ...                                                                                           
                                                                                                       
 Built-in:                                                                                             
 - VRExRegularizer: Variance of risks across environments                                              
 - EnvironmentBalancedSampler: Implicit regularization via batching                                    
 - GradientPenaltyRegularizer: ||∇(loss)||² per env (IRM v1)                                           
                                                                                                       
 5. PreferencePairBuilder (class)                                                                      
                                                                                                       
 src/ludic/pvg/data.py                                                                                 
 Constructs (honest, sneaky) pairs for verifier training:                                              
 - build_pairs(honest_rollouts, sneaky_rollouts) -> List[PreferencePair]                               
 - Enforces same-problem-ID pairing                                                                    
 - Filters out failed sneaky (uncertified)                                                             
 - Assigns environment labels (round, role) for IRM                                                    
                                                                                                       
 6. RoundDataStore (class)                                                                             
                                                                                                       
 src/ludic/pvg/data.py                                                                                 
 JSONL-backed storage per round:                                                                       
 - save_rollouts(round_id, role, rollouts)                                                             
 - load_rollouts(round_ids, roles) -> Iterator[Rollout]                                                
 - get_mixture(weights: Dict[int, float]) -> Iterator[Rollout]                                         
 - Provides streaming for memory efficiency                                                            
                                                                                                       
 7. PVGMetrics (dataclass)                                                                             
                                                                                                       
 src/ludic/pvg/metrics.py                                                                              
 Full observability per round:                                                                         
 - sneaky_incorrect_rate: float                                                                        
 - score_gap: float (helpful_mean - sneaky_mean)                                                       
 - verifier_accuracy: float                                                                            
 - certificate_success_rate: float                                                                     
 - similarity_distribution: Distribution                                                               
 - steps_to_exploit: int (when sneaky fools verifier)                                                  
 - gradient_norms: Dict[str, float]                                                                    
 - kl_from_base: float                                                                                 
 - per_problem_breakdown: Dict[str, ProblemMetrics]                                                    
                                                                                                       
 8. VerifierScorer (class)                                                                             
                                                                                                       
 src/ludic/pvg/scoring.py                                                                              
 Batched verifier inference:                                                                           
 - score_batch(solutions: List[str], problems: List[str]) -> List[float]                               
 - Uses vLLM client for efficiency                                                                     
 - Caches scores by (solution_hash, problem_hash)                                                      
                                                                                                       
 ---                                                                                                   
 Integration with Existing Ludic                                                                       
                                                                                                       
 Environment Layer                                                                                     
                                                                                                       
 - Extend SneakyCodeExecEnv: Add sequential_observation mode where prompt includes honest solution as  
 "Reference" block                                                                                     
 - Keep existing signals: test_pass_rate, sneaky_certified, similarity_score in info dict              
 - Add verifier_score attachment: After batch generation, attach scores via RoundDataStore             
                                                                                                       
 Credit Assignment                                                                                     
                                                                                                       
 - New PVGCreditAssigner: Wraps a RewardStrategy and composes signals from env info                    
 - Integrates with existing CreditAssigner protocol                                                    
 - Group awareness: Uses GroupNormalizedReturn internally for baseline                                 
                                                                                                       
 Algorithm                                                                                             
                                                                                                       
 - Verifier: Use existing make_bradley_terry() with score regularization                               
 - Extend for IRM: Add IRMRegularizer to verifier loss via CompositeLoss                               
 - Prover: Use make_scalerl() or make_cispo() with PVGCreditAssigner                                   
                                                                                                       
 BatchSource                                                                                           
                                                                                                       
 - Verifier phase: OfflineBatchSource reading from RoundDataStore                                      
 - Prover phase: RolloutBatchSource with SneakyCodeExecEnv + VerifierScorer                            
 - Custom PVGBatchSource: Wraps both, switches based on phase                                          
                                                                                                       
 Trainer                                                                                               
                                                                                                       
 - No changes to core Trainer: Algorithm injection handles PVG-specific logic                          
 - Lifecycle hooks: Add callbacks for metrics logging per round                                        
                                                                                                       
 ---                                                                                                   
 Round Loop Structure (Script-Level)                                                                   
                                                                                                       
 # examples/pvg/train_pvg.py                                                                           
                                                                                                       
 async def run_pvg_training(config: PVGGameConfig):                                                    
     data_store = RoundDataStore(config.output_dir / "data")                                           
     metrics_logger = PVGMetricsLogger(config.output_dir / "metrics")                                  
                                                                                                       
     # Initialize models                                                                               
     verifier = load_verifier(config.verifier_model_path)                                              
     prover = load_prover(config.prover_model_path)                                                    
                                                                                                       
     for round_id in range(config.num_rounds):                                                         
         round_config = config.get_round_config(round_id)                                              
                                                                                                       
         # === Phase 1: Data Minting ===                                                               
         if round_id == 0:                                                                             
             # Bootstrap: few-shot base model generations                                              
             sneaky_rollouts = await generate_bootstrap_sneaky(                                        
                 prover, config.data_split.d_v, round_config                                           
             )                                                                                         
         else:                                                                                         
             # Generate fresh sneaky samples with current prover                                       
             sneaky_rollouts = await generate_sneaky_rollouts(                                         
                 prover, config.data_split.d_v, round_config                                           
             )                                                                                         
                                                                                                       
         # Load honest solutions from dataset                                                          
         honest_rollouts = load_honest_from_dataset(config.data_split.d_v)                             
                                                                                                       
         # Build preference pairs (filters failed sneaky)                                              
         pairs = PreferencePairBuilder().build_pairs(                                                  
             honest_rollouts, sneaky_rollouts,                                                         
             env_labels={"round": round_id}                                                            
         )                                                                                             
         data_store.save_pairs(round_id, pairs)                                                        
                                                                                                       
         # === Phase 2: Verifier Training ===                                                          
         verifier = reinitialize_verifier(config.verifier_model_path)  # IRM: fresh start              
                                                                                                       
         verifier_batch_source = build_verifier_batch_source(                                          
             data_store,                                                                               
             rounds=list(range(round_id + 1)),  # All previous rounds                                  
             irm_mode=round_config.irm_mode                                                            
         )                                                                                             
                                                                                                       
         verifier_algorithm = make_bradley_terry_with_irm(                                             
             irm_regularizer=get_irm_regularizer(round_config.irm_mode)                                
         )                                                                                             
                                                                                                       
         verifier_trainer = Trainer(                                                                   
             model=verifier,                                                                           
             algorithm=verifier_algorithm,                                                             
             batch_source=verifier_batch_source,                                                       
             config=verifier_trainer_config,                                                           
         )                                                                                             
                                                                                                       
         await verifier_trainer.train(max_steps=round_config.verifier_steps)                           
                                                                                                       
         # Publish verifier to inference                                                               
         await publish_verifier_to_inference(verifier)                                                 
                                                                                                       
         # === Phase 3: Prover Training ===                                                            
         scorer = VerifierScorer(verifier_client)                                                      
                                                                                                       
         prover_batch_source = build_prover_batch_source(                                              
             env_factory=lambda: SneakyCodeExecEnv(                                                    
                 sequential_observation=round_config.sequential_observation                            
             ),                                                                                        
             scorer=scorer,                                                                            
             dataset=config.data_split.d_pi,                                                           
         )                                                                                             
                                                                                                       
         prover_algorithm = make_scalerl(                                                              
             credit_assigner=PVGCreditAssigner(                                                        
                 reward_strategy=config.reward_strategy,                                               
                 scorer=scorer,                                                                        
             )                                                                                         
         )                                                                                             
                                                                                                       
         prover_trainer = Trainer(                                                                     
             model=prover,                                                                             
             algorithm=prover_algorithm,                                                               
             batch_source=prover_batch_source,                                                         
             config=prover_trainer_config,                                                             
         )                                                                                             
                                                                                                       
         # Train with early stopping                                                                   
         await train_prover_with_stopping(                                                             
             prover_trainer,                                                                           
             max_steps=round_config.prover_steps,                                                      
             incorrect_threshold=round_config.sneaky_incorrect_threshold,                              
             score_parity_threshold=round_config.score_parity_threshold,                               
             metrics_logger=metrics_logger,                                                            
         )                                                                                             
                                                                                                       
         # === Phase 4: Checkpoint ===                                                                 
         save_round_checkpoint(                                                                        
             round_id=round_id,                                                                        
             verifier=verifier,                                                                        
             prover=prover,                                                                            
             metrics=metrics_logger.get_round_metrics(round_id),                                       
             output_dir=config.output_dir / "checkpoints",                                             
         )                                                                                             
                                                                                                       
         # Log round summary                                                                           
         metrics_logger.log_round_summary(round_id)                                                    
                                                                                                       
 ---                                                                                                   
 File Structure                                                                                        
                                                                                                       
 src/ludic/pvg/                                                                                        
 ├── __init__.py                                                                                       
 ├── config.py          # PVGRoundConfig, PVGGameConfig, DataSplitConfig                               
 ├── rewards.py         # RewardStrategy protocol + implementations                                    
 ├── irm.py             # IRMRegularizer protocol + VREx, balanced, gradient penalty                   
 ├── data.py            # RoundDataStore, PreferencePairBuilder, PreferencePair                        
 ├── scoring.py         # VerifierScorer (batched inference)                                           
 ├── metrics.py         # PVGMetrics, PVGMetricsLogger                                                 
 ├── credit.py          # PVGCreditAssigner                                                            
 ├── algorithm.py       # make_bradley_terry_with_irm(), helpers                                       
 └── utils.py           # Model loading, weight publishing helpers                                     
                                                                                                       
 examples/pvg/                                                                                         
 ├── train_pvg.py       # Main training script                                                         
 ├── configs/                                                                                          
 │   ├── apps_pvg.yaml  # APPS dataset config                                                          
 │   └── gsm8k_pvg.yaml # GSM8K config (if applicable)                                                 
 └── README.md                                                                                         
                                                                                                       
 ---                                                                                                   
 Critical Files to Modify                                                                              
 File: src/ludic/envs/code_exec/sneaky_env.py                                                          
 Modification: Add sequential_observation mode with reference block in prompt                          
 ────────────────────────────────────────                                                              
 File: src/ludic/training/algorithm.py                                                                 
 Modification: Add make_bradley_terry_with_irm() factory                                               
 ────────────────────────────────────────                                                              
 File: src/ludic/training/loss.py                                                                      
 Modification: Add IRM loss wrappers (or keep in pvg/irm.py)                                           
 ────────────────────────────────────────                                                              
 File: src/ludic/training/credit_assignment.py                                                         
 Modification: Add PVGCreditAssigner (or keep in pvg/credit.py)                                        
 ---                                                                                                   
 Edge Case Handling                                                                                    
                                                                                                       
 Sneaky Collapse (all fail tests/certification)                                                        
                                                                                                       
 - Detection: Track sneaky_valid_rate per batch. Alert if < 10%                                        
 - Mitigation:                                                                                         
   - Curriculum: start with easier problems (shorter solutions)                                        
   - Fallback to SFT on known-incorrect samples to bootstrap                                           
   - Increase temperature to encourage exploration                                                     
 - Implementation: Add sneaky_collapse_threshold to config, trigger recovery mode                      
                                                                                                       
 Verifier Goodharting                                                                                  
                                                                                                       
 - Detection: Track score_gap and sneaky_exploit_rate. Alert if gap shrinks while exploit rate         
 increases                                                                                             
 - Mitigation:                                                                                         
   - V-REx regularization (already planned)                                                            
   - Increase environment diversity (more rounds in mixture)                                           
   - Early stopping on validation set accuracy                                                         
 - Implementation: Add Goodharting detector in metrics, optional early stop trigger                    
                                                                                                       
 ---                                                                                                   
 Verification Strategy                                                                                 
                                                                                                       
 Unit Tests                                                                                            
                                                                                                       
 1. test_reward_strategies.py: Test SRC, CGC, composite reward edge cases                              
 2. test_irm_regularizers.py: Test V-REx, gradient penalty math                                        
 3. test_preference_pair_builder.py: Test filtering, same-problem pairing                              
 4. test_round_data_store.py: Test JSONL save/load, mixture sampling                                   
                                                                                                       
 Integration Tests                                                                                     
                                                                                                       
 1. test_single_round.py: Run one full round (verifier + prover) on tiny dataset                       
 2. test_multi_round.py: Run 2-3 rounds, verify data accumulation and metrics                          
 3. test_sequential_observation.py: Verify prompt construction with honest reference                   
                                                                                                       
 End-to-End Validation                                                                                 
                                                                                                       
 1. Run on APPS-control-arena subset (10 problems, 2 rounds)                                           
 2. Verify:                                                                                            
   - Sneaky incorrect rate increases over rounds                                                       
   - Verifier accuracy improves                                                                        
   - Certificate success rate maintained                                                               
   - No OOM during model transitions                                                                   
 3. Compare metrics to paper's reported values (qualitative sanity check)                              
                                                                                                       
 ---                                                                                                   
 Detailed Specifications (expanded)                                                                    
                                                                                                       
 Verifier Initialization Per Round                                                                     
                                                                                                       
 Each round, verifier is reinitialized from base weights (not warm-started):                           
                                                                                                       
 # In train_pvg.py, before verifier training phase                                                     
 def reinitialize_verifier(base_path: str, round_id: int) -> nn.Module:                                
     """Load fresh verifier weights from base checkpoint."""                                           
     verifier = AutoModelForSequenceClassification.from_pretrained(                                    
         base_path,                                                                                    
         num_labels=1,  # Scalar reward output                                                         
         torch_dtype=torch.bfloat16,                                                                   
     )                                                                                                 
     # Optionally: reinit only the classification head for faster convergence                          
     # if config.head_only_reinit:                                                                     
     #     verifier.score.weight.data.normal_(mean=0.0, std=0.02)                                      
     #     verifier.score.bias.data.zero_()                                                            
     return verifier                                                                                   
                                                                                                       
 Rationale: Fresh weights force verifier to learn from scratch on the accumulated multi-round data,    
 preventing overfitting to adversarial patterns from specific rounds. This is the IRM-aligned          
 approach: the verifier must find features that work across all environments (rounds).                 
                                                                                                       
 Configuration:                                                                                        
 @dataclass                                                                                            
 class VerifierInitConfig:                                                                             
     base_path: str                                                                                    
     reinit_mode: Literal["full", "head_only"] = "full"                                                
     head_init_std: float = 0.02                                                                       
                                                                                                       
 Prover Inference Placement & Weight Sync                                                              
                                                                                                       
 Architecture (single-node, multi-GPU):                                                                
 GPU 0: vLLM server (verifier + prover for inference)                                                  
 GPU 1-N: Prover training (FSDP2)                                                                      
                                                                                                       
 Weight Sync Flow:                                                                                     
 1. Start of round: Load prover checkpoint into training GPUs (FSDP2 sharded)                          
 2. Publish to inference: After each sync_interval training steps:                                     
 # In prover training loop                                                                             
 if step % config.sync_interval == 0:                                                                  
     state_dict = gather_full_state_dict(prover_model)  # FSDP2 gather                                 
     await publisher.publish(state_dict, version=step)                                                 
     # vLLM server receives via NCCL broadcast                                                         
 3. Inference for rollouts: vLLM serves both prover (for sneaky generation) and verifier (for scoring) 
 4. On-policy guarantee: Rollouts use the same policy version that will process gradients              
                                                                                                       
 Implementation:                                                                                       
 @dataclass                                                                                            
 class InferencePlacementConfig:                                                                       
     inference_gpu_ids: List[int] = field(default_factory=lambda: [0])                                 
     training_gpu_ids: List[int] = field(default_factory=lambda: [1, 2, 3])                            
     sync_interval: int = 1  # Sync weights every N training steps                                    
     vllm_tensor_parallel: int = 1  # TP within inference GPUs                                         
                                                                                                       
 Verifier during prover training:                                                                      
 - Verifier weights are frozen during prover training (no gradients)                                   
 - Verifier is loaded into vLLM server on inference GPU                                                
 - VerifierScorer calls vLLM to get scores for sneaky samples                                          
                                                                                                       
 Verifier training phase:                                                                              
 - Prover is not generating during verifier training (offline data only)                               
 - Verifier trains on GPU(s), no vLLM needed                                                           
 - Can use all GPUs for verifier training if desired                                                   
                                                                                                       
 Data Split Implementation (D_V / D_π)                                                                 
                                                                                                       
 Split Strategy: Deterministic hash-based split for reproducibility:                                   
                                                                                                       
 # In src/ludic/pvg/data.py                                                                            
                                                                                                       
 @dataclass                                                                                            
 class DataSplitConfig:                                                                                
     split_ratio: float = 0.5  # 50% D_V, 50% D_π                                                      
     split_seed: int = 42                                                                              
     stratify_by: Optional[str] = "difficulty"  # Optional stratification key                          
                                                                                                       
 def create_data_split(                                                                                
     dataset: Dataset,                                                                                 
     config: DataSplitConfig,                                                                          
 ) -> Tuple[Dataset, Dataset]:                                                                         
     """Split dataset into D_V (verifier) and D_π (prover) subsets."""                                 
                                                                                                       
     # Compute deterministic hash for each sample                                                      
     def sample_hash(sample) -> int:                                                                   
         # Use problem_id or stable identifier                                                         
         key = sample.get("problem_id", str(sample))                                                   
         return int(hashlib.sha256(key.encode()).hexdigest(), 16)                                      
                                                                                                       
     # Assign to split based on hash                                                                   
     hashes = [sample_hash(s) for s in dataset]                                                        
     threshold = int(config.split_ratio * (2**256))                                                    
                                                                                                       
     d_v_indices = [i for i, h in enumerate(hashes) if h < threshold]                                  
     d_pi_indices = [i for i, h in enumerate(hashes) if h >= threshold]                                
                                                                                                       
     # Optional stratification (ensure balanced difficulty in both splits)                             
     if config.stratify_by:                                                                            
         d_v_indices, d_pi_indices = stratified_resample(                                              
             dataset, d_v_indices, d_pi_indices, config.stratify_by                                    
         )                                                                                             
                                                                                                       
     return dataset.select(d_v_indices), dataset.select(d_pi_indices)                                  
                                                                                                       
 Properties:                                                                                           
 - Deterministic: Same split given same dataset + seed                                                 
 - No leakage: Problem IDs are strictly in one split                                                   
 - Balanced: Optional stratification by difficulty/length/category                                     
 - Configurable: Ratio can vary (e.g., 60/40 if more prover data needed)                               
                                                                                                       
 Usage in round loop:                                                                                  
 # At start of training                                                                                
 d_v, d_pi = create_data_split(full_dataset, config.data_split)                                        
                                                                                                       
 # Verifier training uses D_V                                                                          
 verifier_problems = d_v                                                                               
                                                                                                       
 # Prover training uses D_π                                                                            
 prover_problems = d_pi                                                                                
                                                                                                       
 ---                                                                                                   
 Open Questions (for future iteration)                                                                 
                                                                                                       
 1. Verifier size: Should we support different verifier sizes (smaller than prover)?                   
 2. Multi-GPU verifier training: Current plan assumes single-GPU verifier. Need FSDP?                  
 3. Async data minting: Could we generate next round's data while current round trains?                
 4. Curriculum over problems: Should harder problems be introduced in later rounds?                    
                                                                                                       
 ---                                                                                                   
 Implementation Principles                                                                             
                                                                                                       
 1. Mirror Existing Env Patterns                                                                       
                                                                                                       
 - PVGEnv (if needed) wraps SneakyCodeExecEnv, doesn't replace it                                      
 - Follow the same env_reset / env_step / StepOutcome contract                                         
 - Info dict structure matches existing envs (add keys, don't change existing)                         
                                                                                                       
 2. Keep pvg/ Self-Contained                                                                           
                                                                                                       
 - All new code in src/ludic/pvg/ except:                                                              
   - Minimal hook in SneakyCodeExecEnv for sequential observation mode                                 
   - Optional factory registration in algorithm.py                                                     
 - No changes to core Trainer, RolloutEngine, or BatchSource protocols                                 
 - Import from ludic.pvg, don't monkey-patch existing modules                                          
                                                                                                       
 3. Prioritize Testability                                                                             
                                                                                                       
 - Every class takes dependencies via __init__, not globals                                            
 - Protocols/ABCs for all major components (RewardStrategy, IRMRegularizer, etc.)                      
 - Factory functions for complex object construction                                                   
 - Example:                                                                                            
 # Good: dependency injection                                                                          
 class PVGCreditAssigner:                                                                              
     def __init__(self, reward_strategy: RewardStrategy, scorer: VerifierScorer):                      
         self.reward_strategy = reward_strategy                                                        
         self.scorer = scorer                                                                          
                                                                                                       
 # Bad: global state                                                                                   
 class PVGCreditAssigner:                                                                              
     def __init__(self):                                                                               
         self.scorer = get_global_scorer()  # Don't do this                                            
                                                                                                       
 ---                                                                                                   
 Implementation Order                                                                                  
                                                                                                       
 1. Core abstractions (config, rewards, IRM, data store) - foundation                                  
 2. Extend SneakyCodeExecEnv (sequential observation) - enables key experiment                         
 3. VerifierScorer (batched inference) - required for prover training                                  
 4. PVGCreditAssigner (reward composition) - ties signals together                                     
 5. Round loop script (train_pvg.py) - brings it all together                                          
 6. Metrics and logging (PVGMetrics, logger) - observability                                           
 7. Edge case handlers (collapse detection, Goodharting) - robustness                                  
 8. Tests - validation                                                                                 
                                                                                                       
 ---                                                                                                   
 Dependencies                                                                                          
                                                                                                       
 - Existing: SneakyCodeExecEnv, Trainer, BradleyTerryLoss, RolloutEngine, CheckpointManager            
 - New Python packages: None (all built on existing infrastructure)    