import os
import torch
import numpy as np
import logging
from datetime import datetime
import json

from configs import main_config as cfg
from envs.vrp_gym import VRPCGEnv
from agents.dqn_learner import DQNAgent
from utils.utils import create_logger, parse_args, visualize_solution, save_checkpoint, get_checkpoint_path

def train_process(args):
    # Setup Logger
    logger = create_logger(cfg.LOG_FILE_TRAIN)
    logger.info(">>> START TRAINING SESSION <<<")
    logger.info(f"Episodes: {args.episodes}")
    
    # Env & Agent
    env = VRPCGEnv() # Auto load default train file config
    
    # Init Agent
    agent = DQNAgent()
    
    # Load previous training checkpoint if needed
    if args.model_path and os.path.exists(args.model_path):
        agent.load(args.model_path)
    
    # Main Loop
    global_step = 0
    best_train_reward = -float('inf')
    
    for episode in range(1, args.episodes + 1):
        try:
            obs = env.reset()
        except RuntimeError as e:
            logger.error(f"Episode {episode} reset failed: {e}")
            continue
            
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < cfg.DQN_MAX_STEPS_PER_EPISODE:
            global_step += 1
            steps += 1
            
            # Training=True để bật Epsilon Greedy
            action_idx = agent.select_action(obs, training=True)
            
            if action_idx is None:
                logger.info(">> No valid action/candidates selected. Ending episode early.")
                break
                
            next_obs, reward, done, _, info = env.step(action_idx)
            
            # Chỉ lưu memory khi có action hợp lệ
            if action_idx is not None:
                # --- FIX: Gọi agent.store và agent.train ---
                agent.store_transition(obs, action_idx, reward, next_obs, done)
                
                loss = agent.train_step()
                if loss:
                    # Optional: Log loss
                    pass
            
            obs = next_obs
            episode_reward += reward
              
            if done:
                logger.info(f"Ep {episode}: Done in {steps} steps. Rwd: {episode_reward:.2f}. Obj: {info.get('obj', 0):.2f}")
                break
        
        # Save checkpoint periodically
        if episode % 10 == 0:
            ckpt_path = get_checkpoint_path(cfg.DIR_DQN_OUTPUT, "dqn_ckpt", episode)
            agent.save(ckpt_path)
            
    # Final save
    final_path = get_checkpoint_path(cfg.DIR_DQN_OUTPUT, "dqn_final", args.episodes)
    agent.save(final_path)
    logger.info("TRAINING COMPLETED.")

    

def test_process(args):
    # --- 1. SETUP RESULT DIR ---
    # outputs/vrp_results/result_ddmm_hhmm/
    timestamp = datetime.now().strftime("%d%m_%H%M")
    result_folder = os.path.join(cfg.DIR_VRP_RESULTS, f"inference_{timestamp}")
    os.makedirs(result_folder, exist_ok=True)
    
    log_file = os.path.join(result_folder, "inference_log.txt")
    logger = create_logger(log_file)
    
    logger.info(f">>> START TESTING SESSION [{timestamp}] <<<")
    logger.info(f"Instance: {args.test_file}")
    
    if args.model_path and os.path.exists(args.model_path):
        logger.info(f"Loading DQN Model: {args.model_path}")
    else:
        logger.info("No DQN Model loaded")

    # Override config instance file để Env load file test này
    # (Vì VRPCGEnv init load từ cfg.INSTANCE_FILE)
    # Hack tạm thời:
    cfg.TEST_INSTANCE_FILE = args.test_file
    
    # --- 2. INIT & SOLVE ---
    import time
    start_time = time.time()
    
    env = VRPCGEnv()
    agent = DQNAgent()
    
    # Load Model if available
    if args.model_path:
        agent.load(args.model_path)
    # Test thì epsilon = 0 (Greedy)
    agent.epsilon = 0.0 
    
    obs = env.reset()
    init_obj = env.init_obj
    logger.info(f"Initial Obj (Dummy Solution): {init_obj:.2f}")
    
    done = False
    steps = 0
    final_obj = init_obj
    
    while not done and steps < 50: # Max test steps hardcap
        steps += 1
        
        action_idx = agent.select_action(obs, training=False)
        next_obs, reward, done, _, info = env.step(action_idx)
        
        current_obj = info.get('obj', final_obj)
        logger.info(f"Step {steps}: Action {action_idx}. Obj: {current_obj:.2f} (Reward {reward:.2f})")
        
        final_obj = current_obj
        obs = next_obs
        
        if done:
            logger.info("Converged (Done).")

    solve_time = time.time() - start_time
    improvement = (init_obj - final_obj) / init_obj * 100.0
    
    # --- 3. FINAL LOG & METRICS ---
    logger.info("\n=== FINAL REPORT ===")
    logger.info(f"Initial Cost: {init_obj:.2f}")
    logger.info(f"Final Cost:   {final_obj:.2f}")
    logger.info(f"Improvement:  {improvement:.2f} %")
    logger.info(f"CG Iterations:{steps}")
    logger.info(f"Time Taken:   {solve_time:.2f} sec")
    
    # Dump Summary JSON
    summary = {
        "instance": args.test_file,
        "init_obj": init_obj,
        "final_obj": final_obj,
        "improvement_pct": improvement,
        "iterations": steps,
        "solve_time": solve_time,
        "date": timestamp,
        "routes_count": len(env.rmp_service.routes_data)
    }
    with open(os.path.join(result_folder, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=4)
        
    # --- 4. VISUALIZATION ---
    # Get all active routes (Variable in basis)
    # Lấy solution basis từ RMP (các biến > 0)
    # Chạy lại solve 1 lần cuối để lấy biến số chính xác
    _, _, basis_vals = env.rmp_service.solve()
    
    active_routes = []
    # RMP Vars là object ORTools, routes_data là list lưu tương ứng index
    for idx, val in basis_vals.items():
        if val > 0.9: # Integer solution approximation
            route = env.rmp_service.routes_data[idx]
            active_routes.append(route)
            
    # Visualize
    visualize_solution(active_routes, env.locations_df, result_folder)
    logger.info(f"Visualizations saved to: {result_folder}")

if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train_process(args)
    else:
        test_process(args)