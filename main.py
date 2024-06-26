import os
import torch
import torch.optim as optim
from torch.multiprocessing import Process, Pipe
from environment.worker import Worker
from model.model import paac_ff
import gym
import numpy as np
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import logging
import time
import sys
import torch.distributed as dist
from datetime import timedelta

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def setup_logging(rank):
    # Clear existing handlers if any
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # Set up logging according to the process rank
    log_level = logging.DEBUG if rank == 0 else logging.INFO
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        level=log_level, stream=sys.stdout)


def train(rank, world_size, args):
    print(f"Rank {rank} Starting training. World size world_size, {world_size}, num_workers {args.num_workers}" )

    # Set up logging and TensorBoard writer
    setup(rank, world_size)
    setup_logging(rank)
    writer = SummaryWriter() if rank == 0 else None

    # Parse parameters
    num_envs = args.num_envs
    num_workers = args.num_workers
    total_envs = num_workers * num_envs
    game_name = args.env_name
    max_train_steps = args.max_train_steps
    n_steps = args.n_steps
    init_lr = args.lr
    clip_grad_norm = args.clip_grad_norm
    num_action = gym.make(game_name).action_space.n
    image_size = 84
    n_stack = 4
    
    # Initialize the model and distribute it across GPUs
    model = paac_ff(min_act=num_action, in_channels=n_stack).cuda(rank)
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_parameters} parameters")
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])  

    # Initialize tensors
    x = torch.zeros(total_envs, n_stack, image_size, image_size, requires_grad=False).cuda(rank)
    xs = [torch.zeros(total_envs, n_stack, image_size, image_size).cuda(rank) for _ in range(n_steps)]
    share_reward = [torch.zeros(total_envs).cuda(rank) for _ in range(n_steps)]
    share_mask = [torch.zeros(total_envs).cuda(rank) for _ in range(n_steps)]
    optimizer = optim.Adam(model.parameters(), lr=init_lr)

    # Initialize actors - each one in a different thread
    workers, parent_conns, child_conns = [], [], []
    for i in range(num_workers):
        parent_conn, child_conn = Pipe()
        w = Worker(i, num_envs, game_name, n_stack, child_conn, args)
        w.start()
        workers.append(w)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    # Training loop
    new_s = np.zeros((total_envs, n_stack, image_size, image_size))
    total_episode_rewards = [0] * total_envs
    emulator_steps = [0] * total_envs
    start_time = time.time()
    total_rewards = []
    global_step_start = 1
    
    for global_step in range(1, max_train_steps+1):
        loop_start_time = time.time()
        cache_v_series, entropies, sampled_log_probs = [], [], []

        # RL agents interact with the environment during n_steps
        for step in range(n_steps):
            xs[step].data.copy_(torch.from_numpy(new_s))

            # Forward pass: predict value and policy
            v, pi = model(xs[step])
            cache_v_series.append(v)

            # Sample actions from current policy
            sampling_action = pi.data.multinomial(1)
            log_pi = (pi+1e-12).log()
            # Store entropy to use for exploitation
            entropy = -(log_pi*pi).sum(1)
            sampled_log_prob = log_pi.gather(1, sampling_action).squeeze()
            sampled_log_probs.append(sampled_log_prob)
            entropies.append(entropy)

            # Send actions to workers/environments
            send_action = sampling_action.squeeze().cpu().numpy()
            send_action = np.split(send_action, num_workers)
            for parent_conn, action in zip(parent_conns, send_action):
                parent_conn.send(action)

            # Get states and rewards collected by worker/environments
            batch_s, batch_r, batch_mask = [], [], []
            for parent_conn in parent_conns:
                s, r, mask = parent_conn.recv()
                batch_s.append(s)
                batch_r.append(r)
                batch_mask.append(mask)

            # Update states, rewards, and masks
            new_s = np.vstack(batch_s)
            r = np.hstack(batch_r).clip(-1, 1)
            mask = np.hstack(batch_mask)

            # Log and reset metrics on episode completion
            for envstep, (done, reward) in enumerate(zip(mask, r)):
                total_episode_rewards[envstep] += reward
                emulator_steps[envstep] += 1
                # If episode is done, log metrics and reset counters
                if done: 
                    total_rewards.append(total_episode_rewards[envstep])
                    if writer:
                        writer.add_scalar('rl/reward', total_episode_rewards[envstep], global_step)
                        writer.add_scalar('rl/episode_length', emulator_steps[envstep], global_step)
                        writer.flush()
                    total_episode_rewards[envstep] = 0
                    emulator_steps[envstep] = 0
                share_reward[step].data.copy_(torch.from_numpy(r))
                share_mask[step].data.copy_(torch.from_numpy(mask))

        # Compute value based on final state
        x.data.copy_(torch.from_numpy(new_s))
        v, _ = model(x)
        R = v.data.clone()

        # Iterate from last to first step to compute advantages that lead to V
        # placeholder for losses
        v_loss = 0.0
        policy_loss = 0.0
        entropy_loss = 0.0
        for i in reversed(range(n_steps)):
            R =  share_reward[i] + 0.99 * share_mask[i] * R
            advantage = R - cache_v_series[i]
            v_loss += advantage.pow(2).mul(0.5).mean()
            policy_loss -= sampled_log_probs[i].mul(advantage.detach()).mean()
            entropy_loss -= entropies[i].mean()

        # Compute total loss and perform a step of optimization
        total_loss = policy_loss + entropy_loss.mul(0.02) +  v_loss*0.5
        total_loss = total_loss.mul(1/(n_steps))
        optimizer.zero_grad()

        # Obs: Backward pass is computed on each GPU separately
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        # Synchronize gradients across GPUs
        optimizer.step()

        # Log training information
        if global_step % (2048 / total_envs) == 0 and rank == 0:
            curr_time = time.time()
            last_ten = 'na' if len(total_rewards) < 10 else np.mean(total_rewards[-10:])
            logging.info("Ran {} steps, at {} steps/s ({} steps/s avg), last 10 rewards avg {}"
                            .format(global_step,
                                    n_steps * total_envs / (curr_time - loop_start_time),
                                    (global_step - global_step_start) / (curr_time - start_time),
                                    last_ten))
            torch.save(model.module.state_dict(), f'./saved_models/model_{game_name}_{global_step}.pth')

    # Clanup after finish training
    if writer:
        writer.close()
    cleanup()
    for parent_conn in parent_conns:
        parent_conn.send(None)
    for w in workers:
        w.join()

def parse_args():
    parser = argparse.ArgumentParser(description='parameters_setting')
    parser.add_argument('--lr', type=float, default=0.00025, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                        help='number of workers(default: 4)')
    parser.add_argument('--num-envs', type=int, default=4, metavar='W',
                        help='number of environments a worker holds(default: 4)')
    parser.add_argument('--n-steps', type=int, default=5, metavar='NS',
                        help='number of forward steps in PAAC (default: 5)')
    parser.add_argument('--env-name', default='BreakoutDeterministic-v4', metavar='ENV',
                        help='environment to train on (default: BreakoutDeterministic-v4)')
    parser.add_argument('--max-train-steps', type=int, default=500000, metavar='MS',
                        help='max training step to train PAAC (default: 500000)')
    parser.add_argument('--clip-grad-norm', type=int, default=3.0, metavar='MS',
                        help='globally clip gradient norm(default: 3.0)')
    parser.add_argument('--record', type=bool, default=False, metavar='R',
                        help='record scores of every environment (default: False)')

    return parser.parse_args()


def setup(rank, world_size):
    print(f"Setting up Rank {rank}.")
    dist.init_process_group(
        backend="nccl", 
        rank=rank, 
        world_size=world_size,
        timeout=timedelta(minutes=20)
    )
    torch.cuda.set_device(rank)
    print(f"Rank {rank} setup complete.")

def cleanup():
    dist.destroy_process_group()

if __name__ == "__main__":
    # os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  

    # Parse command line arguments
    args = parse_args()

    # Determine the number of processes to run based on SLURM environment or default to the number of available GPUs
    ntasks = int(os.getenv('SLURM_NTASKS', 1))
    world_size = torch.cuda.device_count() if ntasks == 1 else ntasks

    # Adjust the number of workers per process
    num_gpus_per_node = int(os.getenv('SLURM_GPUS_ON_NODE', torch.cuda.device_count()))

    args.num_workers = int(os.getenv('SLURM_CPUS_PER_TASK'))
    if ntasks == 1:
        args.num_workers = max(1, args.num_workers // num_gpus_per_node)

    args.num_envs = args.num_workers
    
    # Start training
    print(f"World size:{world_size}, 'slurm tasks' {ntasks}, gpus/task: {num_gpus_per_node}, num_envs = {args.num_envs}, workers/task: {args.num_workers}")
    torch.multiprocessing.set_start_method('forkserver')
    torch.multiprocessing.spawn(train, args=(world_size, args), nprocs=world_size, join=True)