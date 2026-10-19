import torch


class PPOBuffer:
    """
    Trajectory memory buffer for Proximal Policy Optimization (PPO).
    Stores rollout experiences and computes generalized advantages and returns.
    """
    def __init__(self, clip_eps=0.2):
        self.clip_eps = clip_eps
        self.trajectories = []
        self.old_log_probs = []
        self.rewards = []
        self.values = []

    def store(self, trajs, old_log_prob, reward, value):
        """
        Stores rollout batch data.
        """
        self.trajectories.extend(trajs)
        self.old_log_probs.append(old_log_prob.detach())
        self.rewards.append(reward.detach())
        self.values.append(value.detach())

    def get_data(self):
        """
        Concatenates stored batches and computes empirical returns and advantages.
        """
        old_log_probs = torch.cat(self.old_log_probs, dim=0).view(-1)
        rewards = torch.cat(self.rewards, dim=0).view(-1)
        values = torch.cat(self.values, dim=0).view(-1)

        # In bandit/episode terminal setting, returns G_t = R
        returns = rewards

        # Advantage: A = R - V(s)
        advantages = returns - values

        # Normalize advantages across batch to stabilize gradient updates
        if advantages.numel() > 1 and advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = advantages - advantages.mean()

        return {
            'trajectories': self.trajectories,
            'old_log_probs': old_log_probs,
            'returns': returns.unsqueeze(-1),
            'advantages': advantages
        }

    def clear(self):
        """Empties buffer after PPO update phase."""
        self.trajectories.clear()
        self.old_log_probs.clear()
        self.rewards.clear()
        self.values.clear()
