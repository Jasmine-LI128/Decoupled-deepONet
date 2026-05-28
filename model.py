import torch
import torch.nn as nn


# ==========================================
# Basic components (unchanged)
# ==========================================
class CNNEncoder(nn.Module):
    """
    Encode a 1500-dimensional temporal signal into a feature sequence.
    Input: (B, 1500) -> Output: (B, 64, ~47)
    """

    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        # Add the channel dimension: (B, 1500) -> (B, 1, 1500)
        if x.dim() == 2: x = x.unsqueeze(1)
        return self.net(x)


# ==========================================
# Branch Type A: CNN + LSTM (unchanged)
# Suitable for plastic strain and residual stress, which require history-path modeling.
# ==========================================
class CNN_LSTM_Branch_Module(nn.Module):
    def __init__(self, p=64):
        super(CNN_LSTM_Branch_Module, self).__init__()
        self.cnn = CNNEncoder()
        # CNN output: (B, 64, 47); LSTM input: (B, 47, 64)
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        self.branch_out = nn.Linear(64, p)

    def forward(self, branch_input):
        # 1. Extract features with CNN
        feat_cnn = self.cnn(branch_input)  # (B, 64, 47)

        # 2. Rearrange dimensions for LSTM
        feat_seq = feat_cnn.permute(0, 2, 1)  # (B, 47, 64)

        # 3. Process the sequence with LSTM
        _, (h_n, _) = self.lstm(feat_seq)

        # 4. Use the final hidden state as the output representation
        return self.branch_out(h_n[-1])


# ==========================================
# Branch Type B: CNN + MLP (unchanged)
# Suitable for elastic strain, which is mainly an instantaneous response and does not require complex gated memory.
# ==========================================
class CNN_MLP_Branch_Module(nn.Module):
    def __init__(self, p=64):
        super(CNN_MLP_Branch_Module, self).__init__()
        self.cnn = CNNEncoder()

        # Compute the flattened dimension after the CNN output
        # Input 1500 -> CNNEncoder -> (64, 47)
        # 64 * 47 = 3008
        self.flatten_dim = 64 * 47

        self.mlp = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, p)
        )

    def forward(self, branch_input):
        # 1. Extract features with CNN
        feat_cnn = self.cnn(branch_input)  # (B, 64, 47)

        # 2. Flatten the feature map: (B, 3008)
        feat_flat = feat_cnn.view(feat_cnn.size(0), -1)

        # 3. Map features with MLP
        return self.mlp(feat_flat)


# ==========================================
# Main Hybrid Multi-Task DeepONet model (modified)
# ==========================================
class Hybrid_MultiTask_DeepONet(nn.Module):
    """
    Architecture:
    - N independent branches, using CNN-MLP or CNN-LSTM depending on the task.
    - One shared trunk backbone for the early coordinate-processing layers.
    - N independent trunk heads for task-specific coordinate projections.
    """

    def __init__(self, trunk_dim=4, num_tasks=4, p=64):
        """
        :param num_tasks: default is 4 (Elastic, Plastic, Thermal, Stress)
        """
        super(Hybrid_MultiTask_DeepONet, self).__init__()
        self.num_tasks = num_tasks

        # -----------------------------------------------------------
        # 1. Define the branch list (logic unchanged)
        # -----------------------------------------------------------
        self.branches = nn.ModuleList()

        for i in range(num_tasks):
            if i == 0:
                # Task 0: Elastic -> use CNN-MLP
                self.branches.append(CNN_MLP_Branch_Module(p))
            else:
                # Task 1 (Plastic), Task 2 (Thermal), Task 3 (Stress) -> use CNN-LSTM
                self.branches.append(CNN_LSTM_Branch_Module(p))

        # -----------------------------------------------------------
        # 2. Define the trunk structure (core modification)
        # -----------------------------------------------------------

        # (A) Shared backbone: extract general geometric/coordinate features
        # Input trunk_dim -> 128 -> 128
        self.trunk_backbone = nn.Sequential(
            nn.Linear(trunk_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
            # The last layer is removed here to allow task-specific branching
        )

        # (B) Independent heads: provide a separate linear projection for each task
        # 128 -> p for each task
        self.trunk_heads = nn.ModuleList()
        for _ in range(num_tasks):
            self.trunk_heads.append(nn.Linear(128, p))

        # 3. Bias
        self.biases = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, trunk_in, branch_in, grid=None):
        # --------------------------------------
        # Step 1: Pass through the shared backbone
        # --------------------------------------
        # shared_feat shape: (B, 128)
        shared_feat = self.trunk_backbone(trunk_in)

        outputs = []

        # --------------------------------------
        # Step 2: Iterate over tasks and pass each one through its own head
        # --------------------------------------
        for i in range(self.num_tasks):
            # A. Obtain trunk features through the i-th head
            # t_feat shape: (B, p)
            t_feat = self.trunk_heads[i](shared_feat)

            # B. Obtain branch features through the i-th branch
            # b_feat shape: (B, p)
            b_feat = self.branches[i](branch_in)

            # C. Combine branch and trunk features using the DeepONet dot product
            val = torch.sum(b_feat * t_feat, dim=1, keepdim=True) + self.biases[i]
            outputs.append(val)

        # Concatenate outputs from all tasks: (B, 4)
        return torch.cat(outputs, dim=1)