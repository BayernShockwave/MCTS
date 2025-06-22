import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import os


class ConflictResolutionNet(nn.Module):
    def __init__(self, input_size: int = 512, hidden_size: int = 256, action_size: int = 1000):
        super(ConflictResolutionNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Tanh()
        )
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        p = self.policy_head(x)
        v = self.value_head(x)
        return F.log_softmax(p, dim=1), v


class NeuralNetWrapper:
    def __init__(self, input_size: int = 512, hidden_size: int = 256, action_size: int = 1000):
        self.nnet = ConflictResolutionNet(input_size, hidden_size, action_size)
        self.input_size = input_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nnet.to(self.device)
        self.optimizer = torch.optim.Adam(self.nnet.parameters(), lr=0.001)

    def train(self, examples):
        self.nnet.train()
        states = torch.FloatTensor(np.array([ex[0] for ex in examples])).to(self.device)
        target_p = torch.FloatTensor(np.array([ex[1] for ex in examples])).to(self.device)
        target_v = torch.FloatTensor(np.array([ex[2] for ex in examples])).to(self.device)
        out_p, out_v = self.nnet(states)
        p_loss = -torch.sum(target_p * out_p) / target_p.size()[0]
        v_loss = torch.sum((target_v - out_v.view(-1)) ** 2) / target_v.size()[0]
        total_loss = p_loss + v_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return p_loss.item(), v_loss.item()

    def predict(self, state):
        self.nnet.eval()
        features = self.state_to_features(state)
        features = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            p, v = self.nnet(features)
        return torch.exp(p).cpu().numpy()[0], v.cpu().numpy()[0][0]

    # 将ConflictState转化为神经网络的输入
    def state_to_features(self, state) -> np.ndarray:
        features = []
        total_conflicts = len(state.conflicts)
        node_conflicts = sum(1 for c in state.conflicts if c['place'] == 'NODE')
        link_conflicts = sum(1 for c in state.conflicts if c['place'] == 'LINK')
        features.extend([
            total_conflicts / 10.0,
            node_conflicts / 10.0,
            link_conflicts / 10.0,
            state.depth / 10.0
        ])
        if state.conflicts:
            earliest_conflict_time = min(c['start_secs'] for c in state.conflicts)
            features.append(earliest_conflict_time / 86400.0)
        else:
            features.append(0.0)
        applied_cras = len(state.applied_cras)
        total_cost = sum(cra.estimated_resolution_time for cra in state.applied_cras)
        sum_stt = sum(1 for cra in state.applied_cras if 'STT' in cra.ecs_list)
        sum_rrt = sum(1 for cra in state.applied_cras if 'RRT' in cra.ecs_list)
        features.extend([
            applied_cras / 20.0,
            total_cost / 3600.0,
            sum_stt / 20.0,
            sum_rrt / 20.0
        ])
        if state.conflicts:
            involved_trains = set()
            for c in state.conflicts:
                involved_trains.add(c['course_id_1'])
                involved_trains.add(c['course_id_2'])
            features.append(len(involved_trains) / 50.0)
        else:
            features.append(0.0)
        if state.conflicts:
            involved_nodes = set()
            for c in state.conflicts:
                involved_nodes.add(c['start_node'])
                if c['place'] == 'LINK':
                    involved_nodes.add(c['end_node'])
            features.append(len(involved_nodes) / 30.0)
        else:
            features.append(0.0)
        if state.conflicts and total_conflicts > 1:
            conflict_times = [c['start_secs'] for c in state.conflicts]
            time_span = max(conflict_times) - min(conflict_times)
            density = total_conflicts / (time_span + 1) * 3600
            features.append(density / 10.0)
        else:
            features.append(0.0)
        while len(features) < self.input_size:
            features.append(0.0)
        return np.array(features[:self.input_size])

    def save_checkpoint(self, folder, filename):
        filepath = os.path.join(folder, filename)
        torch.save({
            'state_dict': self.nnet.state_dict(),
            'input_size': self.input_size,
            'action_size': self.action_size
        }, filepath)

    def load_checkpoint(self, folder, filename):
        filepath = os.path.join(folder, filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        self.nnet.load_state_dict(checkpoint['state_dict'])
