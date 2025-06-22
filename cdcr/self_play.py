from cdcr.cdcr_mcts import MCTS
import numpy as np
from collections import deque
from tqdm import tqdm
import random
import os
import pickle


class Coach:
    def __init__(self, initial_state, neural_net, args):
        self.initial_state = initial_state
        self.nnet = neural_net
        self.pnet = self.nnet.__class__(neural_net.input_size, 256, neural_net.action_size)  # 用于比较的先前网络
        self.args = args
        self.mcts = MCTS(self.nnet, args)
        self.trainExamplesHistory = []
        self.skipFirstSelfPlay = False

    def executeEpisode(self):
        trainExamples = []
        state = self.initial_state
        episode_step = 0
        while not state.terminal:
            episode_step += 1
            temp = int(episode_step < self.args['tempThreshold'])
            p = self.mcts.search(state)
            action_vector = self.actions_to_vector(p, state)
            trainExamples.append([
                self.nnet.state_to_features(state),
                action_vector,
                None
            ])
            action = self.select_action(p, temp)
            if action is None:
                break
            state = state.apply_action(action)
            if self.args['verbose']:
                print(f"Step {episode_step}: Applied {action.ecs_list} to trains {action.target_trains}")
                print(f"Remaining conflicts: {len(state.conflicts)}")
        final_value = state.get_value()
        for i in range(len(trainExamples)):
            decay_factor = self.args['valueDecay'] ** (len(trainExamples) - i - 1)
            trainExamples[i][2] = final_value * decay_factor
        return trainExamples

    def actions_to_vector(self, action_probs, state):
        vector = np.zeros(self.nnet.action_size)
        legal_actions = state.get_legal_actions()
        for i, action in enumerate(legal_actions):
            if i >= self.nnet.action_size:
                break
            if action in action_probs:
                vector[i] = action_probs[action]
        if np.sum(vector) > 0:
            vector = vector / np.sum(vector)
        return vector

    def select_action(self, action_probs, temp):
        if not action_probs:
            return None
        actions = list(action_probs.keys())
        probs = np.array([action_probs[a] for a in actions])
        if temp == 0:
            best_idx = np.argmax(probs)
            return actions[best_idx]
        else:
            probs = probs / np.sum(probs)
            idx = np.random.choice(len(actions), p=probs)
            return actions[idx]

    def learn(self):
        for i in range(1, self.args['numIters'] + 1):
            print(f'\nIteration {i}')
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args['maxlenOfQueue'])
                for eps in tqdm(range(self.args['numEps']), desc="Self Play"):
                    self.mcts = MCTS(self.nnet, self.args)
                    episode_examples = self.executeEpisode()
                    iterationTrainExamples += episode_examples
                self.trainExamplesHistory.append(iterationTrainExamples)
            if len(self.trainExamplesHistory) > self.args['numItersForTrainExamplesHistory']:
                self.trainExamplesHistory.pop(0)
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            random.shuffle(trainExamples)
            self.nnet.save_checkpoint(self.args['checkpoint'], 'temp.pth.tar')
            self.pnet.load_checkpoint(self.args['checkpoint'], 'temp.pth.tar')
            print(f'Training Net on {len(trainExamples)} examples')
            for epoch in range(self.args['epochs']):
                print(f'EPOCH: {epoch + 1}')
                batch_count = int(len(trainExamples) / self.args['batch_size'])
                for batch_idx in range(batch_count):
                    start_idx = batch_idx * self.args['batch_size']
                    end_idx = start_idx + self.args['batch_size']
                    batch = trainExamples[start_idx:end_idx]
                    p_loss, v_loss = self.nnet.train(batch)
                    if batch_idx % 10 == 0:
                        print(f'Policy Loss: {p_loss:.4f}, Value Loss: {v_loss:.4f}')
            if self.args['arenaCompare']:
                wins, draws, losses = self.compare_nets(self.nnet, self.pnet, self.args['arenaCompareGames'])
                print(f'NEW/PREV WINS: {wins} / {losses} ; DRAWS: {draws}')
                if wins + losses == 0 or float(wins) / (wins + losses) < self.args['updateThreshold']:
                    print('REJECTING NEW MODEL')
                    self.nnet.load_checkpoint(self.args['checkpoint'], 'temp.pth.tar')
                else:
                    print('ACCEPTING NEW MODEL')
                    self.nnet.save_checkpoint(self.args['checkpoint'], self.getCheckpointFile(i))
                    self.nnet.save_checkpoint(self.args['checkpoint'], 'best.pth.tar')

    def compare_nets(self, net1, net2, num_games):
        wins = 0
        draws = 0
        losses = 0
        for _ in range(num_games):
            mcts1 = MCTS(net1, self.args)
            state1 = self.initial_state
            while not state1.terminal:
                action_probs = mcts1.search(state1)
                action = self.select_action(action_probs, temp=0)
                if action is None:
                    break
                state1 = state1.apply_action(action)
            mcts2 = MCTS(net2, self.args)
            state2 = self.initial_state
            while not state2.terminal:
                action_probs = mcts2.search(state2)
                action = self.select_action(action_probs, temp=0)
                if action is None:
                    break
                state2 = state2.apply_action(action)
            value1 = state1.get_value()
            value2 = state2.get_value()
            if value1 > value2 + 0.01:
                wins += 1
            elif value2 > value1 + 0.01:
                losses += 1
            else:
                draws += 1
        return wins, draws, losses

    def getCheckpointFile(self, iteration):
        return f'checkpoint_{iteration}.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args['checkpoint']
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, f"checkpoint_{iteration}.examples")
        with open(filename, "wb+") as f:
            pickle.dump(self.trainExamplesHistory, f)

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args['load_folder_file'][0], self.args['load_folder_file'][1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            print(f'File "{examplesFile}" with trainExamples not found!')
        else:
            print("Loading trainExamples from file...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = pickle.load(f)
            self.skipFirstSelfPlay = True
