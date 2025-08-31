#mainのプログラム

import gym.spaces
import networkx as nx
import matplotlib.pyplot as plt
import copy
import time
import csv
import numpy as np
import random
from deepexactsolution import Solve_exact_solution
from LP import Solve_LP_solution
from flow import Flow
import graph_making
import os
from keras.models import Sequential,Model
from keras.layers import Dense, Activation, Flatten,Input,InputLayer,Concatenate,Masking,LeakyReLU,Lambda
from keras.optimizers import Adam
import rl.callbacks
import datetime
import gym

class min_maxload_KSPs_Env(gym.core.Env): # クラスの定義
    def __init__(self, K, n_action, obs_low, obs_high, max_step, node_l, node_h, range_commodity_l, range_commodity_h, capa_l,capa_h,demand_l,demand_h,graph_model, degree, initialstate, rewardstate, countlimit):
        self.K = K
        self.node = random.randint(node_l, node_h)
        self.commodity = random.randint(range_commodity_l, range_commodity_h)
        self.capa_l = capa_l
        self.capa_h = capa_h
        self.demand_l = demand_l
        self.demand_h = demand_h
        self.graph_model = graph_model
        self.degree = degree
        self.initialstate = initialstate
        self.rewardstate = rewardstate
        self.countlimit = countlimit
        self.n_action = n_action
        self.action_space = gym.spaces.Discrete(self.n_action) # 行動の取りうる値
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, shape=(self.n_action,)) # 観測データの取りうる値
        self.time = 0 # ステップ
        self.max_step = max_step
        self.candidate_list = [] # 経路の組み替えの候補リスト
    def render(self, mode): # 画面への描画・可視化
        pass
    def close(self): # 終了時の処理
        pass
    def seed(self): # 乱数の固定
        pass
    def check_is_done(self): # 終了条件
        if self.time >= self.max_step: # 最大ステップ数に達したら終了
            return True
        if all(val == -100.0 for val in self.observation): # 観測変数が全てこの値なら終了
            return True
        return False
    def generate_commodity(self, commodity): # 品種の定義
        determin_st = []
        commodity_list = []
        for i in range(commodity):
            s , t = tuple(random.sample(self.G.nodes, 2)) # 始点と終点の定義
            demand = random.randint(self.demand_l, self.demand_h) # フロー量設定
            tentative_st = [s,t]
            while True:
                if tentative_st in determin_st: # 存在する組み合わせでは作成しない
                    s , t = tuple(random.sample(self.G.nodes, 2)) # 始点と終点の再定義
                    tentative_st = [s,t]
                else:
                    break
            determin_st.append(tentative_st)
            commodity_list.append([s,t,demand])
        commodity_list.sort(key=lambda x: -x[2]) # フロー量大きいものから降順
        return commodity_list
    def search_ksps(self, K, G, commodity,commodity_list): # 経路探索(KSP)
        allcommodity_ksps = []
        for i in range(commodity):
            X = nx.shortest_simple_paths(G, commodity_list[i][0], commodity_list[i][1]) # Yen's algorithm
            ksps_list = []
            for counter, path in enumerate(X):
                ksps_list.append(path)
                if counter == K - 1:
                    break
            allcommodity_ksps.append(ksps_list)
        return allcommodity_ksps

    def searh_combination(self, allcommodity_ksps): # 初期状態：最短経路
        comb = []
        L = len(allcommodity_ksps)
        for z in range(L):
            comb.append(allcommodity_ksps[z][0])
            random.choice(allcommodity_ksps[z]) # 初期状態ランダムの時と比較するときに乱数の数を揃えるため
        combination = [comb]
        return combination
    def searh_randomcombination(self, allcommodity_ksps): # 初期状態：ランダム
        comb = []
        L = len(allcommodity_ksps)
        for z in range(L):
            comb.append(random.choice(allcommodity_ksps[z]))
        combination = [comb]
        return combination
    def get_pair_list(self): # 品種と経路のペアを列挙
        pair_list = []
        for c in range(self.commodity):
            for p in range(len(self.allcommodity_ksps[c])):
                path = self.allcommodity_ksps[c][p]
                pair_list.append([c, path])
        return pair_list
    def exchange_path_action(self, grouping, action): # 行動の値に応じて経路交換
        new_grouping = grouping.copy()
        c, path, cost = self.candidate_list[action]
        new_grouping[c] = path
        return new_grouping
    def exchange_path_pair(self, grouping, c, path): # 候補を作成するための経路交換
        new_grouping = grouping.copy() # grouping:[[1,2,3],[2,4]]
        new_grouping[c] = path
        return new_grouping
    def zero_one(self, grouping): # 負荷率計算時の辺の変換0or1
        zo_combination = []
        for l in range(self.commodity):
            x_kakai = len(self.edges_notindex)*[0]
            for a in range(len(grouping[l])):
                if a == len(grouping[l])-1:
                    break
                set = (grouping[l][a],grouping[l][a+1])
                idx = self.edges_notindex.index(set)
                x_kakai[idx] = 1
            zo_combination.append(x_kakai)
        return zo_combination
    def LoadFactor(self, grouping): # 負荷率計算
        loads = []
        zo_combination = self.zero_one(grouping)
        for e in range(len(self.edge_list)): # 容量制限
            load = sum((zo_combination[l][e])*(self.commodity_list[l][2])for l in range(self.commodity)) / self.capacity_list[self.edge_list[e][1]]
            loads.append(load)
        return loads
    def MaxLoadFactor(self, grouping): # 最大負荷率計算
        self.maxload = max(self.LoadFactor(grouping))
        return self.maxload
    def get_difference(self, grouping): # 最大負荷率の変化差
        new_maxloadfactor = self.MaxLoadFactor(grouping)
        difference = new_maxloadfactor - self.old_maxloadfactor
        return -1 * difference
    
    def get_reward_maxload(self, grouping): # 報酬定義１
        return -1 * self.MaxLoadFactor(grouping)
    def get_reward_difference(self, grouping, bfmaxloadfactor): # 報酬定義２
        new_maxloadfactor = self.MaxLoadFactor(grouping)
        difference = new_maxloadfactor - bfmaxloadfactor
        return -1 * difference
    def get_observation(self): # 観測変数
        grouping = self.grouping.copy()
        candidate_list = []
        self.old_maxloadfactor = self.MaxLoadFactor(grouping)
        for pair in self.pair_list:
            c, path = pair # 品種と経路のペアリストから１つずつ取り出す
            if(self.rewardstate==1):
                cost = self.get_reward_maxload(self.exchange_path_pair(grouping, c, path)) # 今の状態から対象の品種に対して経路を変更して報酬を計算
                if cost>=-1: # オーバーフローじゃない場合にエントリー
                    candidate_list.append([c, path, cost])
            elif(self.rewardstate==2):
                cost = self.get_difference(self.exchange_path_pair(grouping, c, path))
                candidate_list.append([c, path, cost])
        self.candidate_list = sorted(candidate_list, key=lambda x:-x[2])[0:self.n_action] # コストの大きい順に並び替えて個数分取り出す
        mask = [cand[2] for cand in self.candidate_list]
        if len(mask)<self.n_action: # エントリー数が足りない場合　ニューラルネットワークの入力次元に合わせるため
            i = self.n_action - len(mask) # 足りない数を取得 i
            for n in range(i):
                mask.append(-100.0) # 観測変数のリストにこの値を追加
        return mask

    def step(self, action): # ステップ
        self.time += 1
        observation = self.get_observation() # 観測データ取得
        oldmaxload = self.MaxLoadFactor(self.grouping) # 現状の最大負荷率
        if all(val == -100.0 for val in observation):
            done = True  # エピソード終了
            info = {}
            if(self.rewardstate == 1): # 報酬の計算
                self.reward = self.get_reward_maxload(self.grouping)
            elif(self.rewardstate == 2):
                self.reward = self.get_reward_difference(self.grouping,oldmaxload) 
            self.observation = self.get_observation()
        else:
            self.grouping = self.exchange_path_action(self.grouping, action).copy() # 行動の実行
            if(self.rewardstate == 1):
                self.reward = self.get_reward_maxload(self.grouping)
            elif(self.rewardstate == 2):
                self.reward = self.get_reward_difference(self.grouping,oldmaxload)
            self.observation = self.get_observation() # 実行後の次の観測データ取得
            done = self.check_is_done() # 終了条件の判定
            info = {}
        maxload = self.MaxLoadFactor(self.grouping)
        return self.observation, self.reward, done, info, maxload
    def reset(self): # エピソードの初期化
        self.time = 0 # ステップ数初期化
        self.grouping = self.get_random_grouping() # 状態初期化
        self.pair_list = self.get_pair_list() # 行動の範囲を求める
        self.observation = self.get_observation() # 初期状態の観測変数
        count = 0 # 更新回数
        while True: # 初期状態から全ての行動のエントリーが一つも存在しない場合
            if all(val == -100.0 for val in self.observation):
                self.combination = self.searh_randomcombination(self.allcommodity_ksps) # 初期状態の再定義
                self.grouping = self.combination[0]
                count = count + 1
                self.observation = self.get_observation()
                if count == self.countlimit:
                    break
            else:
                break
        return self.observation
    def get_random_grouping(self): # 状態初期化
        self.node = random.randint(node_l, node_h) # グリッドグラフの行列数決定
        self.commodity = random.randint(range_commodity_l, range_commodity_h) # 品種数決定
        if(graph_model == 'grid'): # gridgraph
            self.G = graph_making.Graphs(self.commodity)
            self.G.gridMaker(self.G,self.node*self.node,self.node,self.node,0.1,self.capa_l,self.capa_h)
        if(graph_model == 'random'): # randomgraph
            self.G = graph_making.Graphs(self.commodity)
            self.G.randomGraph(self.G, self.degree , self.node, self.capa_l, self.capa_h)
        if(graph_model == 'nsfnet'): # nsfnetgraph
            self.G = graph_making.Graphs(self.commodity)
            self.G.nsfnet(self.G, self.capa_l, self.capa_h)
        self.capacity_list = nx.get_edge_attributes(self.G,'capacity') # 全辺の容量取得
        self.edge_list = list(enumerate(self.G.edges()))
        self.edges_notindex = []
        for z in range(len(self.edge_list)):
            self.edges_notindex.append(self.edge_list[z][1]) 
        nx.write_gml(self.G, "./deepvalue/graph.gml") # グラフの保存
        self.commodity_list = self.generate_commodity(self.commodity) # 品種作成
        with open('./deepvalue/commodity_data.csv','w') as f:
            writer=csv.writer(f,lineterminator='\n')
            writer.writerows(self.commodity_list) # 品種の保存
        self.allcommodity_ksps = self.search_ksps(self.K, self.G, self.commodity, self.commodity_list) # 経路探索(KSP)
        
        if(self.initialstate==1): # 初期状態：最短経路
            self.combination = self.searh_combination(self.allcommodity_ksps)
            self.grouping = self.combination[0]
        elif(self.initialstate==2): # 初期状態：ランダム
            self.combination = self.searh_randomcombination(self.allcommodity_ksps)
            self.grouping = self.combination[0]

        return self.grouping

def NNmodel(env): # 中間層１つのニューラルネットワーク
    model = Sequential()
    model.add(Dense(layer1, input_dim=env.observation_space.shape[0], activation='relu'))  # 入力層＋中間層１
    model.add(Dense(env.action_space.n, activation='linear'))
    model.summary()
    model.compile(loss='mse', optimizer=Adam(learning_rate=best_learning_rate))
    return model
def DNNmodel(env): # 中間層３つのニューラルネットワーク
    model = Sequential()
    model.add(Dense(layer1, input_dim=env.observation_space.shape[0], activation='relu'))  # 入力層＋中間層１
    model.add(Dense(layer2, activation='relu'))
    model.add(Dense(layer3, activation='relu'))
    model.add(Dense(env.action_space.n, activation='linear'))
    model.summary()
    model.compile(loss='mse', optimizer=Adam(learning_rate=best_learning_rate))
    return model
def choose_action(model, state): # ニューラルネットワークの予測値に基づく行動選択
    q_values = model.predict(state) # 各行動の価値予測
    filtered_q_values = np.where(state[0] == -100.0, -np.inf, q_values) # 観測変数がこの値の場合は、その行動を選択しない
    action = np.argmax(filtered_q_values) # 最大の価値を持つ行動を選択
    return action
def train(env, model, episodes, epsilon, epsilon_decay ,gamma): #学習
    for e in range(episodes): # 学習エピソード分の繰り返し
        total_reward = 0
        state = env.reset() # 戻り値：観測変数
        state = np.reshape(state, [1, env.observation_space.shape[0]])
        done = False
        while not done:
            # ε-greedy
            if np.random.rand() <= epsilon: # 探索
                action = np.random.randint(env.action_space.n)
            else:  # 活用
                action = choose_action(model, state)
            next_state, reward, done, _, Maxload = env.step(action)
            next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
            total_reward += reward #エピソードの累積報酬
            target = reward
            if not done:
                target = reward + gamma * np.amax(model.predict(next_state)[0]) # ターゲット値計算
            target_f = model.predict(state)
            target_f[0][action] = target
            history = model.fit(state, target_f, epochs=1, verbose=0) # ニューラルネットワークの更新
            loss = history.history['loss'][0]
            state = next_state # 行動を反映させた状態の更新
            epsilon = max(0.01, epsilon * epsilon_decay) # εの更新
            if done:
                print(f"Episode {e+1}/{episodes} finished with totalreward: {total_reward}, loss: {loss}")
def test_agent(env, model, nb_episodes, nb_max_episode_steps=None, callbacks=None): # テスト
    for e in range(nb_episodes): # テストエピソード分の繰り返し
        state = env.reset()
        state = np.reshape(state, [1, env.observation_space.shape[0]])
        done = False
        total_reward = 0
        steps = 0
        if callbacks:
            for callback in callbacks: # エピソード開始時のコールバック呼び出し
                callback.on_episode_begin(e)
        while not done:
            action = choose_action(model, state) # 行動の選択
            next_state, reward, done, _ , Maxload = env.step(action)
            next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
            total_reward += reward
            steps += 1
            state = next_state
            if callbacks:
                for callback in callbacks: # ステップ終了時のコールバック呼び出し
                    callback.on_step_end(steps, logs={"state": state, "reward": reward, "done": done})
            if nb_max_episode_steps and steps >= nb_max_episode_steps: # エピソードの終了判定
                break
        if callbacks:
            for callback in callbacks: # エピソード終了時のコールバック呼び出し
                callback.on_episode_end(e, logs={"totalreward": total_reward, "steps": steps})
        print(f"Episode {e+1}/{nb_episodes} finished with totalreward: {total_reward} and maxload: {Maxload} and steps: {steps}")
class CustomEpisodeLogger(rl.callbacks.Callback): # コールバック
    def __init__(self,env):
        self.env = env
        self.episode = 0
        self.start_time = 0 # エピソードごとの計算時間
        self.rewards = {}  # エピソードごとの報酬を保存
        self.objective_values = []
        self.objective_time = []
        self.LP_values = []
        self.LP_time = []
        self.apploximatesolutions = []
        self.apploximatetime = []
    def on_episode_begin(self, episode, logs=None): # エピソード開始時
        self.episode = episode
        self.start_time = time.time() # エピソード開始時間取得
        self.rewards[self.episode] = []
    def on_step_end(self, step, logs): # ステップ終了時
        reward = logs['reward']
        self.rewards[self.episode].append(reward)
        if logs.get('action') == -1:  # 終了フラグとして設定された場合：-1
            self.episode_ended = True
            self.on_episode_end(self.episode, logs=None)
    def on_episode_end(self, episode, logs=None): # エピソード終了時
        end_time = time.time() # エピソード終了時間取得
        elapsed_time = end_time - self.start_time # 計算時間
        apploximate_solution = env.old_maxloadfactor
        self.apploximatesolutions.append(apploximate_solution)
        self.apploximatetime.append(elapsed_time)
        with open('./deepvalue/commodity_data.csv','w') as f:
            writer=csv.writer(f,lineterminator='\n')
            writer.writerows(self.env.commodity_list) # 品種の保存
        nx.write_gml(self.env.G, "./deepvalue/graph.gml") # グラフの保存
        with open(approx_file_name, 'a', newline='') as f:
            out = csv.writer(f)
            out.writerow([self.episode, apploximate_solution, elapsed_time]) # 近似解の保存
        E = Solve_exact_solution(self.episode,solver_type,exact_file_name,limittime)
        objective_value,objective_time = E.solve_exact_solution_to_env() # 厳密解計算
        self.objective_values.append(objective_value)
        self.objective_time.append(objective_time)
        E = Solve_LP_solution(self.episode,solver_type,LP_file_name,limittime)
        LP_value,LP_time = E.solve_LP_solution_to_env() # LPrelaxation
        self.LP_values.append(LP_value)
        self.LP_time.append(LP_time)

def Graph1():
        now = datetime.datetime.now()
        save_dir = 'episode_graphs_exact_deep/' # 保存先ディレクトリ
        os.makedirs(save_dir, exist_ok=True) # ディレクトリが存在しない場合は作成
        epi = episode_logger.rewards

        keys_to_remove = [key for key, value in epi.items() if isinstance(value, list) and len(value) < test_max_step-1] # 削除キーの収集
        for key in keys_to_remove: # 収集したキーの削除
            del epi[key]

        ## ステップごとの平均報酬推移 ##
        mean_reward_list = []
        # 全てのキーをチェックし、処理をスキップする
        for i in range(test_max_step):
            heikin = 0
            for j in range(len(epi)):
                if j in epi:  # キーの存在チェック
                    heikin += epi[j][i]  # 存在する場合のみ処理を実行
            mean_reward_list.append((heikin / len(epi)))
        x = list(range(1, test_max_step + 1))
        plt.plot(x, mean_reward_list, label='N={}'.format(len(epi)))
        plt.xlabel('step', fontsize=15)
        plt.ylabel('mean reward', fontsize=15)
        plt.legend(loc='upper right', fontsize=15)
        plt.xticks(x)
        filename = f"episode_{kurikaeshi + 1}_{range_commodity_l}_{now.strftime('%Y-%m-%d_%H-%M-%S')}_step_reward.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath)
        plt.clf()

        ## 最大負荷率 ##
        y1 = episode_logger.objective_values # 厳密解の最大負荷率
        y2 = episode_logger.apploximatesolutions # 近似解の最大負荷率
        x = np.arange(len(y1)) # x軸
        valid_indices1 = [i for i, value in enumerate(y1) if value is not None]
        valid_indices2 = [i for i, value in enumerate(y2) if value is not None]
        valid_data1 = [y1[i] for i in valid_indices1] # 有効なデータのみを抽出
        valid_data2 = [y2[i] for i in valid_indices2]
        valid_x1 = [x[i] for i in valid_indices1]
        valid_x2 = [x[i] for i in valid_indices2]
        plt.plot(valid_x1, valid_data1, marker='o', linestyle='-', label='exactsolution')
        plt.plot(valid_x2, valid_data2, marker='o', linestyle='-', label='approximatesolution')
        plt.xlabel('episode')
        plt.ylabel('value')
        plt.legend()
        filename = f"episode_{kurikaeshi + 1}_{range_commodity_l}_{now.strftime('%Y-%m-%d_%H-%M-%S')}_value.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath)
        plt.clf()

        ## 近似率 ##
        apploximate_rate = []
        for i in range(nb_episodes):
            if y1[i] is None:
                apploximaterate = 110
            elif y2[i] is None:
                apploximaterate = 0
            else:
                apploximaterate = y1[i]/y2[i]*100
            apploximate_rate.append(apploximaterate)
        x = list(range(1, nb_episodes + 1))
        plt.plot(x, apploximate_rate, label='approximate rate')
        plt.xlabel('episode')
        plt.ylabel('value')
        plt.legend()
        filename = f"episode_{kurikaeshi + 1}_{range_commodity_l}_{now.strftime('%Y-%m-%d_%H-%M-%S')}_rate.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath)
        plt.clf()

        ## 計算時間 ##
        y1 = episode_logger.objective_time # 厳密解の計算時間
        y2 = episode_logger.apploximatetime # 近似解の計算時間
        x = np.arange(len(y1))
        valid_indices1 = [i for i, value in enumerate(y1) if value is not None]
        valid_indices2 = [i for i, value in enumerate(y2) if value is not None]
        valid_data1 = [y1[i] for i in valid_indices1]
        valid_data2 = [y2[i] for i in valid_indices2]
        valid_x1 = [x[i] for i in valid_indices1]
        valid_x2 = [x[i] for i in valid_indices2]
        plt.plot(valid_x1, valid_data1, marker='o', linestyle='-', label='exactsolution time')
        plt.plot(valid_x2, valid_data2, marker='o', linestyle='-', label='approximatesolution time')
        plt.xlabel('episode')
        plt.ylabel('s')
        plt.legend()
        filename = f"episode_{kurikaeshi + 1}_{range_commodity_l}_{now.strftime('%Y-%m-%d_%H-%M-%S')}_time.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath)
        plt.clf()

random.seed(1)
solver_type = "pulp" # ソルバーの種類：mip,pulp,SCIP
limittime = 1800 # ソルバーの制限時間(秒)
result_model = "graph1" # 比較結果のプロット　厳密解との比較：graph1
K = 10 # 探索経路数

## グラフ ##
graph_model = "nsfnet" # グラフの種類：grid,random,nsfnet
capa_l = 500 # 容量の下限
capa_h = 5000 # 容量の上限
node_l = 20 # グリッドグラフの列数下限
node_h = 20 # グリッドグラフの列数上限
degree = 3 # ランダムグラフの頂点の次元数
## 品種 ##
range_commodity_l_list = [25,30,35,40,45,50] # 品種数の下限(複数回実験用)
range_commodity_h_list = [25,30,35,40,45,50] # 品種数の上限(複数回実験用)
demand_l = 100 # フロー量の下限
demand_h = 500 # フロー量の上限

## 強化学習 ##
initialstate = 1 # 初期状態　１：最短経路　２：ランダム
countlimit = 100000 # 初期状態の最大更新回数
rewardstate = 2 # 報酬の定義 １：最大負荷率　２：最大負荷率の差
ln_episodes_list = [1000,1000,1000,1000,1000,1000] # 学習エピソード数(複数回実験用)
nb_episodes = 100 # テストエピソード数
## 学習環境 ##
n_action = 20 # 行動の候補数
obs_low = -20 # 観測変数のスペース下限
obs_high = 20 # 観測変数のスペース上限
train_max_step = 20 # 学習時のステップ数
test_max_step = 20 # テスト時のステップ数
## ニューラルネットワーク ##
epsilon = 0.8 # 探索　活用の割合
epsilon_decay = 0.995 # 減衰率
gamma = 0.85 # 割引率
best_learning_rate = 0.0001 # 学習率
layer1 = 32 # ニューロン数
layer2 = 32
layer3 = 32

for kurikaeshi in range(len(ln_episodes_list)):
    ln_episodes = ln_episodes_list[kurikaeshi] # 学習エピソード数
    range_commodity_l = range_commodity_l_list[kurikaeshi] # 品種数の下限
    range_commodity_h = range_commodity_h_list[kurikaeshi] # 品種数の上限
    # capa_l = range_commodity_l*demand_h # 厳密解が必ず発生する問題の設定
    # capa_h = capa_l+1

    env = min_maxload_KSPs_Env(K, n_action, obs_low, obs_high, train_max_step, node_l, node_h, range_commodity_l, range_commodity_h, capa_l,capa_h,demand_l,demand_h,graph_model,degree,initialstate,rewardstate,countlimit) # 環境の定義
    # model = NNmodel(env) # 強化学習
    model = DNNmodel(env) # 深層強化学習
    print("training start")
    train(env, model, ln_episodes, epsilon, epsilon_decay, gamma) # 学習

    # 結果書き込みファイル
    exact_file_name = f'./deepvalue/exactsolution_{kurikaeshi}_{range_commodity_l}.csv' # 厳密解
    with open(exact_file_name, 'w') as f:
        out = csv.writer(f)
    approx_file_name = f'./deepvalue/approximatesolution_{kurikaeshi}_{range_commodity_l}.csv' # 近似解
    with open(approx_file_name, 'w') as f:
        out = csv.writer(f)
    LP_file_name = f'./deepvalue/LPsolution_{kurikaeshi}_{range_commodity_l}.csv' # LP
    with open(LP_file_name, 'w') as f:
        out = csv.writer(f)

    random.seed(7) # テストデータの固定(複数回実験用)
    episode_logger = CustomEpisodeLogger(env)
    print("test start")
    test_agent(env, model, nb_episodes=nb_episodes, nb_max_episode_steps=test_max_step,callbacks=[episode_logger]) # テスト
    if (result_model == 'graph1'): # 結果のプロット
        Graph1()
    env.close()
    print("")