import torch

class BeamsearchUELB:
    def __init__(self, y_pred_edges, beam_size, batch_size, edges_capacity, commodities, dtypeFloat, dtypeLong, mode_strict=False):
        self.y_pred_edges = y_pred_edges  # (batch_size, num_nodes, num_nodes, num_commodities)
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.edges_capacity = edges_capacity  # (batch_size, num_nodes, num_nodes)
        self.commodities = commodities  # (batch_size, num_commodities, 3) -> (source_node, target_node, demand)
        self.dtypeFloat = dtypeFloat
        self.dtypeLong = dtypeLong
        self.mode_strict = mode_strict
        self.max_iter = 10

    def search(self):
        # batchサイズでの結果を格納するリスト
        node_orders = []
        all_commodity_paths = []

        for batch in range(self.batch_size):
            # バッチごとにフローを計算
            batch_node_orders, commodity_paths, is_feasible = self._beam_search_single_batch(batch)
            node_orders.append(batch_node_orders)
            all_commodity_paths.append(commodity_paths)
            #tensor_node_orders = torch.tensor(node_orders, dtype=self.dtypeLong)


        #return tensor_node_orders, all_commodity_paths
        return all_commodity_paths, is_feasible

    def _beam_search_single_batch(self, batch):
        # バッチ内の各コモディティに対してルートを探索
        batch_y_pred_edges = self.y_pred_edges[batch]
        commodities = self.commodities[batch]
        count = 0

        while count < self.max_iter:

            batch_edges_capacity = self.edges_capacity[batch]
            #if count == 0:
                #print("元々のエッジ容量：", batch_edges_capacity)
            random_indices = torch.randperm(commodities.size(0))
            #  探索する品種の順番をシャッフル
            shaffled_commodities = commodities[random_indices]
            shaffled_pred_edges = batch_y_pred_edges[:, :, random_indices]
            #shaffled_commodities = commodities
            #shaffled_pred_edges = batch_y_pred_edges
            _, original_indices = torch.sort(random_indices)

            node_orders = []
            commodity_paths = []
            is_feasible = True

            for index, commodity in enumerate(shaffled_commodities):
                source_node = commodity[0].item()
                target_node = commodity[1].item()
                demand = commodity[2].item()
                # ビームサーチで最適なパスを探索
                node_order, remaining_edges_capacity, best_path = self._beam_search_for_commodity(batch_edges_capacity, shaffled_pred_edges[:, :, index], source_node, target_node, demand)
                if best_path == []:
                    break
                # ノード順の追加
                node_orders.append(node_order)
                # エッジ容量の更新
                if self.mode_strict:
                    batch_edges_capacity = remaining_edges_capacity
                commodity_paths.append(best_path)
                # 最後まで行くと終了
            if len(commodity_paths) == commodities.shape[0]:
                count = self.max_iter
                unshaffled_commodity_paths = [commodity_paths[i] for i in original_indices]
                #print("無事探索終了")
            else:
                #print("探索失敗のため、繰り返しを行いました", count)
                is_feasible = False
                #unshaffled_commodity_paths = [[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]]
                unshaffled_commodity_paths = [[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9]]
                count += 1
        
        return node_orders, unshaffled_commodity_paths, is_feasible

    def _beam_search_for_commodity(self, edges_capacity, y_pred_edges, source, target, demand):
        # 初期状態のキュー
        beam_queue = [(source, [source], 0, edges_capacity.clone())]  # (current_node, path, current_score, remaining_edges_capacity)

        best_paths = []

        while beam_queue:
            # スコアでソートし、上位ビームサイズだけを残す
            current_scores = [item[2] for item in beam_queue]
            beam_queue = sorted(beam_queue, key=lambda x: x[2], reverse=True)[:self.beam_size]

            next_beam_queue = []
            for current_node, path, current_score, remaining_edges_capacity in beam_queue:
                if current_node == target:
                    best_paths.append((path, current_score, remaining_edges_capacity))
                    continue

                # 隣接ノードへの探索
                for next_node in range(edges_capacity.shape[0]):
                    # next_nodeがすでにパスに含まれている場合
                    if next_node in path:
                        continue
                    if (edges_capacity[current_node, next_node].item() == 0):
                        continue  # 容量が0、つまりエッジが存在しない場合

                    # 次ノードへの移動でのフローの確率スコア
                    flow_probability = y_pred_edges[current_node, next_node]
                    new_score = current_score + flow_probability

                    # 容量制約の確認
                    if demand <= remaining_edges_capacity[current_node, next_node]:
                        # 新しい remaining_edges_capacity を作成して更新
                        updated_capacity = remaining_edges_capacity.clone()
                        updated_capacity[current_node, next_node] -= demand
                        # 新しいパスを作成
                        new_path = path + [next_node]
                        next_beam_queue.append((next_node, new_path, new_score, updated_capacity))
                        
            beam_queue = next_beam_queue

        # 厳密解が出なかった時の処理
        if not best_paths:
            # デフォルトのnode_order、remaining_edges_capacity、およびbest_pathを設定
            node_order = [0] * edges_capacity.shape[0]
            #print("探索失敗")
            return node_order, edges_capacity, []  # 空のパスを返す

        # 最もスコアの高いパスを返す
        best_paths = sorted(best_paths, key=lambda x: x[1], reverse=True)
        # パスをノードのリストに変換
        node_order = [0] * edges_capacity.shape[0]
        for idx, node in enumerate(best_paths[0][0]):
            node_order[node] = idx + 1  # 通った順番（1から始まる
        return node_order, best_paths[0][2], best_paths[0][0]  # ノードのリスト、残りの容量, パスのリスト
