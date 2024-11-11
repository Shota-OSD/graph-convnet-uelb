import torch

class BeamsearchUELB:
    def __init__(self, y_pred_edges, beam_size, batch_size, edges_capacity, commodities, dtypeFloat, dtypeLong):
        self.y_pred_edges = y_pred_edges  # (batch_size, num_nodes, num_nodes, num_commodities)
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.edges_capacity = edges_capacity  # (batch_size, num_nodes, num_nodes)
        self.commodities = commodities  # (batch_size, num_commodities, 3) -> (source_node, target_node, demand)
        self.dtypeFloat = dtypeFloat
        self.dtypeLong = dtypeLong

    def search(self):
        # batchサイズでの結果を格納するリスト
        node_orders = []
        all_commodity_paths = []

        for batch in range(self.batch_size):
            # バッチごとにフローを計算
            batch_node_orders, commodity_paths = self._beam_search_single_batch(batch)
            node_orders.append(batch_node_orders)
            all_commodity_paths.append(commodity_paths)
            tensor_node_orders = torch.tensor(node_orders, dtype=self.dtypeLong)

        return tensor_node_orders, all_commodity_paths

    def _beam_search_single_batch(self, batch):
        # バッチ内の各コモディティに対してルートを探索
        batch_edges_capacity = self.edges_capacity[batch]
        batch_y_pred_edges = self.y_pred_edges[batch]
        commodities = self.commodities[batch]

        node_orders = []
        commodity_paths = []

        for index, commodity in enumerate(commodities):
            source_node = commodity[0].item()
            target_node = commodity[1].item()
            demand = commodity[2].item()
            # ビームサーチで最適なパスを探索
            node_order, remaining_edges_capacity, best_path = self._beam_search_for_commodity(batch_edges_capacity, batch_y_pred_edges[:, :, index], source_node, target_node, demand)           
            node_orders.append(node_order)
            batch_edges_capacity = remaining_edges_capacity
            commodity_paths.append(best_path)
        
        return node_orders, commodity_paths

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

        # 最もスコアの高いパスを返す
        best_paths = sorted(best_paths, key=lambda x: x[1], reverse=True)
        # パスをノードのリストに変換
        node_order = [0] * edges_capacity.shape[0]
        for idx, node in enumerate(best_paths[0][0]):
            node_order[node] = idx + 1  # 通った順番（1から始まる
            print(best_paths[0][0])
        return node_order, best_paths[0][2], best_paths[0][0]  # ノードのリスト、残りの容量, パスのリスト
