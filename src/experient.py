import random
import time
import math
import pickle
from pathlib import Path
import networkx as nx
import numpy as np
from collections import defaultdict
from collections import deque
from typing import Any, Dict, Iterator, List, Optional, Tuple
from decimal import Decimal, getcontext
getcontext().prec = 50   # 设置50位精度
import mpmath
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from scipy.stats import rankdata

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"
RAW_RESULT_DIR = RESULT_DIR / "raw"
DATA_DIR = Path(__file__).resolve().parent / "data"
SEED = 42

# ---------------------------------------------------------
# 精确介数中心性（Brandes）
# ---------------------------------------------------------

def exact_bc(G):
    return nx.betweenness_centrality(G, normalized=True)


# ---------------------------------------------------------
# 均匀随机采样算法BP
# ---------------------------------------------------------

def BP_sampling_bc(G, eplison):
    nodes = list(G.nodes())
    bc = defaultdict(float)

    T=round(1.0/eplison*math.sqrt(len(nodes)))+1

    for _ in range(T):

        s, t = random.sample(nodes, 2)

        try:
            paths = list(nx.all_shortest_paths(G, s, t))
        except:
            continue

        if len(paths) == 0:
            continue

        path = random.choice(paths)

        for v in path[1:-1]:
            bc[v] += 1.0

    # 归一化
    for v in nodes:
        bc[v] /= T

    return bc

def RK(G,eplison,delta):
    nodes = list(G.nodes())
    VD=len(nodes)
    bc= defaultdict(float)
    for v in nodes:
        bc[v]=0
    r=round(0.5/eplison/eplison*(round(math.log2(VD))-math.log(delta)))+1
    for _ in range(r):
        s, t = random.sample(nodes, 2)
        try:
            paths = list(nx.all_shortest_paths(G, s, t))
        except:
            continue

        if len(paths) == 0:
            continue

        path = random.choice(paths)
        # 一条路径的长度包括首尾，中间节点是 path[1:-1]
        for v in path[1:-1]: 
            bc[v] += 1.0 / r
                
    return bc
def compute_f_w(G:nx, u: int, v: int) :
    """
    计算从 u 到 v 的所有最短路径上，每个节点 w 的 f_w(u,v) = σ_uv(w) / σ_uv
    返回长度为 n 的列表，下标为节点编号
    """
    n = (len)(nx.nodes(G))
    # BFS 从 u 开始
    dist = [-1] * n
    sigma = [0] * n
    pred = [[] for _ in range(n)]
    queue = deque([u])
    dist[u] = 0
    sigma[u] = 1
    while queue:
        cur = queue.popleft()
        for nb in G[cur]:
            if dist[nb] == -1:
                dist[nb] = dist[cur] + 1
                queue.append(nb)
            if dist[nb] == dist[cur] + 1:
                sigma[nb] += sigma[cur]
                pred[nb].append(cur)
    # 如果 v 不可达
    if dist[v] == -1:
        return [0.0] * n
    # 反向传播计算 σ_uv(w)
    # 使用一个字典记录每个节点在反向 BFS 中从 v 出发的依赖
    # 更简单的方式：按距离递减顺序处理节点，计算每个节点 w 的 σ_uv(w)
    # 公式: σ_uv(w) = sigma_uw * sigma_wv，其中 sigma_uw 是正向 BFS 得到的，
    # sigma_wv 是从 w 到 v 的最短路径数，可通过反向 BFS 得到。
    # 下面使用反向 BFS 计算 sigma_wv
    dist_rev = [-1] * n
    sigma_rev = [0] * n
    queue_rev = deque([v])
    dist_rev[v] = 0
    sigma_rev[v] = 1
    while queue_rev:
        cur = queue_rev.popleft()
        for nb in G[cur]:
            if dist_rev[nb] == -1:
                dist_rev[nb] = dist_rev[cur] + 1
                queue_rev.append(nb)
            if dist_rev[nb] == dist_rev[cur] + 1:
                sigma_rev[nb] += sigma_rev[cur]
    # 现在对每个节点 w，计算 sigma_uv(w) = sigma[w] * sigma_rev[w]，
    # 但注意 w 必须在从 u 到 v 的一条最短路径上，即 dist[u][w] + dist_rev[w] == dist[u][v]
    total_sp = sigma[v]  # σ_uv
    if total_sp == 0:
        return [0.0] * n
    f = [0.0] * n
    d_uv = dist[v]
    for w in range(n):
        if w == u or w == v:
            continue
        if dist[w] != -1 and dist_rev[w] != -1 and dist[w] + dist_rev[w] == d_uv:
            f[w] = (sigma[w] * sigma_rev[w]) / total_sp
    return f

def compute_omega(l2_sq: List[float], ell: int) -> float:
    # 避免除以零：如果所有 l2_sq 为零，则 ω*=0
    if max(l2_sq) == 0.0:
        return 0.0

    def obj(s):
        if s <= 0:
            return float('inf')
        # 使用 logsumexp 避免溢出
        max_val = max((s * s * sq) / (2 * ell * ell) for sq in l2_sq)
        total = 0.0
        for sq in l2_sq:
            total += math.exp((s * s * sq) / (2 * ell * ell) - max_val)
        log_sum = max_val + math.log(total)
        return log_sum / s

    # 寻找合适的搜索上界
    # 粗略估计：当 s 使得指数项变得非常大时，obj ~ s * max(l2_sq)/(2 ell^2) 线性增长
    # 可以设置上界为 10 或自适应
    s_low = 1e-6
    s_high = 10000.0
    # 如果把上界翻倍后目标函数还更小，说明在 s_high 处仍处于下降段
    while s_high < 1e6 and obj(s_high * 2) < obj(s_high):
        s_high *= 2
    # 黄金分割搜索
    phi = (math.sqrt(5) - 1) / 2
    a, b = s_low, s_high
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = obj(c)
    fd = obj(d)
    for _ in range(100):
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - phi * (b - a)
            fc = obj(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + phi * (b - a)
            fd = obj(d)
        if abs(b - a) < 1e-8:
            break
    s_opt = (a + b) / 2
    print("sopt",s_opt)
    return obj(s_opt)

def compute_delta(omega: float, ell: int, delta_param: float) -> float:
    """
    根据公式 (5) 和 (6) 计算 Δ_S
    """
    if ell == 0:
        return float('inf')
    log_term = math.log(2.0 / delta_param)
    # 计算 alpha (公式 5)
    sqrt_term = math.sqrt((2 * ell * omega + log_term) * log_term)
    alpha = log_term / (log_term + sqrt_term)
    if alpha <= 0 or alpha >= 1:
        # 回退处理
        alpha = 0.5
    term1 = omega / (1 - alpha)
    term2 = log_term / (2 * ell * alpha * (1 - alpha))
    term3 = math.sqrt(log_term / (2 * ell))
    print("term ",term1," ",term2," ",term3)
    return term1 + term2 + term3

def initial_sample_size(epsilon: float, delta_param: float) -> int:
    """
    公式 (11) 计算的初始样本大小 S1
    """
    log_term = math.log(2.0 / delta_param)
    numerator = (1 + 8 * epsilon + math.sqrt(1 + 16 * epsilon)) * log_term
    denominator = 4 * epsilon * epsilon
    return int(math.ceil(numerator / denominator))

def next_sample_size(omega: float, ell_cur: int, epsilon: float, delta_param: float) -> int:
    """
    自适应确定下一个样本大小，求解不等式 (12) 的三次方程（表1）
    返回大于 ell_cur 的最小整数解
    """
    # 如果 omega 为 0，则直接返回 2 * ell_cur 作为增长
    if omega <= 1e-12:
        return max(ell_cur + 1, int(ell_cur * 1.5))

    log_term = math.log(2.0 / delta_param)

    # 三次方程系数来自公式 (13)，变量 x 即 S_{i+1}
    # 方程为: -8 L^3 + L^2 (-16 ω + (1+4ε)^2) x - 4 L (ω-ε)^2 (1+4ε) x^2 + 4 (ω-ε)^4 x^3 = 0
    L = Decimal(log_term)
    w = Decimal(omega)
    e = Decimal(epsilon)
    # 注意: 原文中方程 (13) 的常数项为 -8 L^3，最后一项系数为 4 (ω-ε)^4
    a =  Decimal(4.0) * pow((w - e),Decimal(4.0))
    b = -Decimal(4.0) * L * (w - e) *(w-e)* (1 + Decimal(4.0) * e)
    c = L * L * (-Decimal(16.0) * w + (1 + Decimal(4.0) * e)*(1 + Decimal(4.0) * e))
    d = -Decimal(8.0) * L * L * L
    #print("abcd",a," ",b," ",c," ",d)
    # 解三次方程 a x^3 + b x^2 + c x + d = 0
    # 使用标准公式（实数根）
    if abs(a) < 1e-10:
        return max(ell_cur + 1, int(ell_cur * 1.5))
        # 降为二次
        if abs(b) < 1e-12:
            return max(ell_cur + 1, int(2 * ell_cur))
        # bx^2 + cx + d = 0
        disc = c * c - 4 * b * d
        if disc < 0:
            return max(ell_cur + 1, int(2 * ell_cur))
        sqrt_disc = pow(disc,Decimal(0.5))
        root1 = (-c - sqrt_disc) / (2 * b)
        root2 = (-c + sqrt_disc) / (2 * b)
        roots = [r for r in (root1, root2) if r > ell_cur and r > 0]
        if roots:
            return int(math.ceil(min(roots)))
        else:
            return max(ell_cur + 1, int(2 * ell_cur))
    # 一般三次方程
    # 先将方程化为 x^3 + B x^2 + C x + D = 0
    B = b / a
    C = c / a
    D = d / a
    # 消去二次项: x = y - B/3
    p = C - B * B / 3
    q = 2 * B * B * B / 27 - B * C / 3 + D
    # 判别式
    disc = (q / 2) ** 2 + (p / 3) ** 3
    if disc >= 0:
        sqrt_disc = pow(disc,Decimal(0.5))
        u = pow(-q / 2 + sqrt_disc,Decimal(1/3)) if (-q / 2 + sqrt_disc) >= 0 else -pow(-q / 2 + sqrt_disc,Decimal(1/3))
        v = pow(-q / 2 - sqrt_disc,Decimal(1/3)) if (-q / 2 - sqrt_disc) >= 0 else -pow(-q / 2 - sqrt_disc,Decimal(1/3))
        y=u+v
        roots = [y - B/3]
    else:
        # 三个实根
        r = pow((-p / 3) ** 3,Decimal(0.5))
        phi = mpmath.acos(-q / (2 * r)) if abs(q) <= 2 * r else math.pi
        # 使用三角函数形式
        y1 = 2 * pow((-p / 3),Decimal(0.5)) * mpmath.cos(phi / 3)
        y2 = 2 * pow((-p / 3),Decimal(0.5)) * mpmath.cos((phi + 2 * math.pi) / 3)
        y3 = 2 * pow((-p / 3),Decimal(0.5)) * mpmath.cos((phi + 4 * math.pi) / 3)
        roots = [y1 - B/3, y2 - B/3, y3 - B/3]
    #roots=np.roots([a,b,c,d])
    for r in roots:
        print("r",r)
    # 筛选 > ell_cur 且 > 0 的实数根
    valid_roots = [r for r in roots if r > ell_cur and r > 0]
    if valid_roots:
        next_s = min(valid_roots)
        # 验证不等式 (12) 是否满足（近似检查）
        # 若不满足，则适当放大
        # 简单起见，直接返回上取整
        return int(math.ceil(next_s))
    else:
        # 无有效根，采用几何增长
        return max(ell_cur + 1, int(ell_cur * 1.5))

def ABRA(G, epsilon: float, delta_param: float, max_samples):
    """
    ABRA-s 算法主函数
    返回每个节点的 BC 近似值
    """
    n = (len)(nx.nodes(G))
    # 初始化统计量
    l1_norm = [0.0] * n      # Σ f_w
    l2_sq = [0.0] * n        # Σ f_w^2
    ell = 0                  # 已采样对数
    # 初始样本大小
    S_cur = initial_sample_size(epsilon, delta_param)
    S_next = S_cur

    # 采样循环
    while ell < max_samples:
        target = min(max_samples,S_next)
        print("target",target)
        # 采样直到达到 target
        while ell < target:
            # 随机选取节点对 (u, v), u != v
            u = random.randrange(n)
            v = random.randrange(n)
            while u == v:
                v = random.randrange(n)
            # 计算 f_w(u,v)
            f_vals = compute_f_w(G, u, v)
            # 更新统计量
            for w in range(n):
                val = f_vals[w]
                l1_norm[w] += val
                l2_sq[w] += val * val
            ell += 1
            # 可选：每隔一定样本检查停止条件（论文中每次添加后都检查？ 为效率可每 100 次检查一次）
            # 这里为简单起见，在每次达到新的 target 后才计算
        # 计算当前样本下的 omega 和 delta
        omega = compute_omega(l2_sq, ell)
        delta_val = compute_delta(omega, ell, delta_param)
        #print(f"Iteration: samples={ell}, omega={omega:.6f}, delta={delta_val:.6f}, epsilon={epsilon}")
        if delta_val <= epsilon:
            # 停止条件满足
            break
        # 否则计算下一轮样本大小
        S_next = next_sample_size(omega, ell, epsilon, delta_param)
        print("Snext",S_next)
        if S_next <= ell:
            S_next = ell + 1   # 至少增加一个样本
    # 计算最终 BC 估计值
    bc=defaultdict(float)
    for w in range(n):
        bc[w] = l1_norm[w] / ell 
    return bc

class SILVAN:
    
    def __init__(self, G: nx.Graph, epsilon: float = 0.01, delta: float = 0.05,
                 c: int = 10, a: float = 2.0):
        self.G = G
        self.n = G.number_of_nodes()
        self.epsilon = epsilon
        self.delta = delta
        self.c = c  # 减少c可以显著加速
        self.a = a
        
        # 预计算节点映射
        self.nodes = list(G.nodes())
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        
        # 邻接表（比NetworkX的邻居查询更快）
        self.adj = {node: list(G.neighbors(node)) for node in self.nodes}
        
        # 增量式估计器
        self.b_sum = np.zeros(self.n)      # 分子累加
        self.w_sum = np.zeros(self.n)      # 方差累加
        self.sample_count = 0
        
    def fast_single_sample(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        单次采样：返回节点出现次数和路径长度（向量化实现）
        """
        s = random.choice(self.nodes)
        t = random.choice(self.nodes)
        if s == t:
            return np.zeros(self.n), np.zeros(self.n)
        
        # 使用BFS获取最短路径距离和前驱
        dist_s = {s: 0}
        pred_s = {s: []}
        queue = deque([s])
        
        while queue:
            u = queue.popleft()
            for v in self.adj[u]:
                if v not in dist_s:
                    dist_s[v] = dist_s[u] + 1
                    pred_s[v] = [u]
                    queue.append(v)
                elif dist_s[v] == dist_s[u] + 1:
                    pred_s[v].append(u)
        
        if t not in dist_s:
            return np.zeros(self.n), np.zeros(self.n)
        
        # 动态规划计算最短路径数 sigma_s
        sigma = defaultdict(int)
        sigma[s] = 1
        
        # 按距离排序处理节点
        nodes_ordered = sorted(dist_s.keys(), key=lambda x: dist_s[x])
        for u in nodes_ordered[1:]:  # 跳过s
            sigma[u] = sum(sigma[p] for p in pred_s[u] if p in sigma)
        
        if sigma[t] == 0:
            return np.zeros(self.n), np.zeros(self.n)
        
        # 采样一条路径（按概率回溯）
        path_nodes = set()
        curr = t
        while curr != s:
            path_nodes.add(curr)
            preds = [p for p in pred_s[curr] if p in sigma]
            if not preds:
                break
            weights = [sigma[p] for p in preds]
            total = sum(weights)
            if total == 0:
                break
            probs = [w/total for w in weights]
            curr = np.random.choice(preds, p=probs)
        path_nodes.add(s)
        
        # 构建特征向量（排除端点s,t）
        f = np.zeros(self.n)
        internal_nodes = path_nodes - {s, t}
        for node in internal_nodes:
            f[self.node_to_idx[node]] = 1.0
        
        # 返回特征和平方特征（用于方差估计）
        return f, f**2
    
    def batch_sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量采样，聚合结果
        """
        f_sum = np.zeros(self.n)
        f_sq_sum = np.zeros(self.n)
        
        for _ in range(batch_size):
            f, f_sq = self.fast_single_sample()
            f_sum += f
            f_sq_sum += f_sq
        
        return f_sum, f_sq_sum
    
    def empirical_peeling_fast(self) -> List[np.ndarray]:
        """
        快速经验剥离：基于当前估计的方差分层
        返回节点索引数组的列表
        """
        if self.sample_count == 0:
            return [np.arange(self.n)]
        
        # 计算经验wimpy方差
        w_v = self.w_sum / self.sample_count
        
        # 处理零方差节点
        w_v = np.clip(w_v, 1e-10, 1.0)
        
        # 计算分层索引
        log_vals = np.log(np.minimum(1.0 / w_v, self.sample_count)) / np.log(self.a)
        j_vals = np.ceil(log_vals).astype(int)
        j_vals = np.clip(j_vals, 0, 20)  # 限制最大层数
        
        # 按层分组
        max_j = j_vals.max()
        partitions = []
        for j in range(max_j + 1):
            mask = j_vals == j
            if mask.any():
                partitions.append(np.where(mask)[0])
        
        if not partitions:
            return [np.arange(self.n)]
        
        return partitions
    
    def run_fast(self, max_iterations: int = 100, batch_size: int = 100,
                 geometric_factor: float = 1.3) -> Dict:
        """
        快速运行SILVAN（优化版）
        """
        print(f"FastSILVAN: n={self.n}, epsilon={self.epsilon}, c={self.c}")
        
        # Phase 1: 快速初始采样用于经验剥离
        print("Phase 1: Initial sampling...")
        m_prime = min(200, max(50, int(10 / self.epsilon)))
        
        f_sum, f_sq_sum = self.batch_sample(m_prime)
        self.b_sum = f_sum
        self.w_sum = f_sq_sum
        self.sample_count = m_prime
        
        # 经验剥离
        partitions = self.empirical_peeling_fast()
        print(f"  Created {len(partitions)} partitions")
        
        # Phase 2: 渐进采样
        print("Phase 2: Progressive sampling...")

        # 填充初始样本
        # 这里简化：我们直接维护累加和，不存储所有样本
        
        iteration = 0
        total_samples = m_prime
        
        # 生成Rademacher向量（复用）
        np.random.seed(42)
        
        while iteration < max_iterations:
            iteration += 1
            
            # 计算当前估计
            b_tilde = self.b_sum / total_samples
            w_tilde = self.w_sum / total_samples
            
            # 检查停止条件：对所有分区计算epsilon边界
            max_eps = 0.0
            delta_i = self.delta / (2 ** min(iteration, 10))
            
            for partition in partitions:
                if len(partition) == 0:
                    continue
                
                # 简化的边界计算（避免存储所有样本）
                # 使用Bernstein型边界
                b_part = b_tilde[partition]
                w_part = w_tilde[partition]
                
                # 经验方差
                nu_part = w_part.max() if len(w_part) > 0 else 0.25
                
                # 简化的Rademacher边界（避免昂贵的MCERA）
                log_term = math.log(2.0 / delta_i) / total_samples
                R_part = math.sqrt(2 * nu_part * log_term) + log_term / 3
                
                # epsilon边界
                eps_part = 2 * R_part + math.sqrt(2 * log_term * (nu_part + 4 * R_part))
                eps_part += log_term / 3
                
                max_eps = max(max_eps, eps_part)
            
            if iteration % 10 == 0 or max_eps <= self.epsilon:
                print(f"  Iter {iteration}: samples={total_samples}, max_eps={max_eps:.6f}")
            
            if max_eps <= self.epsilon:
                #print(f"  ✓ Converged!")
                break
            
            # 几何增长采样
            new_batch = max(batch_size, int(batch_size * (geometric_factor ** iteration)))
            new_batch = min(new_batch, 10000)  # 限制最大批次
            
            # 批量采样
            f_sum_new, f_sq_new = self.batch_sample(new_batch)
            self.b_sum += f_sum_new
            self.w_sum += f_sq_new
            total_samples += new_batch
            self.sample_count = total_samples
            
            # 定期重新分区（每20次迭代）
            if iteration % 20 == 0:
                partitions = self.empirical_peeling_fast()
        
        # 最终结果
        b_final = self.b_sum / total_samples
        
        results = {node: b_final[self.node_to_idx[node]] for node in self.nodes}
        
        #print(f"\nFinished: {total_samples} samples, {iteration} iterations")
        return results


def _scatter_add(src, index, dim, dim_size):
    """
    MPS-safe scatter_add. index must already be broadcast to src.shape.
    Pre-allocate output with zeros, then scatter_add_.
    """
    if dim < 0:
        dim = src.dim() + dim
    out = torch.zeros(src.shape[:dim] + (dim_size,) + src.shape[dim+1:],
                      device=src.device, dtype=src.dtype)
    out.scatter_add_(dim, index, src)
    return out


class ParallelSIRBridgeCentrality(nn.Module):
    """
    GPU 并行 SIR 传播 + 反向贡献归因

    Parameters
    ----------
    N : int
    steps : int, 最大传播步数 T
    beta : float, 全局传播概率
    gamma : float, 恢复概率
    eps : float, 防除零
    device : torch device
    """

    def __init__(self, N, steps, beta=0.3, gamma=0.1, eps=1e-8, device=None):
        super().__init__()
        self.N = N
        self.steps = steps
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        self.device = device or torch.device('cpu')

    def forward(self, edge_index, source_nodes, edge_weight=None,
                return_hist=False, early_stop=False, threshold=1e-8):
        """
        Parameters
        ----------
        edge_index : [2, E]
        source_nodes : [B] 一批传播源
        edge_weight : [E] or None (per-edge beta)
        return_hist : bool
        early_stop : bool
        threshold : float

        Returns
        -------
        score_per_source : torch.Tensor [B, N]
        optional: S_hist, I_hist, R_hist, contrib_hist, new_inf_hist
        """
        src, dst = edge_index  # each [E]
        src = src.to(self.device)
        dst = dst.to(self.device)
        B = source_nodes.shape[0]
        N = self.N
        T = self.steps
        E = src.shape[0]

        # --- 预广播 index [B, E]（只执行一次，用于 scatter）---
        src_exp = src.unsqueeze(0).repeat(B, 1)
        dst_exp = dst.unsqueeze(0).repeat(B, 1)
        # src, dst 保持 1D 用于数据索引 I[:, src] -> [B, E]

        # --- 边级 beta [E] ---
        if edge_weight is not None:
            beta_e = float(self.beta) * edge_weight
        else:
            beta_e = torch.full((E,), float(self.beta),
                                device=self.device, dtype=torch.float32)

        # --- 初始化 [B, N] ---
        I = torch.zeros(B, N, device=self.device, dtype=torch.float32)
        S = torch.ones(B, N, device=self.device, dtype=torch.float32)
        R = torch.zeros(B, N, device=self.device, dtype=torch.float32)

        # I[b, source_nodes[b]] = 1.0; S[b, source_nodes[b]] = 0.0
        row_idx = torch.arange(B, device=self.device)
        I[row_idx, source_nodes] = 1.0
        S[row_idx, source_nodes] = 0.0

        # --- 缓存 [T+1, B, N] ---
        S_hist = torch.zeros(T + 1, B, N, device=self.device)
        I_hist = torch.zeros(T + 1, B, N, device=self.device)
        R_hist = torch.zeros(T + 1, B, N, device=self.device)
        new_inf_hist = torch.zeros(T + 1, B, N, device=self.device)

        S_hist[0] = S; I_hist[0] = I; R_hist[0] = R

        # --- 正向传播 ---
        for t in range(T):
            # 消息: log(1 - beta_e * I[src])  -> [B, E]  (1D index for gathering)
            I_src = I[:, src]
            prob = torch.clamp(1.0 - beta_e * I_src, min=self.eps, max=1.0)
            msg = torch.log(prob)

            # 聚合 by dst -> [B, N]  (2D expanded index for scatter)
            log_q = _scatter_add(msg, dst_exp, dim=-1, dim_size=N)
            q = torch.exp(log_q)

            # 状态更新
            new_inf = S * (1.0 - q)
            S_new = S * q
            I_new = (1.0 - self.gamma) * I + new_inf
            R_new = R + self.gamma * I
            S, I, R = S_new, I_new, R_new

            S_hist[t + 1] = S.clone()
            I_hist[t + 1] = I.clone()
            R_hist[t + 1] = R.clone()
            new_inf_hist[t + 1] = new_inf.clone()

            if early_stop and new_inf.sum() < threshold:
                for t2 in range(t + 2, T + 1):
                    S_hist[t2] = S.clone()
                    I_hist[t2] = I.clone()
                    R_hist[t2] = R.clone()
                break

        # --- 反向贡献归因 ---
        C = torch.zeros(T + 2, B, N, device=self.device)
        contrib_hist = torch.zeros(T + 1, B, N, device=self.device)

        for t in range(T - 1, -1, -1):
            # numerator: I_hist[t][src] * beta_e -> [B, E]  (1D index for gathering)
            numerator = I_hist[t][:, src] * beta_e

            # denom_j = scatter_add(numerator, dst) + eps -> [B, N]  (2D expanded for scatter)
            denom = _scatter_add(numerator, dst_exp, dim=-1, dim_size=N) + self.eps

            # delta for each edge: [B, E]  (1D index for gathering)
            delta = numerator / denom[:, dst]

            # E_i(t) = sum_j delta_ij * (delta_ij + E_j(t+1)) * I_j(t+1)
            downstream = (delta + C[t + 1][:, dst]) * I_hist[t + 1][:, dst]

            # aggregate back by src -> [B, N]  (2D expanded for scatter)
            C[t] = _scatter_add(delta * downstream, src_exp, dim=-1, dim_size=N)
            contrib_hist[t] = C[t].clone()

        # --- 分数 [B, N] ---
        score_per_source = (I_hist[:T + 1] * C[:T + 1]).sum(dim=0)

        if return_hist:
            return (score_per_source, S_hist, I_hist, R_hist,
                    contrib_hist, new_inf_hist)
        return score_per_source

def nx_to_pyg_edge_index(
    G: nx.Graph,
    directed: bool = False,
    weight_attr: Optional[str] = None,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[Any, int], Dict[int, Any]]:
    """
    将 NetworkX 图转换为 PyTorch Geometric 的 edge_index 格式。
    
    Returns
    -------
    edge_index : [2, num_edges] 的 LongTensor
    edge_weight : [num_edges] 的 FloatTensor（如果 weight_attr 为 None 则返回 None）
    node_map : 原始节点标签 -> 连续整数 ID
    rev_map : 连续整数 ID -> 原始节点标签
    """
    
    # 1. 构建节点映射
    nodes = list(G.nodes())
    node_map = {node: idx for idx, node in enumerate(nodes)}
    rev_map = {idx: node for node, idx in node_map.items()}
    
    # 2. 提取边
    edges = []
    weights = [] if weight_attr is not None else None
    
    for u, v, data in G.edges(data=True):
        edges.append((node_map[u], node_map[v]))
        if weight_attr is not None:
            weights.append(data.get(weight_attr, 1.0))
    
    # 3. 转 Tensor
    edge_index = torch.LongTensor(edges).t().contiguous()
    edge_weight = torch.FloatTensor(weights) if weight_attr is not None else None
    
    # 4. 无向图添加反向边
    if not directed and not G.is_directed():
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        if edge_weight is not None:
            edge_weight = torch.cat([edge_weight, edge_weight], dim=0)
    
    # 5. 设备转移
    if device is not None:
        edge_index = edge_index.to(device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)
    
    return edge_index, edge_weight, node_map, rev_map

def compute_gpu_bridge_centrality(G, beta=0.3, gamma=0.1, steps=10,
                                   sources=None, batch_size=256,
                                   weight_attr=None, directed=False,
                                   device=None, eps=1e-8, normalize=False,
                                   return_hist=False, early_stop=False,
                                   threshold=1e-8):
    """
    包装函数：NetworkX 图 -> GPU 传播桥梁性中心性

    Returns
    -------
    score : dict {original_node_id: score}
    runtime : float
    """

    t0 = time.perf_counter()

    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

    edge_index, edge_weight, node_map, rev_map = nx_to_pyg_edge_index(
        G, directed=directed, weight_attr=weight_attr, device=device)
    N = len(G.nodes())

    if sources is None:
        source_nodes = torch.arange(N, device=device)
    else:
        source_nodes = torch.tensor(
            [node_map[int(s)] for s in sources],
            device=device, dtype=torch.long)

    model = ParallelSIRBridgeCentrality(N, steps, beta, gamma, eps, device)

    total = len(source_nodes)
    all_sps = []

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_src = source_nodes[start:end]
        sps = model(edge_index, batch_src, edge_weight,
                    return_hist=False, early_stop=early_stop,
                    threshold=threshold)
        all_sps.append(sps)

    # 统一平均
    score = torch.cat(all_sps, dim=0).mean(dim=0).cpu().numpy()

    if normalize:
        mn, mx = score.min(), score.max()
        if mx - mn > 1e-15:
            score = (score - mn) / (mx - mn)

    runtime = time.perf_counter() - t0
    result = {rev_map[i]: float(score[i]) for i in range(N)}
    return result, runtime
# ---------------------------------------------------------
# 4. Top-k准确率
# ---------------------------------------------------------

def topk_accuracy(exact, approx, k):

    exact_top = sorted(exact.items(),
                       key=lambda x: x[1],
                       reverse=True)[:k]

    approx_top = sorted(approx.items(),
                        key=lambda x: x[1],
                        reverse=True)[:k]

    exact_nodes = set([x[0] for x in exact_top])
    approx_nodes = set([x[0] for x in approx_top])

    hit = len(exact_nodes & approx_nodes)

    return hit / k


def score_dict_to_array(scores, nodes):
    return np.array([scores.get(node, 0.0) for node in nodes], dtype=float)


def spearman_corr(exact, approx, nodes):
    data1 = score_dict_to_array(approx, nodes)
    rank1 = rankdata(data1, method='average')
    data2 = score_dict_to_array(exact, nodes)
    rank2 = rankdata(data2, method='average')
    rho, _ = spearmanr(rank1, rank2)
    return rho


def evaluate_method(method_name, runner, exact_scores, nodes, top_k):
    start = time.time()
    result = runner()

    if isinstance(result, tuple) and len(result) == 2:
        approx_scores, runtime = result
    else:
        approx_scores = result
        runtime = time.time() - start

    metrics = {
        'topk_accuracy': topk_accuracy(exact_scores, approx_scores, k=top_k),
        'runtime': runtime,
        'scores': approx_scores,
    }

    print(f"\n{method_name}")
    print("Time:", round(metrics['runtime'], 4), "s")
    return metrics


def write_metric_csv(filename, rows):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    file_path = RESULT_DIR / filename
    df = pd.DataFrame(rows)
    df.to_csv(file_path, mode='w', header=True, index=False, encoding='utf-8')


def make_experiment_id(graph_source, graph_name):
    safe_name = graph_name.replace('/', '_').replace(' ', '_')
    return f"{graph_source}__{safe_name}"


def save_experiment_record(record):
    RAW_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    experiment_id = make_experiment_id(record['graph_source'], record['graph_name'])
    file_path = RAW_RESULT_DIR / f"{experiment_id}.pkl"
    with file_path.open('wb') as f:
        pickle.dump(record, f)


def experiment_record_exists(graph_name, graph_source):
    experiment_id = make_experiment_id(graph_source, graph_name)
    file_path = RAW_RESULT_DIR / f"{experiment_id}.pkl"
    return file_path.exists()


def load_experiment_records():
    if not RAW_RESULT_DIR.exists():
        return []
    records = []
    for file_path in sorted(RAW_RESULT_DIR.glob('*.pkl')):
        with file_path.open('rb') as f:
            records.append(pickle.load(f))
    return records


def build_pairwise_spearman_rows(score_by_method, nodes, row_metadata):
    rows = []
    method_names = list(score_by_method.keys())
    for method_x in method_names:
        for method_y in method_names:
            row = dict(row_metadata)
            row['method_x'] = method_x
            row['method_y'] = method_y
            if method_x == method_y:
                row['spearman'] = 1.0
            else:
                row['spearman'] = spearman_corr(
                    score_by_method[method_x],
                    score_by_method[method_y],
                    nodes,
                )
            rows.append(row)
    return rows


def build_row_metadata(graph_name, graph_source, num_nodes, num_edges, epsilon, top_k, exact_time):
    return {
        'graph_name': graph_name,
        'graph_source': graph_source,
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'epsilon': epsilon,
        'top_k': top_k,
        'exact_time': exact_time,
    }


def evaluate_experiment_record(record):
    exact = record['exact_scores']
    nodes = sorted(exact.keys())
    method_order = record['method_order']

    time_rows = []
    acc_rows = []
    spearman_rows = []

    for epsilon_result in record['epsilon_results']:
        row_metadata = build_row_metadata(
            record['graph_name'],
            record['graph_source'],
            record['num_nodes'],
            record['num_edges'],
            epsilon_result['epsilon'],
            record['top_k'],
            record['exact_time'],
        )

        time_row = dict(row_metadata)
        acc_row = dict(row_metadata)
        score_by_method = {'EXACT': exact}

        for method_name in method_order:
            method_result = epsilon_result['method_results'][method_name]
            time_row[method_name] = method_result['runtime']
            acc_row[method_name] = topk_accuracy(exact, method_result['scores'], k=record['top_k'])
            score_by_method[method_name] = method_result['scores']

        time_rows.append(time_row)
        acc_rows.append(acc_row)
        spearman_rows.extend(build_pairwise_spearman_rows(score_by_method, nodes, row_metadata))

    return time_rows, acc_rows, spearman_rows

# ---------------------------------------------------------
# 5. 数据加载
# ---------------------------------------------------------

def relabel_graph_to_integers(graph: nx.Graph) -> nx.Graph:
    return nx.convert_node_labels_to_integers(graph, ordering='sorted')


def load_real_graph(graph_path: Path) -> nx.Graph:
    graph = nx.read_edgelist(graph_path, nodetype=int)
    if not isinstance(graph, nx.Graph) or graph.is_directed():
        graph = nx.Graph(graph)
    # ABRA 等方法依赖 0..n-1 的连续节点编号，真实数据需要先重编号。
    return relabel_graph_to_integers(graph)


def iter_real_graphs(data_dir: Path) -> List[Tuple[str, nx.Graph]]:
    graphs = []
    for graph_path in sorted(data_dir.glob('*/*.txt')):
        graphs.append((graph_path.parent.name, load_real_graph(graph_path)))
    return graphs


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)




def ensure_connected(G: nx.Graph) -> nx.Graph:
    if nx.is_connected(G):
        return relabel_graph_to_integers(G)
    largest_cc = max(nx.connected_components(G), key=len)
    return relabel_graph_to_integers(G.subgraph(largest_cc).copy())

def iter_synthetic_graphs(seed: int = SEED) -> Iterator[Tuple[str, nx.Graph]]:
    configs = [
        # 用于完整精度对比：exact BC + BP/RK/ABRA/SILVAN/GPU_SIR_BRIDGE
        ("BA1000", "BA", 1000, seed + 1),
        ("BA2000", "BA", 2000, seed + 2),
        ("BA3000", "BA", 3000, seed + 3),
        ("BA5000", "BA", 5000, seed + 4),

        ("WS1000", "WS", 1000, seed + 11),
        ("WS2000", "WS", 2000, seed + 12),
        ("WS3000", "WS", 3000, seed + 13),
        ("WS5000", "WS", 5000, seed + 14),

        ("ER1000", "ER", 1000, seed + 21),
        ("ER2000", "ER", 2000, seed + 22),
        ("ER3000", "ER", 3000, seed + 23),
        ("ER5000", "ER", 5000, seed + 24),

        # 只用于扩展性实验，不建议跑 exact BC
        ("BA10000", "BA", 10000, seed + 31),
        # ("WS10000", "WS", 10000, seed + 32),
        # ("ER10000", "ER", 10000, seed + 33),
    ]

    for name, graph_type, n, graph_seed in configs:
        if graph_type == "BA":
            # 平均度约 2m = 10
            G = nx.barabasi_albert_graph(n, 5, seed=graph_seed)
            G = relabel_graph_to_integers(G)

        elif graph_type == "WS":
            # k=10，平均度为 10
            G = nx.watts_strogatz_graph(n, 10, 0.05, seed=graph_seed)
            G = ensure_connected(G)

        elif graph_type == "ER":
            # 平均度约 10
            # m = n * avg_degree / 2 = 5n
            G = nx.gnm_random_graph(n, 5 * n, seed=graph_seed)
            G = ensure_connected(G)

        yield name, G


# ---------------------------------------------------------
# 6. 实验主函数
# ---------------------------------------------------------

def run_experiment(G, graph_name, graph_source):
    nodes = list(G.nodes())

    print("===================================")
    print("Graph Info")
    print("===================================")
    print("Dataset:", graph_name)
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    # -----------------------------------
    # 精确BC
    # -----------------------------------

    print("\nComputing Exact BC...")
    start = time.time()

    exact = exact_bc(G)
    exact_time = time.time() - start

    print("Exact Time:", round(exact_time, 4), "s")

    top_k = 100
    epsilon_sizes = [0.2]  # [0.5, 0.3, 0.2, 0.1]
    method_order = ['BP', 'RK', 'ABRA', 'SILVAN', 'GPU_SIR_BRIDGE']
    epsilon_results = []

    # -----------------------------------
    # 逐个实验
    # -----------------------------------

    for epsilon in epsilon_sizes:
        method_results = {
            'BP': evaluate_method(
                'BP',
                lambda: BP_sampling_bc(G, epsilon),
                exact,
                nodes,
                top_k,
            ),
            'RK': evaluate_method(
                'RK',
                lambda: RK(G, epsilon, 0.1),
                exact,
                nodes,
                top_k,
            ),
            'ABRA': evaluate_method(
                'ABRA',
                lambda: ABRA(G, epsilon, 0.1, 1e5),
                exact,
                nodes,
                top_k,
            ),
            'SILVAN': evaluate_method(
                'SILVAN',
                lambda: SILVAN(G, epsilon, delta=0.05, c=5).run_fast(
                    max_iterations=50,
                    batch_size=20,
                ),
                exact,
                nodes,
                top_k,
            ),
            'GPU_SIR_BRIDGE': evaluate_method(
                'GPU_SIR_BRIDGE',
                lambda: compute_gpu_bridge_centrality(
                    G,
                    beta=0.3,
                    gamma=0.1,
                    steps=10,
                    sources=None,
                    batch_size=256,
                    weight_attr=None,
                    directed=False,
                    device=None,
                    eps=1e-8,
                    normalize=False,
                    return_hist=False,
                    early_stop=False,
                    threshold=1e-8,
                ),
                exact,
                nodes,
                top_k,
            ),
        }

        epsilon_results.append({
            'epsilon': epsilon,
            'method_results': method_results,
        })

    record = {
        'graph_name': graph_name,
        'graph_source': graph_source,
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'top_k': top_k,
        'exact_time': exact_time,
        'exact_scores': exact,
        'method_order': method_order,
        'epsilon_results': epsilon_results,
    }
    save_experiment_record(record)
    return record


def run_all_experiments(seed: int = SEED):
    set_global_seed(seed)

    for index, (graph_name, graph) in enumerate(iter_synthetic_graphs(seed=seed), start=1):
        if experiment_record_exists(graph_name, "synthetic"):
            print(f"Skip existing experiment: synthetic/{graph_name}")
            continue
        set_global_seed(seed + index)
        run_experiment(graph, graph_name, "synthetic")

    real_graph_offset = 1000
    for index, (graph_name, graph) in enumerate(iter_real_graphs(DATA_DIR), start=1):
        if experiment_record_exists(graph_name, "real"):
            print(f"Skip existing experiment: real/{graph_name}")
            continue
        set_global_seed(seed + real_graph_offset + index)
        run_experiment(graph, graph_name, "real")


def evaluate_saved_experiments():
    records = load_experiment_records()
    if not records:
        print("No saved experiment records found in result/raw.")
        return

    time_rows = []
    acc_rows = []
    spearman_rows = []

    for record in records:
        record_time_rows, record_acc_rows, record_spearman_rows = evaluate_experiment_record(record)
        time_rows.extend(record_time_rows)
        acc_rows.extend(record_acc_rows)
        spearman_rows.extend(record_spearman_rows)

    write_metric_csv('time.csv', time_rows)
    write_metric_csv('acc.csv', acc_rows)
    write_metric_csv('spearman.csv', spearman_rows)


