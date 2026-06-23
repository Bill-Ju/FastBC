import networkx as nx

from experient import (
    compute_gpu_bridge_centrality,
    compute_gpu_bridge_centrality_random_sources,
)


def top_k_items(score_dict, k=10):
    return sorted(score_dict.items(), key=lambda item: item[1], reverse=True)[:k]


def validate_scores(name, result, runtime, num_nodes):
    assert len(result) == num_nodes, f"{name}: unexpected result length"
    assert all(isinstance(score, float) for score in result.values()), f"{name}: non-float score found"
    print(f"{name} runtime: {runtime:.4f}s")
    print(f"{name} top-10: {top_k_items(result, k=10)}")


if __name__ == "__main__":
    G = nx.barabasi_albert_graph(100, 3, seed=42)

    single_result, single_runtime = compute_gpu_bridge_centrality(
        G,
        beta=0.3,
        gamma=0.1,
        steps=10,
        batch_size=64,
    )
    random_result, random_runtime = compute_gpu_bridge_centrality_random_sources(
        G,
        beta=0.3,
        gamma=0.1,
        steps=10,
        num_samples=128,
        seed_ratio=0.05,
        batch_size=32,
        random_seed=42,
    )

    validate_scores("single_source", single_result, single_runtime, G.number_of_nodes())
    validate_scores("random_multi_source", random_result, random_runtime, G.number_of_nodes())
