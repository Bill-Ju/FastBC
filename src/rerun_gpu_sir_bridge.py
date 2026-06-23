from experient import SEED, evaluate_saved_experiments, rerun_method_for_saved_experiments


if __name__ == "__main__":
    rerun_method_for_saved_experiments("GPU_SIR_BRIDGE", seed=SEED)
    evaluate_saved_experiments()
