import matplotlib.pyplot as plt

def plot_tuning_results(results, title="Hyperparameter Tuning Results"):
    """
    Visualize tuning results (e.g., loss vs hidden size).
    """
    # Simplified visualization of results list
    params = [str(r['params']) for r in results]
    losses = [r['loss'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.barh(params, losses, color='skyblue')
    plt.xlabel('Validation Loss')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def summarize_best_model(best_params, best_loss):
    """
    Print a summary of the best model found.
    """
    print("-" * 30)
    print("HYPERPARAMETER TUNING SUMMARY")
    print("-" * 30)
    print(f"Best Parameters: {best_params}")
    print(f"Best Validation Loss: {best_loss:.6f}")
    print("-" * 30)
