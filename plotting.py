import matplotlib.pyplot as plt


def plot_results(results, xlabel, ylabel, title):
    plt.figure(figsize=(10, 6))
    for method, data in results.items():
        if len(data['x']) > 0 and len(data['y']) > 0:
            plt.plot(data['x'], data['y'], label=method, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()