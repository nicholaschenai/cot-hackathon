import pandas as pd
import matplotlib.pyplot as plt
import os

def calculate_accuracy(df):
    return df['is_correct'].mean()

def calculate_facet_accuracy(df, facet):
    return df.groupby(facet)['is_correct'].mean()

def plot_accuracies(experiments):
    # Plot and save raw accuracy
    raw_accuracies = {exp: calculate_accuracy(df) for exp, df in experiments.items()}
    plt.figure(figsize=(6, 6))
    bars = plt.bar(raw_accuracies.keys(), raw_accuracies.values(), color=plt.cm.tab20.colors)
    plt.title('Raw Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Experiment')
    for bar, (exp, df) in zip(bars, experiments.items()):
        count = df.shape[0]
        correct = df['is_correct'].sum()
        mean = raw_accuracies[exp]
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{correct}/{count}\n({mean:.2f})', ha='center', va='bottom')
    plt.tight_layout()
    if not os.path.exists('assets'):
        os.makedirs('assets')
    plt.savefig('assets/raw_accuracy.png')
    plt.close()

    # Plot and save accuracy by level
    level_accuracies = {exp: calculate_facet_accuracy(df, 'level') for exp, df in experiments.items()}
    levels = set()
    for acc in level_accuracies.values():
        levels.update(acc.index)
    levels = sorted(levels)
    
    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    index = range(len(levels))
    for i, (exp, acc) in enumerate(level_accuracies.items()):
        plt.bar([x + i * bar_width for x in index], [acc.get(level, 0) for level in levels], bar_width, label=exp)
        for j, level in enumerate(levels):
            count = experiments[exp][experiments[exp]['level'] == level].shape[0]
            correct = experiments[exp][(experiments[exp]['level'] == level) & (experiments[exp]['is_correct'] == 1)].shape[0]
            mean = acc.get(level, 0)
            plt.text(j + i * bar_width, mean, f'{correct}/{count}\n({mean:.2f})', ha='center', va='bottom')

    plt.title('Accuracy by Level')
    plt.ylabel('Accuracy')
    plt.xlabel('Level')
    plt.xticks([x + bar_width for x in index], levels)
    plt.legend()
    plt.tight_layout()
    plt.savefig('assets/accuracy_by_level.png')
    plt.close()

    # Plot and save accuracy by problem type
    type_accuracies = {exp: calculate_facet_accuracy(df, 'problem_type') for exp, df in experiments.items()}
    problem_types = set()
    for acc in type_accuracies.values():
        problem_types.update(acc.index)
    problem_types = sorted(problem_types)
    
    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    index = range(len(problem_types))
    for i, (exp, acc) in enumerate(type_accuracies.items()):
        plt.bar([x + i * bar_width for x in index], [acc.get(ptype, 0) for ptype in problem_types], bar_width, label=exp)
        for j, ptype in enumerate(problem_types):
            count = experiments[exp][experiments[exp]['problem_type'] == ptype].shape[0]
            correct = experiments[exp][(experiments[exp]['problem_type'] == ptype) & (experiments[exp]['is_correct'] == 1)].shape[0]
            mean = acc.get(ptype, 0)
            plt.text(j + i * bar_width, mean, f'{correct}/{count}\n({mean:.2f})', ha='center', va='bottom')

    plt.title('Accuracy by Problem Type')
    plt.ylabel('Accuracy')
    plt.xlabel('Problem Type')
    plt.xticks([x + bar_width for x in index], problem_types)
    plt.legend()
    plt.tight_layout()
    plt.savefig('assets/accuracy_by_problem_type.png')
    plt.close()

    # Get index, level, problem_type for those correct in 1 experiment but not the other and save to a CSV
    exp_names = list(experiments.keys())
    if len(exp_names) == 2:
        df1, df2 = experiments[exp_names[0]], experiments[exp_names[1]]
        right_in_first = df1[(df1['is_correct'] == 1) & (df2['is_correct'] == 0)]
        right_in_second = df2[(df2['is_correct'] == 1) & (df1['is_correct'] == 0)]
        right_in_first.to_csv(f'assets/right_in_{exp_names[0]}_not_in_{exp_names[1]}.csv', index=False)
        right_in_second.to_csv(f'assets/right_in_{exp_names[1]}_not_in_{exp_names[0]}.csv', index=False)


def load_experiment_data():
    experiments = {}
    experiment_names = ['CoT', 'ReAct']
    for exp in experiment_names:
        file_path = f'results/{exp}/comparison_results.csv'
        if os.path.exists(file_path):
            experiments[exp] = pd.read_csv(file_path)
        else:
            print(f"Warning: {file_path} not found. Skipping {exp}.")
    return experiments


def main():
    experiments = load_experiment_data()
    if experiments:
        for exp, df in experiments.items():
            df['total_count'] = df.groupby(['level', 'problem_type'])['is_correct'].transform('count')
            df['total_correct'] = df.groupby(['level', 'problem_type'])['is_correct'].transform('sum')
        plot_accuracies(experiments)


if __name__ == "__main__":
    main()
