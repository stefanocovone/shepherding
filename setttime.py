import numpy as np
import os
import glob

def extract_simulation_data(folder):
    data = {}
    files = glob.glob(os.path.join(folder, "*.npz"))

    for file in files:
        filename = os.path.basename(file)
        parts = filename.split('_')
        env_name = parts[0]
        solver = parts[1]
        sim_dt = parts[2]
        num_herders = parts[3]
        num_targets = parts[4].split('.')[0]

        if (num_herders, num_targets) not in data:
            data[(num_herders, num_targets)] = {'v0': None, 'v1': {}}

        loaded_data = np.load(file)
        rewards = loaded_data['rewards']

        avg_reward = np.mean(rewards)

        if env_name == 'Shepherding-v0':
            data[(num_herders, num_targets)]['v0'] = avg_reward
        else:
            if sim_dt not in data[(num_herders, num_targets)]['v1']:
                data[(num_herders, num_targets)]['v1'][sim_dt] = {}
            data[(num_herders, num_targets)]['v1'][sim_dt][solver] = avg_reward

    return data

def generate_latex_table(data, simulation_dts, solvers):
    header = """
\\begin{table}[ht]
\\centering
\\caption{Average Cumulative Rewards for Different Configurations and Solvers}
\\begin{tabular}{cc|c|cccccccc}
\\toprule
\\multirow{2}{*}{\\textbf{N}} & \\multirow{2}{*}{\\textbf{M}} & \\textbf{Reward v0} & \\multicolumn{8}{c}{\\textbf{New Model}} \\\\
\\cmidrule(lr){3-11}
 &  & \\textbf{EM (0.05)} & \\multicolumn{2}{c}{0.05} & \\multicolumn{2}{c}{0.01} & \\multicolumn{2}{c}{0.005} & \\multicolumn{2}{c}{0.001} \\\\
\\cmidrule(lr){4-11}
 &  &  & \\textbf{EM} & \\textbf{SRI2} & \\textbf{EM} & \\textbf{SRI2} & \\textbf{EM} & \\textbf{SRI2} & \\textbf{EM} & \\textbf{SRI2} \\\\
\\midrule
"""

    configurations = [
        ('1', '1'),
        ('5', '20'),
        ('10', '50'),
        ('20', '100')
    ]

    rows = []
    for (num_herders, num_targets) in configurations:
        if (num_herders, num_targets) in data:
            times = data[(num_herders, num_targets)]
            row = [num_herders, num_targets, f"{times['v0']:.2f}" if times['v0'] is not None else "N/A"]
            for sim_dt in simulation_dts:
                for solver in solvers:
                    if sim_dt in times['v1'] and solver in times['v1'][sim_dt]:
                        row.append(f"{times['v1'][sim_dt][solver]:.2f}")
                    else:
                        row.append("N/A")
            rows.append(" & ".join(row) + " \\\\")

    footer = """
\\bottomrule
\\end{tabular}
\\label{tab:avg_rewards}
\\end{table}
"""

    return header + "\n".join(rows) + footer

if __name__ == "__main__":
    folder = "simulations"
    simulation_dts = ["0.05", "0.01", "0.005", "0.001"]
    solvers = ["Euler", "SRI2"]
    data = extract_simulation_data(folder)
    latex_table = generate_latex_table(data, simulation_dts, solvers)

    with open("average_rewards_table.tex", "w") as f:
        f.write(latex_table)

    print("LaTeX table written to average_rewards_table.tex")
