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
        total_time = loaded_data['total_time']

        if env_name == 'Shepherding-v0':
            data[(num_herders, num_targets)]['v0'] = total_time
        else:
            if sim_dt not in data[(num_herders, num_targets)]['v1']:
                data[(num_herders, num_targets)]['v1'][sim_dt] = {}
            data[(num_herders, num_targets)]['v1'][sim_dt][solver] = total_time

    return data

def generate_latex_table(data, simulation_dts, solvers):
    header = """
\\begin{table}[ht]
\\centering
\\begin{tabular}{cccccccccc}
\\toprule
\\textbf{N} & \\textbf{M} & \\textbf{Sim Time v0} & """ + " & ".join([f"\\multicolumn{{2}}{{c}}{{{sim_dt}}}" for sim_dt in simulation_dts]) + """ \\\\
\\cmidrule(lr){4-11}
 &  &  & """ + " & ".join([f"\\textbf{{Euler}} & \\textbf{{SRI2}}" for _ in simulation_dts]) + """ \\\\
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
\\caption{Simulation Times for Different Configurations}
\\label{tab:sim_times}
\\end{table}
"""

    return header + "\n".join(rows) + footer

if __name__ == "__main__":
    folder = "simulations"
    simulation_dts = ["0.05", "0.01", "0.005", "0.001"]
    solvers = ["Euler", "SRI2"]
    data = extract_simulation_data(folder)
    latex_table = generate_latex_table(data, simulation_dts, solvers)

    with open("simulation_times_table.tex", "w") as f:
        f.write(latex_table)

    print("LaTeX table written to simulation_times_table.tex")
