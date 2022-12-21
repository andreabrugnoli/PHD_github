import numpy as np
import matplotlib.pyplot as plt
from tools_plotting import setup
from matplotlib.ticker import FormatStrFormatter
import pickle

save_plots = input("Save plots? ")

path_fig = "/home/andrea/Pictures/PythonPlots/Hybridization_maxwell/"
bc_case = "_EH"
geo_case = "_3D"

res_file = open("results_hybridmaxwell.pkl", "rb")
results = pickle.load(res_file)

t_vec = results["t_span"]

Hdot_vec = results["Hdot_num_mid"]
bdflow_mid = results["flow_num_mid"]
bdflow_ex_vec = results["flow_ex_mid"]

dt = t_vec[-1] / (len(t_vec)-1)


plt.figure()
plt.plot(t_vec[1:]-dt/2, Hdot_vec - bdflow_mid, 'r-.')
plt.xlabel(r'Time $[\mathrm{s}]$')
# plt.ylabel(r'$P_h -<e^\partial_{h}| f^\partial_{h}>_{\partial M}$')
plt.title(r'Power balance conservation')

if save_plots:
    plt.savefig(path_fig + "pow_bal" + geo_case + bc_case + ".pdf", format="pdf")

plt.figure()
ax = plt.gca()
plt.plot(t_vec[1:]-dt/2, bdflow_mid - bdflow_ex_vec, 'r-.')
plt.xlabel(r'Time $[\mathrm{s}]$')
# plt.ylabel(r'$<e^\partial_{h}| f^\partial_{h}>_{\partial M} - <e^\partial_{\mathrm{ex}}| f^\partial_{\mathrm{ex}}>_{\partial M}$')
plt.title(r'Error numerical and exact boundary flow')

if save_plots:
    plt.savefig(path_fig + "bd_flowù" + geo_case + bc_case + ".pdf", format="pdf")

plt.show()

