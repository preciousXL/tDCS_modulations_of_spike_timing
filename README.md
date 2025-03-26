# tDCS_modulations_of_spike_timing

Code associated with published paper: "Huang XL, Wei XL, Wang J, Yi GS. Effects of dendritic Ca2+ spike on the modulation of neural spike timing with transcranial direct current stimulation in cortical pyramidal neurons. Journal of Computational Neuroscience, 2025, 53(1): 25-36."

## Reproduction of paper results:
1. Compile the `.mod` files in the `Model_AmirDudai(2022)/mods/` and `Model_Schaefer(2003)/mod/`.
2. Run `Model_Schaefer(2003)/Schaefer_Model.ipynb` and `Model_Schaefer(2003)/MultiProcess_Schaefer.py` to genarate the data for plotting Figures 2-6.
4. Run `Model_AmirDudai(2022)/Hay_Model.ipynb` and `Model_AmirDudai(2022)/MultiProcess_Hay.py` to genarate the data for plotting Figures 7 and 8.
5. Run `PaperFigures.ipynb` to reproduce the paper results.
