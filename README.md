# High cofluctuation and Entropy 
This repo contains 
1. Code for the dataset generation and statistical tests for the paper 
  "Metrics of High Cofluctuation and Entropy to Describe Control of Cardiac Function in the Stellate Ganglion"
2. Figures for the Cofluctuation and the Entropy Metric for all the animals used in the paper 


The results pipeline are as follows
 MultiLevel1 -> MultiLevel2 -> Metrics (Neural Specificity, Coactivity) -> Statistical Tests


1. MultiLevel1
 The code and documentation of MultiLevel1 (Ml1) can be found in /MultiLevel1. This is the first code in the pipeline. ML1 takes spike recordings as input and generated spike locations along with the spike 
sign. The code runMutliLevel1.py is run for multichannel muti animal data. Currently the code is modified to take in data from the /Animals folder containing spike recordings of two animals for 2 minutes. 
The outputs of different functions in ML1 for the data provided has been uploaded to the folder, Example_plots. The outputs of ML1 are also present in the /Animals folder.

2. MultiLevel2 (ML2)
Similar to ML2 code and documentation of ML2 can be found in /MultiLevel2. Ml2 takes in the outputs (metadata and outspike.csv) dumped by ML1 in the /animals folder. Ml2 curated the spikes detected from ML1
to provide spike times, spike locations, spike amplitude and target at spike times. The code is modified to take in output from the /Animals folder and the outputs from ML2 are also stored in the /Animals 
folder

3.  Neural Specificity 
The code and documentation of the neural specificity code is uploaded to the /NeuralSpecificity code. Neural Specificity can be run using the runAttentionMetric.py code. The code has been modified to take one
animal data as input (available in supplement data link provided) and the outputs are stored in the /Animals folder. 

4. Coactivity and Cofluctuation 
Similar to neural specificity, runSpikeRateCoact.py in /CoactivityCofluctuationEventRate can be used to get the coactvity matrix an cofluctuation time series with data available in supplement link.   




