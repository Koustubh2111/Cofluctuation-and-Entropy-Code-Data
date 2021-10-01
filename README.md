# High cofluctuation and Entropy Code and Data 
This repo contains code and surrogate data to run the codes for the paper 
  "Metrics of High Cofluctuation and Entropy to Describe Control of Cardiac Function in the Stellate Ganglion"

The results pipeline are to be run as follows
 MultiLevel1 -> MultiLevel2 -> Metrics (Neural Specificity, Coactivity)
 The results from above are used for the statistical tests used in the paper. 

1. **MultiLevel1 (ML1)**
The code and documentation (./MultiLevel1/spike_detection.rst) of MultiLevel1 (Ml1) can be found in **./MultiLevel1/**. This is the first code in the pipeline. ML1 takes spike recordings as input and generated spike locations along with the spike sign. The code runMutliLevel1.py is run for multichannel muti animal data. Currently the code is modified to take in data from the /Animals folder containing spike recordings of two animals for 2 minutes. 
The outputs of different functions in ML1 for the data provided has been uploaded to the folder, Example_plots. The outputs of ML1 are also present in the /Animals folder.

2. **MultiLevel2 (ML2)**
Similar to ML2 code and documentation (./MultiLevel2/spike_curation.rst) of ML2 can be found in **./MultiLevel2**. Ml2 takes in the outputs (metadata and outspike.csv) dumped by ML1 in the /animals folder. Ml2 curated the spikes detected from ML1 to provide spike times, spike locations, spike amplitude and target at spike times. The code is modified to take in output from the /Animals folder and the outputs from ML2 are also stored in the /Animals folder

3.  **Neural Specificity** 
The code and documentation of the neural specificity code (./NeuralSpecificity/neural_specificity_entropy.rst) is uploaded to the **./NeuralSpecificity** folder. Neural Specificity can be run using the runAttentionMetric.py code. The code has been modified to take the same two animal two channel data as input used in ML1 and ML2and the outputs are stored in the /Animals folder. 

4. **Coactivity and Cofluctuation** 
The code and documentation (./CoactivityCofluctuationEventRate/cofluctuations_event_generation.rst) of coactvity and cofluctuation is uploaded to the **./CoactivityCofluctuationEventRate** folder. The code used data from 4 channels (2mins) of single animal to obtain the coactivity matrix and cofluctuation series. 




