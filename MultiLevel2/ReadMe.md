# Running MultiLevel 2

MultiLevel2 (ML2) uses the ouputs metadataMultilevel1_Ax_Channelx.txt and outputSpike_Ax_Channelx.csv from MultiLevel1 (ML1) as inputs.

ML2 can be run using the commands
```{python}
cd ./MultiLevel2
python runMultiLevel2.py
```
MultiLevel2 must be run after ML1. The ouputs are stored in a new folder called ML_Output2
The path of the output is similar to ML1  - ..\Animals\Animal_Data\Animalx\ML2_Output\Ax_Channelx
ML2 produces many six outputs but the ones that are used in the computation of the metrics are

1. metadata_MultiLevel2_Ax_Channelx.txt - contains metadata of all parameters used in ML2
2. Uncurated_Ax_Channelx.csv - contains spike locations, spike times, spike level etc of detected spikes. 

The outputs are available in \Animals\Animal_Data\Animalx\ML2_Output

In order to run ML2 The folders Multilevel1, Multilevel2 and Animals must be downloaded. 

Detailed Documentation and Psuedocode are provided in
```{python}
./spike_curation.rst
```
