## Running Neural Specificity Code
The neural specificity code has been provided to calculate and generate plots of the metric for the same 2 Animals and 2 channels used in Ml1 and ML2 in the ../Animals Folder
This code requries "Uncurated.csv" from ML2 present in ../Animals/AnimalX/AnimalX_ChannelY/ML2_Output/. 

The steps to run the code in the command line are

```{python}
cd ./NeuralSpecificity
python runAttentionMetric
```
This will create an output folder called 'NeuralSpecificity_20s_6minbuff' in ../Animals/AnimalX/AnimalX_ChannelY/. The '20s' represents the window width in meric calcuation. This can be changed in as a hyper parameter in the code depending on the duration of the spike recording. A window width of 20 minutes has been used in the paper for 6-8 hours spike recordings. The '6minbuff' represnets the buffer time before and after any events or interventions in the experiment toplot the metric. For te sake of simplicity, only a 2 min recording with no events have been shown in the repo. 


Detailed Documentation and Pseudocode have been provided in 

```{python}
./neural_specificity_entropy.rst

```
