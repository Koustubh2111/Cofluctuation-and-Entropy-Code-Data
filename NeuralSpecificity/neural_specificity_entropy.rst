Neural Specificity Metric and Entropy
=======================================

The Neural Specificity Metric was develped to compare the degree to ehich stellate populations phase lock to a traget such as Left Ventricular Pressure (LVP) and Respiratpry pressure. The algorithm for the construction of the metric and subsequently entropy is described below.

1. The target array (Tarraw), target start time (Tarstart) and target interval (TarInterval) are read from the readTarget() function. The targets for each animals are stores as .mat files and are parsed into the code using the **H5py** python library. The list of channels (ChannelList) and spike times are read from the output of MultiLevel2 (Uncurated.csv)
2. In order to find a level range tp proceed with attention metric, A histogram of all the positive and negative spikes (PlusSpikes, MinusSpikes) is taken and a Savitsky-Golay (SavGol) filter is applied to them. This is done to smooth the histograms over a range of levels. The smallest peak in the relative difference among all level bins gives us the level bins with the least relative difference  (MinInd, MinValues). Based on the smallest relative differences we pick the level closest to 1.5 and obtain the suggested level (LevelValueSuggested). The level to extract spikeswill be the suggested level if level falls in between 1.25 and 2.
3. Following the extraction of spike times from all channels at the suggested level, the sliding histograms are constructed for each window.
	i. The first histogram matrix (Attention) is calculated by taking the histogram of the the target from the start to the end of the window and a savgol filter is applied for smoothing the histogram.
	ii. The second histogram matrix (Attention_Random) is calculated by taking the histogram of the target calculated at spike times in the same window.
	iii. The third histogram matri (Attention_Sample) is calculated by taking the histogram of the target at randomized spike times in the same window.
4. The neural specificity metric with respect to the target histogram (benchmark) of a window is given by : **Attention - Attention_Random**. The neural specificity with respect to randomized sampling of the target in the window is given by : **Attention_Sample - Attention_Random**
5. A hard threshold of 0.5 is applied to the matriced in step 4 so that the matrices have three values, 1,0 and -1 representing postive, near zero and negative neural specificity towards the target respectively. The neural specificity metrics are then plotted channel wise. 
6. Following the hard threshold, the shannon entropy (base 3) is caluculated for the first difference of the histograms in each of sliding windows. This entropy provides information about the change in neural specificity per window for the course of an animal's recordings. The csv's are then dumped for each event in the animal's experiment. 


Neural Specificity Algorithm : AttentionMetric::


	def AttentionMetric(*args):
	
	'''
        Attention metric calulates the neural specificity of a target, plots the neural specificity and dumps
	the corresponding csv file of the entropy. 

	args:
		hardthreshold (float): Threshold used for the metric [0,1]
		window (float) : Window size  
                num_bin(int) : Number of bins of each window 
		uncurated (bool) = True - Uncurated.csv is used for metric
		hasLVP (bool) = True - LVP is the target		

	outputs : Plots the metric and dumps the entropy 		
	''' 
	
		TarRaw, TarStart, TarInterval = readTarget()
		ChannelList, DataChannel, SpikeTime, Tar = readMultilevel2Spike()
		
		#Remove artifact after selecting time range
		ArtThreshold = NeuralInterval * 0.5 #ArtFraction = 0.5
		
		Index = RemoveArtifact(NumArtCompare, ChannelList, DataChannel, SpikeTime, ArtThreshold)
		
		SpikeTime, SpikeLevel, DataChannel, Tar = SpikeTime[Index], SpikeLevel[Index], DataChannel[Index], Tar[Index]
		
		#Choose Spike Level
		#Based on plus minus dfferences 
		
		BinWidth = 0.0025
		#isolating level values 
		PlusSpikes = SpikeLevel[SpikeLevel > 0]
		MinusSpikes = SpikeLevel[SpikeLevel < 0]
		
		BinEdges = linspace(0, 5, 5/BinWidth)
		
		#A histogram of plus and minus from level 0 to 5
		PlusHist = Histogram(PlusSpikes, BinEdges)
		MinusHist = Histogram(MinusSpikes, BinEdges)
		
		SmoothPlusHist = SavgolFilter(PlusHist)
		SmoothMinusHist = SavgolFilter(MinusHist)
		
		#Compute Desired level
		#Smoothing - get a histcount of spikes from levels 0-5 and smooth it with savgol, why here?
		
		RelDiff = (X - Y)/ max(X, Y) for [X, Y] in [SmoothPlusHist, SmoothMinusHist] 
		
		#Between number of spikes for plus and minus between, take the difference
		[MinInd, MinValues] = DetectPeaks(RelDiff) # small peaks
		

		#min Ind  - Index of rel diff equal to length of plus and minus hist for levels 0-5 - all the levels that have small differemces, close to noise floor
		#bin_edge[min_ind] selects the levels that have the smallest differences
		#  abs(bin_edge[min_ind] - 1.5) - from the levels that have the smallest diffrences, isolating the index of the one closest to 1.5
		#Between 2 and suggested level, choose minimum AND max between level suggested/2 to 1.25 - between 1.25 ad 2.0	
		
		#Finally, we choose 
		LevelValueSuggested = BinEdges[MinInd[argmin(BinEdges[MinInd - 1.5])]]
		
		If{1.25 < LevelValueSuggested < 2.0}
		{
			LevelPlus = LevelValueSuggested
			LevelMinus = LevelPlus
		}
	
		Index = (SpikeLevel > LevelPlus) or (SpikeLevel < -LevelMinus)
		
		SpikeTime, SpikeLevel, DataChannel, Tar = SpikeTime[Index], SpikeLevel[Index], DataChannel[Index], Tar[Index]
		
		#Sliding histogram
		
		SpikeTimes = SpikeTime + Window
		
		For{Spike in SpikeTimes}
		{
			EndIndex[N] = argmax(SpikeTime > Spike)
			N = N + 1
		}
	
		StartIndex = 1 : len(EndIndex)
		
		Attention = Zeros[len(StartIndex), NumBin]
		AttentionNotSet = Zeros(len(StartIndex))
		For{(Start, End) in (StartIndex, EndIndex)}
		{
			Attention[Start, :] = histogram(tar[Start : End], NumBin)
			Attention[Start, :] = SavGolFilter(Attention[Start, :], ???)
			AttentionNotSet[Start] = 1
		}
	
	
		#Attention Random
		RawStartIndex = (SpikeTime[StartIndex] - TarStart) / TarInterval
		RawEndIndex = RawStartIndex $+$ (Window / TarInterval)
		
		AttentionRandom = Zeros[len(StartIndex), NumBin]
		AttentionSample = Zeros[len(StartIndex), NumBin]
		
		AttentionRandomNotSet = Zeros(len(StartIndex))
		
		For{[I, Start, End, TarStart, TarEnd] in [(1 : len(StartIndex)), StartIndex, EndIndex, RawStartIndex, RawEndIndex]}
		{
			AttentionRandom[Start, :] = histogram(TarRaw[TarStart : TarEnd], NumBin)
			Sample = Random(TarStart, TarEnd, End - Start)
			AttentionSample[Start, :] = histogram(TarRaw[Sample], NumBin)
			AttentionRandomNotSet[Start] = 1
			
		}
		
		#Build Metric
		Attention = Attention - AttentionRandom
		Attention = Attention / std(Attention)
		
		
		
		AttentionSample = AttentionSample - AttentionRandom
		AttentionSample = AttentionSample / std(AttentionSample)
		 
		PlotResults(Attention, AttentionRandom, AttentionSample,  HardThreshold
		
		diffAttention = FirstDifference(Attention)
		getEntropy(Attention) - Calculates base 3 Shannon entropy of change in entropy and dumps the .csv file. 
	}
