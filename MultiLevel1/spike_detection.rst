MutiLevel1 - Spike Detection
=======================================

The first job in the pipeline involves detecting peaks or events crossing different amplitude threshold (called levels) in the preprocessed data in an iterative manner until a chosen minimum number of peaks is reached. At each iteration, the number of positive and negative peaks are compared and the higher of the comparison "wins the competition" to be masked and processed for the next pipeline. 

The location of positive or negative peaks at a level is stored with the help of the **getNewLevel** function described in Algorithm below. It checks for locations greater than a level in the data and applies a mask to make them unavailable for the next levels that are going to be lower than the current level. The masking is peformed with the help of a binary variable called "region" that stores a "1" at the locations of detected spikes and "0" otherwise. The same variable is also used to render a peak that was detected at a higher level unavailable for the current level. Once the locations are found, they are updated with the location of the maximum value found in a window of 6ms (120 points) starting at the stored location. Finally, the function returns a list of peak locations (SpikeList) that are spaced atleast 5-6ms apart for the next section of the algorithm

Algorithm I : getNewLevel::

        def getNewLevel(X, Region, Level, MinNewSpike, MinLevel):
        '''
    	getNewLevel returns a list of locations in a spike recordings above an amplitude level 


    	Args:
        	X (list): Spike recording of a channel in an experiment 
        	Level (float): Amplitude level used to detect locations in recording
        	MinNewSpike (int): Minimum number of dtected spikes for a level used as a parameter 
                MinLevel (float): Minumum level to end the spike detection used as a parameter

    	Outputs:
        	Level (float): Return the final Level
                SpikeList (list) : Spike Locations
    	'''
        	Level  =  Level + DeltaLevel
		SpikeWidth = 120
		SpikeList = [ ]
	
		while Spikes > MinNewSpike or Level < MinLevel :
	
			Level = level - DeltaLevel
			SpikeList = (X > Level) and (Region[X > Level] = 0)
			SpikeDomain = zip(SpikeIndex, SpikeList, SpikeList + SpikeWidth)
		
			for N,L,R in SpikeDomain :
				SpikeList[N] = argmax(X[L:R] + SpikeList[N]) * sum(Region[L:R] > 1)					
		
			SpikeList = SpikeList[N + 1] - SpikeList[N] > 100
		return Level, SpikeList

Besides its function of generating a list of peak locations crossing a level, The getNewLevel function is also used to obtain the result or "winner" of a competition between positive and negative peaks. The function is called twice for each iteration in the Multilevel 1 pipeline by passing the data and its flipped version as described in Algorithm I. This yields two lists per iteration, one for the positive and one for the negative peaks. The higher number of locations between the two lists "wins the competition" and is used for the next stage of analysis called **AnalysisLevelgetSpike**. 

The function AnalysisLevelGetSpike performs the tasks of masking the 6ms duration containing the peak thereby masking the entire spike or the action potential and curates the list of spikes based on factors such as proximity with neighboring spikes and a ringing effect. There are four more functions that are called sequentially in this function to perfom the above mentioned tasks. 
The first one is **getSpikeLevel** described in Algorithm II. This function stores the value of the peak level in place of its location and masks 6ms containing the peak - 1ms (20 points) before and 5ms (100 points) after the peak location - by setting the variable region to 1 similar to getNewLevel. It then returns the list of peak levels called "location" and region as output for the next stage. 

Algorithm II : getSpikeLevel::

	def getSpikeLevel(Location, Region, SpikeIndex, Level):

	'''
    	getSpikeLevel stores the value of the spike peak level in place of its location and 
        masks 6ms containing the peak - 1ms (20 points) before and 5ms (100 points) after peak location
        and performs making on each spike.  


    	Args:
        	Location (list): list used to store spike amplitude in spike locations
        	Region(list): List of maksed regions represented by 1s
        	SpikeIndex (list): list of spike locations obatined from GetNewLevel
                Level(float): Level value returned by GetNewlevel
    	Outputs:
        	Location (list): Returns the locations of spike with their amplitudes
                Region (list) : Returns the updated masking list
    	'''
		Location[SpikeIndex] = Level
		for EachSpike in SpikeIndex :	
			Region[EachSpike - 20 : EachSpike + 100] = 1
		return Location, Region

The second function, **getCleanedUp** masks spikes obtained from getSpikeLevel that are close to each other and also masks 3ms (60 points) before and after the spike.

Algorithm III : getCleanedUp::

	def getCleanedUp(X, Level, Region):

	'''
    	getCleanedUp returns an updated masking list after permforming a cleanup masking operation  


    	Args:
        	X (list): Spike recording of a channel in an experiment 
        	Level (float): Amplitude level used to detect locations in recording
		Region (list): Masking list returned from GetSpikeLevel
    	Outputs:
                Region (list) : Returns the updated masking list
    	'''

	
		CloseSpikeList = (Region[N] * Region[N + 120]) == 1
		for A, B in [CloseSpikeList, CloseSpikeList + 120]:		
			Region[A : B] = 1		
	
		AboveLevelList = X > Level * Region[X > Level] = 0		
		for C, D in [AboveLevelList - 60, AbovelevelList + 60]:		
			Region[C : D] = 1
		
		return Region

Following the spike clean up, the function **GetRingingClanedUp** is called.
It aims to eliminate the spikes with a ringing effect in the spikes list curated by getCleanedUp. This function uses the parameters "RingSecond" which is the duration to look for ringing in seconds for each spike, "RingNumPeriod" which indicated a period of ringing used as a threshold to eliminate ringing.The peaks (maxima) greater than RingCutoff in a duration of RingSecond after the Spike peak location is recorded using the DetectPeaks function. The ringing metric is then calculated. 
The spikes that exceed a threshold set by the "RingCutOff" parameter for the ringing metric are eliminated and new list is curated for the next stage in the function.

Algorithm IV : getRingingCleanedUp::

	def getRingingCleanedp(X, Location, Region, RingCutoff, RingThreshold, RingSecond, RingNumPeriod, Level, NeuralInterval):

	'''
    	getRingingCleanedUp returns an updated spike, location and masking list after removing spikes with ringing.   


    	Args:
        	X (list): Spike recording of a channel in an experiment 
        	Location (list): list used to store spike amplitude in spike locations
		Region(list): List of maksed regions represented by 1s
                RingCutoff (float) : Threshold placed on ringing metric
                RingSecond (float): duration of ringing
                RingNumPeriod (float):period of ringing used as a threshold
                Level (float): Amplitude level used to detect locations in recording
                NeuralInterval (float) : delta time of the neural recordings.
             
    	Outputs:
                Location (list) : Returns updated spike amplitides
                Region (list) : Returns the updated masking list
                LocationList (list) : Returns updated list of spike locations
    	'''

		RingHorizon  =  RingSecond/NeuralInterval
		
		LocationList = Location == Level
		
		for N in LocationList:
	
			MaxValueSet = DetectPeaks(X, mpd)
			MaxValueSet = MaxValueSet > RingCutoff
			if  Len(MaxValueSet) > RingNumPeriod}:
		
				Ring = sum( (MaxValueSet) * MaxValueSet/Len(MaxValueSet) ) #RINGING METRIC
				
				if Ring > RingThreshold
			
					Location[N : N + RingHorizon] = 0
					Region[N : N + RingHorizon] = 1	

		LocationList = Location == Level
		SpikeData = zip(LocationList, X[LocationList], LocationList - 20, LocationList + 100) 
		for N, XPeak, L, R in SpikeData:
	
			if (mean(X[L : R]) > 0.75 * Xpeak) or (Xpeak - mean(X[L:R]) < 1)
		
				Location[N] = 0		
		LocationList = Location == Level
		return Location, Region, LocationList

Follwing the elimination of spikes after ringing, **getRidOfIsland** is called.This function is used for two purposes. It is used to widen small islands detected using the Region varaible and it is used to merge small islands or small masked regions separated by les than the spike width 		
	
Algorithm V : getRidOfIsland::

	def getRidOfIsland(region):
        '''
        getRidOfIsland returns an updated masking list after removing islands 

    	Args:
		Region(list): List of maksed regions represented by 1s
             
    	Outputs:
                Region (list) : Returns the updated masking list
    	'''
		#JOB 1: Widen small islands
		SpikeWidth = 120
		DetectIsland = Region[N + 1] - Region[N]
		StartOf = DetectIsland > 0
		EndOf = DetectIsland < 0
		
		
		WidthOf = EndOf - StartOf
		if Len(WidthOf < Spikewidth) > 0
	
			for (Start, Widen) in zip(StartOf[WidthOf < Spikewidth], SpikeWidth - WidthOf[WidthOf < Spikewidth])
			
					Region[Start - Widen] = 1
			
	
		#JOB 2 : Merge small islands 
		DetectIsland = Region[N + 1] - Region[N]
		StartOf = DetectIsland > 0
		EndOf = DetectIsland < 0		
		SpaceBetween = StartOf[1:] - EndOf[:-1]
		if Len(SpaceBetween < Spikewidth) > 0
	
			For (EndPrev, StartNext) in (EndOf[SpaceBetween < Spikewidth], StartOf[SpaceBetween < Spikewidth + 1]):
		
				Region[EndPrev - StartNext] = 1

		return Region
		
The five functions are sequentially called in **AnalysisLevelgetSpike** after **getNewLevel** for either a larger plus or a minus level until a minmum level is reached. 
