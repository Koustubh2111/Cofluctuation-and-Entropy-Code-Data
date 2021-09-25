Cofluctuations and Event Rate Generation
=========================================

The coactivity matrix and cofluctuation matrix is obtained by a sliding window pearson correlation of the channels in the superdiagonal of the correlation matrix of a channels. The coactivity matrix is calculated using the following steps:

1. A dataframe containg the spike rates of all channels (df) is passed into the function. 
2. A window size small enough to work around the non stationaroty of the spike rates is selected for performing the correlations.
3. For every super diagonal in the correlation matrix i.e.channels that separated by multiples of intra electrode distances (500 Micro meter), the pearson correlation is calculated for all the sliding window for the course of the experiment (coactivity_vals). 
4. Follow the correlations from step 3, a correlation threshold is applied to each sliding window to get the cofluctuation series (coactivity_stats). The cofluctuation series represents the number of channels in the superdiaginal exceeding the correlation threshold (corr_threshold) in percentage. 

Coactivity and Cofluctuation Algorithm : Coactivity::


	def getSpikeRateCoactivity(df,Time,Window,Corr_Threshold):

	'''
	getSpikeRateCoactivity returns the coactivity matrix, cofluctuation series for an animal

	args:
		df - Dataframe containing spike rate of all channels of an animal
		Time (list) - time stamp series
		Window (float) - coactivity sliding window width
		Corr_Threshold (float) - Threshold for cofluctuation series

	Outputs:
		Dumps the cofluctuations series and plots the coactivity matrix.
	
	'''
		
		#Number of channels
		ch_numbers = [1,2,3,4,5,6,7,8,9,10] #10 example channels 
		num_ch = 10 (sent as an argument)

		#Oddness of windows
		half_win = int((window - 1) / 2)
		window = 2 * half_win + 1

		coactivity_vals = []

		for super_diagonal in range(1,num_ch):

			for index in range(0, num_ch - super_diagonal):

				n0 = ch_numbers[index]
				n1 = ch_numbers[index] + super_diagonal

				values = df[n0].rolling(window=window, center=True).corr(df[n1])[half_win:-half_win]
                
				# append values
                		coactivity_vals.append(values)

		[rows, cols] = coactivity_vals.shape
		coactivity_stats= []

		for row in rows:
			
			filter_row = coactivity_vals[row]			
			coactivity_stats[row] = len(filter_row>corr_threshold) * 100 #csv file is dumped


A second threshold is used for extracting events from the cofluctuation Series (threshold). The percentage cofluctuation of each sliding window above the the second threshold are classifies as events.
The event rate is calculated by dividing the number of events by the duration of the cofluctuation series. 

Event Rate ALgorithm : Event_Rate::
	
	def getEventRate(coactivity_stats, threshold, duration):
	
	'''
	getEventRate return the event rate from the Cofulctuation Series

	args:
		coactivity_stats : The cofluctuation series obtained from getSpikeRateCoactivity
		threshold : Threshold for extracting events 
		duration : Duration of cofluctuation series

	outputs:
		event_rate: returns the event rate

	'''
	
		state_array = []
		for val in coactivity_stats:
			if coactivity_stats[val] > threshold:
				state_array[val] = 1 
			else state_array[val] = 0

		i = 0
		for state in state_array:
			if state[i+1] - state[i] = 1:
				events.append(i + 1)
			i = i + 1

		event_rate = len(events)/duration
		return event_rate

	


			
