MultiLevel2 - Spike Curation
=======================================

MultiLevel2 (ML2) is the next job in the pipeline after MultiLevel1 (ML1). It used the metadata and output spike outputs from ML1 to generated another metadata file and other spike information files. Although ML2 is built to generate different kinds of spike information including a list of curated, uncurated and spikes for template matching information, this documentation will focus on the uncurated spike information. 

Writing Uncurated csv algorithm : Write_MultiLevel2()::
	
	def Write_MultiLevel2(*args):
	
	'''
	Writes all the spike information into a csv called Uncurated.csv

	'''

	if spike_plus[0] > 0 and spike_minus[0] > 0:

		spike = [spike_plus, spike_minus]

		prom = [prom_plus, prom_minus]

		width = [width_plus, width_minus]
	else:
		[spike_plus, prom_plus, width_plus] or [spike_minus, prom_minus, width_minus]

	index = np.argsort(spike)

	spike_time = neural_start + neural_interval * spike

	spike[index], prom[index], width[index], plus_minus_id[index]

	start_time = Max(neural_start, lvp_start, resp_start)

	
    	neural_end = neural_start + Max(spike) * neural_interval
    	lvp_end = lvp_start + Max(len(lvpraw)) * lvp_interval
    	resp_end = resp_start + Max(len(respraw)) * resp_interval    

    
    	end_time = Min(end for end in [neural_end, lvp_end, resp_end] if end > 0.0])

	index = spike_time > start_time AND spike_time < end_time
	spike[index], spike_time[index], spike_x_level[index], plus_minus_id[index], prom[index], width[index]

	#Spike target values
	lvp_index = (spike_time - lvp_start) / lvp_interval
	lvp = lvpraw[lvp_index]
	
	resp_index = (spike_time - resp_start) / resp_interval
	resp = respraw[resp_index] 

	 df = pd.DataFrame(\
        	{'spike_time':spike_time,\
         	'spike':spike,\
         	'lvp':lvp,\
         	'resp':resp,\
         	'plus_minus_id':plus_minus_id,\
         	'prom':prom,\
         	'width':width,\
         	'spike_x_level':spike_x_level,\
        	 })
	df.to_csv('Uncurated.csv')


ML2 Algorithm : AnalysisLevelUncurated::

	def AnalysisLevelUncurated(x, location_list,mean_shift_n,width_lower_bound,left,right):
	'''
	AnalysisLevelUncurated returns the Uncurated.csv for a channel of an animal containing information such
	as spike time, spike level, spike location, spike target value etc

	args:
		x(list) - smoothed list of spike recordings of a channel
		mean_shift_n - Parameter used in obtaining prominence and width of an spike

	Outputs:
		Prom, Width, Location_list

	'''

	[Prom, Width, Locationlist] =   getAllPromWidth(X,LocationList, mean_shift_n,width_lower_bound,left,right))

        return prom, width, spike_list

Prominence and Width Algorithm : getAllPromWidth::

	def getAllPromWidth():


		SpikeWidth = 120
		for N in LocationList
		
			Xn = X[N - 20 : N + 100]
			Xn = Xn - Mean(Xn[0 : mean_shift_n])
			MinIndex = 22 + argmin(DetectPeaks(-Xn[22 : 120]))
		
			Prom[N] = X[20] - X[MinIndex]
			Width[N] = MinIndex - 20
		return Prom, Width, LocationList



	
			




		


                
