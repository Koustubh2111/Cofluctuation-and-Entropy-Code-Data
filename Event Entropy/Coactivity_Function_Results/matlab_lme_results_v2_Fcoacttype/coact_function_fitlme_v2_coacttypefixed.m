%% statistical questions:
%% 1. is entropy of event vs non event different regardless of animal type?
% % Entropy_Mean ~ Event_type + (Coact_Type) + (1|Animal) + (1|Channel)
% % Entropy_Mean ~ Event_type + (Coact_Type) + (1|Animal) + (1|Baseline_Entropy_Mean) + (1|Channel)
% % Entropy_Std ~ Event_type + (Coact_Type) + (1|Animal) + (1|Channel)
% % Entropy_Std ~ Event_type + (Coact_Type) + (1|Animal) + (1|Baseline_Entropy_Std) + (1|Channel)

%% 2. is entropy of event vs non event different between animal types?
Entropy_Mean ~ Event_type + Animal_Type + (Coact_Type) + (1|Animal) + (1|Channel)
Entropy_Mean ~ Event_type + Animal_Type + (Coact_Type) + (1|Animal) + (1|Baseline_Entropy_Mean) + (1|Channel)
Entropy_Std ~ Event_type + Animal_Type + (Coact_Type) + (1|Animal) + (1|Channel)
Entropy_Std ~ Event_type + Animal_Type + (Coact_Type) + (1|Animal) + (1|Baseline_Entropy_Std) + (1|Channel)

%% path

addpath('C:\Users\ngurel\Documents\Stellate_Recording_Files\Coactivity_Function\coact_function_0825_csvs')


%% Linear Mixed Effects:
T = readtable('coact_function_dataset_numerical.csv');

% 1. is entropy of event vs non event different regardless of animal type?
% Entropy_Mean ~ Event_type + (Coact_Type) + (1|Animal) + (1|Channel) : YES
% Entropy_Mean ~ Event_type + (Coact_Type) + (1|Animal) +
% (1|Baseline_Entropy_Mean) + (1|Channel) : YES
% Entropy_Std ~ Event_type + (Coact_Type) + (1|Animal) + (1|Channel) : YES
% Entropy_Std ~ Event_type + (Coact_Type) + (1|Animal) +
% (1|Baseline_Entropy_Std) + (1|Channel) : YES

lme=fitlme(T, 'Entropy_Mean ~ Event_type + Coact_Type + (1|Animal) + (1|Channel)') % p = 0.021851
lme=fitlme(T, 'Entropy_Mean ~ Event_type + Coact_Type + (1|Animal) + (1|Baseline_Entropy_Mean) + (1|Channel)') % p = 0.00061659 
lme=fitlme(T, ' Entropy_Std ~ Event_type + Coact_Type + (1|Animal) + (1|Channel)') % p =1.5192e-14
lme=fitlme(T, 'Entropy_Std ~ Event_type + Coact_Type + (1|Animal) + (1|Baseline_Entropy_Std) + (1|Channel)') % p = 3.9889e-17

% 2. is entropy of event vs non event different between animal types?
% Entropy_Mean ~ Event_type + Animal_Type + (Coact_Type) + (1|Animal) +
% (1|Channel) : NO
% Entropy_Mean ~ Event_type + Animal_Type + (Coact_Type) + (1|Animal) + (1|Baseline_Entropy_Mean) + (1|Channel): NO
% Entropy_Std ~ Event_type + Animal_Type + (Coact_Type) + (1|Animal) +
% (1|Channel): YES
% Entropy_Std ~ Event_type + Animal_Type + (Coact_Type) + (1|Animal) +
% (1|Baseline_Entropy_Std) + (1|Channel): YES

lme=fitlme(T, 'Entropy_Mean ~ Event_type + Animal_Type + Coact_Type + (1|Animal) + (1|Channel)') % % p(Event_type) = 0.021848,  p(Animal_Type) = 0.072525 
lme=fitlme(T, 'Entropy_Mean ~ Event_type + Animal_Type + Coact_Type + (1|Animal) + (1|Baseline_Entropy_Mean) + (1|Channel)') % p(Event_type) = 0.00061668 , p(Animal_Type) = 0.075029
lme=fitlme(T, 'Entropy_Std ~ Event_type + Animal_Type + Coact_Type + (1|Animal) + (1|Channel)') % p(Event_type) = 1.5252e-14  , p(Animal_Type)= 0.012124 
lme=fitlme(T, 'Entropy_Std ~ Event_type + Animal_Type + Coact_Type + (1|Animal) + (1|Baseline_Entropy_Std) + (1|Channel)') % p(Event_type) =  4.0063e-17, p(Animal_Type) =  0.011961










