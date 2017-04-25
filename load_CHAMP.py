# Purpose: Script to load all the data from the testing
# Authors: Thomas Jeffries, Ryan Hardy
# Created: 20161114
# Modified: 20161114

import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
import pandas as pd
import re

#Define plot styles

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='w'
plt.rcParams['figure.facecolor']='w'
plt.rcParams['axes.edgecolor']='w'
plt.rcParams['axes.grid']=False
plt.rcParams['figure.subplot.bottom'] = 0.12
plt.rcParams['savefig.facecolor']='w'
plt.rcParams['axes.color_cycle'] = [u'#30a2da', u'#fc4f30', u'#e5ae38',  '#beaed4', '#fdc086']
#plt.rcParams.update({'figure.autolayout': True})

# Load the data dictionary
dic_file = "Data Dictionary - Spring 2017.xlsx"
dic = pd.read_excel(dic_file)

# Save the locations of the different questions for easy access
locs = np.unique(dic['Location'])

# Make the datatype array
datatype = np.zeros(len(dic['Data_type']), np.object)

# Loop through all the different types of data, and change the data type into
# the correct type. Everything was originally saved as an object.
for i in range(len(datatype)):
	if dic['Data_type'][i] == 'Binary':
		datatype[i] = np.object
	elif dic['Data_type'][i] == 'Categorical':
		datatype[i] = np.object
	elif dic['Data_type'][i] == 'Continuous':
		datatype[i] = np.float
	elif dic['Data_type'][i] == 'Count':
		datatype[i] = int
	elif dic['Data_type'][i] == 'Date':
		datatype[i] = np.object
	elif dic['Data_type'][i] == 'Free response':
		datatype[i] = np.object
	elif dic['Data_type'][i] == 'Comment':
		datatype[i] = np.object
	elif dic['Data_type'][i] == 'Nominal':
		datatype[i] = np.object
	elif dic['Data_type'][i] == 'Multiple selection':
		datatype[i] = np.object
	elif dic['Data_type'][i] == 'Ordinal':
		datatype[i] = float
	else:
		datatype[i] = np.object

# Save the file path and name of the data
responses_file = "Spring 2017 CHAMP Responses.xlsx"

#Output path
output_path = '../results'

# Load in the test data responses
df = pd.read_excel(responses_file, names=dic["Spring_2017_Question_Code"], parse_cols=len(dic))

# There should be no negative values in the data
# Zero is allowed because of number of flight hours
df.ix[np.where(df<0)] = np.nan

# Change the column names of the DataFrame
df.columns = dic["Spring_2017_Question_Code"]

#Exclude responses according to dictionary
'''exclude = np.zeros(df.shape, bool)
for i in range(len(df.T)):
	# Skips rows that have nothing in exclude_from column
	if pd.isnull(dic['Exclude_from'][i]):
		continue
	codes = dic['Exclude_from'][i].split(',')
	for code in codes:
		if len(code)==3:
			test = int(code[1:])
			exclude[df.crew_test==test, i] = True
		else:
			re.split("T|CM", code)[1:]
			test = int(code[2])
			crew = int(code[2])
			exclude[(df.crew_test==test)*(df.crew_test==crew), i] = True
df[exclude*(datatype!=np.object)] = np.nan'''

#===Ad-hoc data corrections go here===
#	Always justify corrections to the data

# Changing zeros to nan in # of spaceflights and hours in space
# Kimia Seyedmadani
df.ix[3,16] = np.nan
df.ix[3,17] = np.nan
#df.loc[(df.crew_id == 1)*(df.crew_test == 0), 'crew_name']

# Venkata Vimal Kakaraparti
df.ix[6,16] = np.nan
df.ix[6,17] = np.nan

# Venkata Vimal Kakaraparti
df.ix[9,16] = np.nan
df.ix[9,17] = np.nan

# Venkata Vimal Kakaraparti
df.ix[15,16] = np.nan
df.ix[15,17] = np.nan

# Venkata Vimal Kakaraparti
df.ix[25,16] = np.nan
df.ix[25,17] = np.nan

# Allison Anderson entered the wrong Crew Member ID
df.ix[2,3] = 3

# Thomas Jeffries entered the wrong Crew Member ID
df.ix[11,3] = 3

# Participants in test 3 given the wrong test number
df.ix[10,2] = 3
df.ix[11,2] = 3
df.ix[12,2] = 3
df.ix[13,2] = 3

'''
#T17CM4 swapped bideltoid breadth and thumb-tip reach
thumb = df.crew_shoulder.ix[(df.crew_id==4)*(df.crew_test==17)]
shoulder = df.crew_thumb.ix[(df.crew_id==4)*(df.crew_test==17)]
df.crew_shoulder.ix[(df.crew_id==4)*(df.crew_test==17)] = shoulder
df.crew_thumb.ix[(df.crew_id==4)*(df.crew_test==17)] = thumb

#T05CM1 entered his height as 59 instead of 69 inches
df.crew_height.ix[(df.crew_id==1)*(df.crew_test==5)] += 10

#Corbin Cowan entered the wrong Crew Member ID
df.crew_id[df.crew_name=='Corbin Cowan'] = 3
'''
#=== End ad-hoc corrections ===

# Classify participant experience
# Creates a matrix where columns are experience and rows are True/False values
# corresponding to each participant
experience = -DataFrame(index=df.index, columns= dic['Data_values'][13].split(';'),
				dtype=bool)
for i in range(len(df)):
	for j in range(experience.shape[-1]):
		if df.crew_experience_01[i] != df.crew_experience_01[i]:
			continue
		experience.ix[i, j] =  experience.columns[j] in df.crew_experience_01[i]


# === Classify questions by tag and location ===

# Pulls last 10 columns of data dictionary for tag matrix
tag_matrix = dic.ix[:,-10:].isin([1]).T
tag_matrix.columns = df.columns
tags = np.array(tag_matrix.index)
tags[3] = 'Location'
tag_matrix.index = tags

loc_matrix =  pd.get_dummies(dic['Location']).astype(bool)
loc_tag = (loc_matrix.astype(int).T.dot(np.matrix(tag_matrix).T))
loc_tag.index = ['Auxiliary', 'Airlock', 'Command', 'Emergency',
				'Exercise', 'Galley', 'Hatches', 'Hygiene', 'ICH',
				'Science', 'Sleep Stations', 'Technology Development']
loc_tag.columns = tag_matrix.index

# === End classify questions by tag and location ===

# Plot parameters
plt.rcParams.update({'figure.autolayout': True})

plt.figure(figsize=(8, 8))
plt.imshow(loc_tag.ix[:, :8], interpolation='nearest', cmap=plt.cm.Blues)
plt.gca().xaxis.tick_top()
plt.xticks(np.arange(8), loc_tag.columns, rotation=90)
plt.yticks(np.arange(len(loc_tag)), loc_tag.index)
plt.colorbar(label='Number of questions', format='%i')
plt.savefig('../results/figs/question_breakdown')

plt.close()

loc_tag.sum(0).plot('barh')
plt.xlabel("Number of questions")
plt.savefig('../results/figs/tag_breakdown')
plt.close()

loc_tag.sum(1).plot('barh')
plt.xlabel("Number of questions")
plt.savefig('../results/figs/loc_breakdown')
plt.close()

plt.rcParams.update({'figure.autolayout': False})


# Read in the particpant nationalities responses
current_nationalities_file = "../participant_current_nationalities.xlsx"
majors_file = "../participant_majors.xlsx"

cn_f = pd.read_excel(current_nationalities_file, header=0)
m_f  = pd.read_excel(majors_file, header=0)

# Identify the stem vs non-stem majors_file
m_f['stem_major'] = (m_f.aerospace_engineering==1) | (m_f.physics==1) | \
 	(m_f.math==1) | (m_f.astronautics_engineering) | \
	(m_f.chemical_engineering==1) | (m_f.biomedical_engineering==1) | \
	(m_f.electrical_engineering==1) | (m_f.mechanial_engineering==1) | \
	(m_f.computer_science==1)

m_f['non_stem'] = -m_f.stem_major

# Classify by gender
gender_ratio_test = pd.Series(index=df.crew_test.astype(np.int64).unique())
gender_ratio_data = pd.Series(index=df.index)
for i in gender_ratio_test.index:
	gender_ratio_test[i] = (df.crew_gender[df.crew_test==i]=='Male').mean()
	gender_ratio_data[df.crew_test==i] = gender_ratio_test[i]

#Categorize participants
categories = DataFrame(index=np.arange(len(df)))
categories['All'] = np.copy(np.ones(len(df), bool))
categories['Male'] = np.copy(df.crew_gender=='Male')
categories['Female'] = np.copy(df.crew_gender=='Female')
categories['Flight Experience'] = np.copy(df.crew_flight_01=='Yes')
categories['Flight Experience'] += np.copy(experience.ix[:, [0, 1, 2, 3, 4, 5, 6, 7]].sum(1) > 0)
categories['Habitat Experience'] = np.copy(experience.ix[:, [9, 13, 14, 15, 16, 17]].sum(1) > 0)
categories['Space Experience'] = np.copy(experience.ix[:, [7, 8, 9, 10, 11, 12]].sum(1) > 0)
categories['Expert'] = np.copy((experience.sum(1)>=3)*categories['Space Experience'])
categories['Any Experience'] = np.copy(experience.sum(1)>0)
categories['No Experience'] = np.copy(experience.sum(1)==0)
categories['US National'] = (cn_f['united_states']==1)
categories['International'] = -categories['US National']
categories['30 and older'] = np.copy(df.crew_age>= 30)
categories['Under 30'] = np.copy(df.crew_age< 30)
ansur_f = np.array([58.5, 14.9, 25.6])
ansur_m = np.array([76.6, 22.1, 35.8])
above = (df.crew_height > ansur_m[0])+(df.crew_shoulder > ansur_m[1])+(df.crew_thumb > ansur_m[2])
below = (df.crew_height < ansur_f[0])+(df.crew_shoulder < ansur_f[1])+(df.crew_thumb < ansur_f[2])
#categories['Above Limits'] = np.copy(above)
categories['Height <64\"'] = df.crew_height <= 64.
categories['Height 64-67\"'] = (df.crew_height > 64.)*(df.crew_height <= 66.6)
categories['Height 67-69\"'] = (df.crew_height > 66.6)*(df.crew_height <= 69.3)
categories['Height >69\"'] = (df.crew_height > 69.3)

#categories['Below Limits'] = np.copy(below)
categories['New Participant'] = np.copy(df.crew_prior=='No')
categories['Repeat Participant'] = np.copy(df.crew_prior.str.contains('Yes'))
categories['CHAMP'] = np.copy((df.crew_champ=='Yes (former)')+(df.crew_champ=='Yes (current)'))
categories['Non-CHAMP'] = np.copy(df.crew_champ=='No')

#categories['Mixed Gender'] = np.copy((gender_ratio_data % 1) !=0)
#categories['All Male'] = np.copy(gender_ratio_data ==1)

categories['CM1'] = np.copy(df.crew_id==1)
categories['CM2'] = np.copy(df.crew_id==2)
categories['CM3'] = np.copy(df.crew_id==3)
categories['CM4'] = np.copy(df.crew_id==4)

category_names = categories.columns

#Category classes
subcat_height = pd.concat([categories['All'],
							categories['Height <64\"'],
							categories['Height 64-67\"'],
							categories['Height 67-69\"'],
							categories['Height >69\"']], axis=1)
subcat_gender = pd.concat([categories['All'],
							categories['Male'],  categories['Female']], axis=1)
subcat_champ = pd.concat([categories['All'],
							categories['CHAMP'],  categories['Non-CHAMP']], axis=1)
subcat_repeat = pd.concat([categories['All'], categories['Repeat Participant'],
							categories['New Participant']], axis=1)
subcat_cm = pd.concat([categories['All'], categories['CM1'], categories['CM2'],
							categories['CM3'], categories['CM4']], axis=1)
subcat_national= pd.concat([categories['All'], categories['US National'],
							categories['International']], axis=1)
subcat_experience = pd.concat([categories['All'], categories['Flight Experience'],
								categories['Habitat Experience'],
								categories['Space Experience'],
								categories['Expert'],
								categories['Any Experience'],
								categories['No Experience']], axis=1)

subcats = [subcat_height, subcat_gender, subcat_experience, subcat_champ, subcat_repeat,
				subcat_cm, subcat_national]
subcat_names = ['height', 'gender', 'experience', 'champ', 'repeat', 'cm', 'national']

(categories.mean(0)*100)[::-1].plot(kind='barh')
plt.subplots_adjust(left=0.25)
plt.xlabel("%")
plt.savefig('../results/figs/categories')
plt.close()

plt.figure(figsize=(10, 10))
plt.imshow(np.array(np.ma.masked_values(categories.T.dot(categories.astype(int))/categories.sum(), 0), dtype=float), 
			interpolation='nearest', cmap=plt.cm.Reds)
plt.yticks(np.arange(25), category_names)
plt.gca().xaxis.tick_top()
plt.xticks(np.arange(25), category_names, rotation=90)
plt.colorbar(label="Number of partcipants", shrink=0.5)
plt.subplots_adjust(left=0.2, top=0.8, right=1)
plt.savefig('../results/figs/categories_cooccurrence')
plt.close()