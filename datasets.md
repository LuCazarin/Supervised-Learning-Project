# Candidate Datasets

## Abalone

[Link](https://archive.ics.uci.edu/dataset/1/abalone)

> Abalone é um molusco. Conhecido como lesma do mar com casco.

**Goal:** Predict the age of abalone from physical measurements

**Instances:** 4177

**8 Features:**

Name / Data Type / Measurement Unit / Description
-----------------------------
Sex / nominal / -- / M, F, and I (infant)
Length / continuous / mm / Longest shell measurement
Diameter	/ continuous / mm / perpendicular to length
Height / continuous / mm / with meat in shell
Whole weight / continuous / grams / whole abalone
Shucked weight / continuous	 / grams / weight of meat
Viscera weight / continuous / grams / gut weight (after bleeding)
Shell weight / continuous / grams / after being dried
Rings / integer / -- / +1.5 gives the age in years

## Phoneme

[Link](https://www.openml.org/search?type=data&sort=nr_of_likes&status=active&qualities.NumberOfFeatures=lte_10&qualities.NumberOfClasses=%3D_2&id=1489)

**Goal:** distinguish between nasal (class 0) and oral sounds 

**Instances:** 5404

**6 Features:** Target + the amplitudes of the five first harmonics AHi, normalised by the total energy Ene (integrated on all the frequencies): AHi/Ene

## Wine

[Link](https://archive.ics.uci.edu/dataset/109/wine)

**Goal:** Chemical analysis to determine the origin of wines

**Instances:** 178

**13 Features:**

Variable Name	Role	Type	Description	Units	Missing Values
class	Target	Categorical			no
Alcohol	Feature	Continuous			no
Malicacid	Feature	Continuous			no
Ash	Feature	Continuous			no
Alcalinity_of_ash	Feature	Continuous			no
Magnesium	Feature	Integer			no
Total_phenols	Feature	Continuous			no
Flavanoids	Feature	Continuous			no
Nonflavanoid_phenols	Feature	Continuous			no
Proanthocyanins	Feature	Continuous			no
