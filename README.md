# P-C-Insurance

import pandas as pd
import pandas_profiling
import seaborn

df = pd.read_csv("IOWA.csv")
df.head()
df.info()

eda_report=pandas_profiling.ProfileReport(df)
eda_report.to_file("iowa.html")

eda_report=pandas_profiling.ProfileReport(df)
eda_report.to_file("iowa.html")

#download the files
from google.colab import files
files.download("iowa.html")
pandas_profiling.ProfileReport(df

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity ="all"

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

df = pd.read_csv('IOWA.csv', header=0)

df.head()

Year 	Iowa Code Chapter 	State 	Company Name 	Line of Insurance 	Premiums Written 	Losses Paid 	Taxes Paid 	NAIC Number 	Iowa Company Code
0 	2017 	515.48 	WI 	1st Auto & Casualty Insurance Company 	Farmowners Multiple Peril 	73953 	3900 	754 	44725 	2894
1 	2017 	515.48 	WI 	1st Auto & Casualty Insurance Company 	Homeowners Multiple Peril 	30778 	7500 	314 	44725 	2894
2 	2017 	515.48 	WI 	1st Auto & Casualty Insurance Company 	Other Liability - Occcurence 	50931 	0 	520 	44725 	2894
3 	2017 	515.48 	WI 	1st Auto & Casualty Insurance Company 	Other Private Passenger Auto Liability 	1434294 	826321 	14632 	44725 	2894
4 	2017 	515.48 	WI 	1st Auto & Casualty Insurance Company 	Other Commerical Auto Liability 	72651 	28321 	741 	44725 	2894


Overview

Dataset info
Number of variables 	10
Number of observations 	21156
Total Missing (%) 	0.0%
Total size in memory 	1.6 MiB
Average record size in memory 	80.0 B

Variables types
Numeric 	4
Categorical 	4
Boolean 	0
Date 	0
Text (Unique) 	0
Rejected 	2
Unsupported 	0

Warnings

    Company Name has a high cardinality: 856 distinct values Warning
    Losses Paid is highly correlated with Premiums Written (ρ = 0.91409) Rejected
    Premiums Written has 17011 / 80.4% zeros Zeros
    Premiums Written is highly skewed (γ1 = 33.921) Skewed
    Taxes Paid has 16442 / 77.7% zeros Zeros
    Taxes Paid is highly skewed (γ1 = 33.242) Skewed
    Year has constant value 2017 Rejected

Variables

Company Name
Categorical
Distinct count 	856
Unique (%) 	4.0%
Missing (%) 	0.0%
Missing (n) 	0
SU Insurance Company 	
 
43
Southern General Insurance Company 	
 
43
Valley Forge Insurance Company 	
 
43
Other values (853) 	
21027
Toggle details
Value 	Count 	Frequency (%) 	 
SU Insurance Company 	43 	0.2% 	
 
Southern General Insurance Company 	43 	0.2% 	
 
Valley Forge Insurance Company 	43 	0.2% 	
 
Integrity Mutual Insurance Company 	43 	0.2% 	
 
North Star Mutual Insurance Company 	43 	0.2% 	
 
Toyota Motor Insurance Company 	43 	0.2% 	
 
Pekin Insurance Company 	43 	0.2% 	
 
Peachtree Casualty Insurance Company 	43 	0.2% 	
 
Travelers Property Casualty Company of America 	43 	0.2% 	
 
Service American Indemnity Company 	43 	0.2% 	
 
Other values (846) 	20726 	98.0% 	
 

Iowa Code Chapter
Categorical
Distinct count 	4
Unique (%) 	0.0%
Missing (%) 	0.2%
Missing (n) 	43
515.48 	
20202
515C 	
 
609
520 	
 
302
(Missing) 	
 
43
Toggle details
Value 	Count 	Frequency (%) 	 
515.48 	20202 	95.5% 	
 
515C 	609 	2.9% 	
 
520 	302 	1.4% 	
 
(Missing) 	43 	0.2% 	
 

Iowa Company Code
Numeric
Distinct count 	856
Unique (%) 	4.0%
Missing (%) 	0.0%
Missing (n) 	0
Infinite (%) 	0.0%
Infinite (n) 	0
Mean 	2131.9
Minimum 	193
Maximum 	3246
Zeros (%) 	0.0%
Toggle details

    Statistics
    Histogram
    Common Values
    Extreme Values

Quantile statistics
Minimum 	193
5-th percentile 	615
Q1 	1598
Median 	2356
Q3 	2848
95-th percentile 	3164
Maximum 	3246
Range 	3053
Interquartile range 	1250

Descriptive statistics
Standard deviation 	840.37
Coef of variation 	0.39419
Kurtosis 	-0.85195
Mean 	2131.9
MAD 	713.34
Skewness 	-0.6242
Sum 	45102820
Variance 	706220
Memory size 	165.4 KiB
Value 	Count 	Frequency (%) 	 
2015 	43 	0.2% 	
 
2345 	43 	0.2% 	
 
2774 	43 	0.2% 	
 
695 	43 	0.2% 	
 
663 	43 	0.2% 	
 
2710 	43 	0.2% 	
 
2582 	43 	0.2% 	
 
2566 	43 	0.2% 	
 
2502 	43 	0.2% 	
 
407 	43 	0.2% 	
 
Other values (846) 	20726 	98.0% 	
 

Minimum 5 values
Value 	Count 	Frequency (%) 	 
193 	43 	0.2% 	
 
196 	43 	0.2% 	
 
406 	43 	0.2% 	
 
407 	43 	0.2% 	
 
408 	20 	0.1% 	
 

Maximum 5 values
Value 	Count 	Frequency (%) 	 
3239 	43 	0.2% 	
 
3243 	43 	0.2% 	
 
3244 	43 	0.2% 	
 
3245 	43 	0.2% 	
 
3246 	1 	0.0% 	
 

Line of Insurance
Categorical
Distinct count 	43
Unique (%) 	0.2%
Missing (%) 	0.0%
Missing (n) 	0
Other Liability - Occcurence 	
 
639
Workers Compensation 	
 
622
Inland Marine 	
 
601
Other values (40) 	
19294
Toggle details
Value 	Count 	Frequency (%) 	 
Other Liability - Occcurence 	639 	3.0% 	
 
Workers Compensation 	622 	2.9% 	
 
Inland Marine 	601 	2.8% 	
 
Other Commerical Auto Liability 	594 	2.8% 	
 
Commercial Auto Physical Damage 	587 	2.8% 	
 
Commerical Multiple Peril (Non-liability portion) 	569 	2.7% 	
 
Commerical Multiple Peril (Liability portion) 	563 	2.7% 	
 
Allied Lines 	559 	2.6% 	
 
Fire 	549 	2.6% 	
 
Private Passenger Physical Damage 	543 	2.6% 	
 
Other values (33) 	15330 	72.5% 	
 

Losses Paid
Highly correlated

This variable is highly correlated with Premiums Written and should be ignored for analysis
Correlation 	0.91409

NAIC Number
Numeric
Distinct count 	856
Unique (%) 	4.0%
Missing (%) 	0.0%
Missing (n) 	0
Infinite (%) 	0.0%
Infinite (n) 	0
Mean 	24291
Minimum 	10003
Maximum 	45934
Zeros (%) 	0.0%
Toggle details

    Statistics
    Histogram
    Common Values
    Extreme Values

Quantile statistics
Minimum 	10003
5-th percentile 	10749
Q1 	15756
Median 	23469
Q3 	31470
95-th percentile 	41653
Maximum 	45934
Range 	35931
Interquartile range 	15714

Descriptive statistics
Standard deviation 	9720.6
Coef of variation 	0.40017
Kurtosis 	-0.92923
Mean 	24291
MAD 	7986.1
Skewness 	0.33866
Sum 	513904624
Variance 	94489000
Memory size 	165.4 KiB
Value 	Count 	Frequency (%) 	 
10227 	43 	0.2% 	
 
10219 	43 	0.2% 	
 
31208 	43 	0.2% 	
 
29017 	43 	0.2% 	
 
37141 	43 	0.2% 	
 
22748 	43 	0.2% 	
 
26794 	43 	0.2% 	
 
30872 	43 	0.2% 	
 
20621 	43 	0.2% 	
 
39012 	43 	0.2% 	
 
Other values (846) 	20726 	98.0% 	
 

Minimum 5 values
Value 	Count 	Frequency (%) 	 
10003 	1 	0.0% 	
 
10006 	43 	0.2% 	
 
10014 	5 	0.0% 	
 
10030 	14 	0.1% 	
 
10051 	43 	0.2% 	
 

Maximum 5 values
Value 	Count 	Frequency (%) 	 
43753 	43 	0.2% 	
 
44393 	13 	0.1% 	
 
44725 	8 	0.0% 	
 
44768 	43 	0.2% 	
 
45934 	1 	0.0% 	
 

Premiums Written
Numeric
Distinct count 	4014
Unique (%) 	19.0%
Missing (%) 	0.0%
Missing (n) 	0
Infinite (%) 	0.0%
Infinite (n) 	0
Mean 	292990
Minimum 	-724770
Maximum 	207475612
Zeros (%) 	80.4%
Toggle details

    Statistics
    Histogram
    Common Values
    Extreme Values

Quantile statistics
Minimum 	-724770
5-th percentile 	0
Q1 	0
Median 	0
Q3 	0
95-th percentile 	584420
Maximum 	207475612
Range 	208200382
Interquartile range 	0

Descriptive statistics
Standard deviation 	3425100
Coef of variation 	11.69
Kurtosis 	1549.9
Mean 	292990
MAD 	532440
Skewness 	33.921
Sum 	6198576277
Variance 	11731000000000
Memory size 	165.4 KiB
Value 	Count 	Frequency (%) 	 
0 	17011 	80.4% 	
 
1 	4 	0.0% 	
 
2 	4 	0.0% 	
 
500 	4 	0.0% 	
 
100 	4 	0.0% 	
 
200 	4 	0.0% 	
 
121 	3 	0.0% 	
 
188 	3 	0.0% 	
 
70 	3 	0.0% 	
 
250 	3 	0.0% 	
 
Other values (4004) 	4113 	19.4% 	
 

Minimum 5 values
Value 	Count 	Frequency (%) 	 
-724770 	1 	0.0% 	
 
-486039 	1 	0.0% 	
 
-484413 	1 	0.0% 	
 
-262957 	1 	0.0% 	
 
-256870 	1 	0.0% 	
 

Maximum 5 values
Value 	Count 	Frequency (%) 	 
100812984 	1 	0.0% 	
 
115636287 	1 	0.0% 	
 
167540376 	1 	0.0% 	
 
184408780 	1 	0.0% 	
 
207475612 	1 	0.0% 	
 

State
Categorical
Distinct count 	40
Unique (%) 	0.2%
Missing (%) 	0.0%
Missing (n) 	0
IL 	
 
2267
OH 	
 
2037
IA 	
 
1690
Other values (37) 	
15162
Toggle details
Value 	Count 	Frequency (%) 	 
IL 	2267 	10.7% 	
 
OH 	2037 	9.6% 	
 
IA 	1690 	8.0% 	
 
WI 	1661 	7.9% 	
 
CT 	1647 	7.8% 	
 
PA 	1415 	6.7% 	
 
NY 	1368 	6.5% 	
 
TX 	938 	4.4% 	
 
DE 	914 	4.3% 	
 
IN 	852 	4.0% 	
 
Other values (30) 	6367 	30.1% 	
 

Taxes Paid
Numeric
Distinct count 	3079
Unique (%) 	14.6%
Missing (%) 	0.0%
Missing (n) 	0
Infinite (%) 	0.0%
Infinite (n) 	0
Mean 	4300.4
Minimum 	-120339
Maximum 	2482189
Zeros (%) 	77.7%
Toggle details

    Statistics
    Histogram
    Common Values
    Extreme Values

Quantile statistics
Minimum 	-120339
5-th percentile 	0
Q1 	0
Median 	0
Q3 	0
95-th percentile 	10582
Maximum 	2482189
Range 	2602528
Interquartile range 	0

Descriptive statistics
Standard deviation 	43487
Coef of variation 	10.112
Kurtosis 	1541.7
Mean 	4300.4
MAD 	7765.3
Skewness 	33.242
Sum 	90979931
Variance 	1891100000
Memory size 	165.4 KiB
Value 	Count 	Frequency (%) 	 
0 	16442 	77.7% 	
 
1 	90 	0.4% 	
 
23 	51 	0.2% 	
 
2 	41 	0.2% 	
 
4 	40 	0.2% 	
 
3 	35 	0.2% 	
 
7 	32 	0.2% 	
 
5 	25 	0.1% 	
 
6 	24 	0.1% 	
 
24 	22 	0.1% 	
 
Other values (3069) 	4354 	20.6% 	
 

Minimum 5 values
Value 	Count 	Frequency (%) 	 
-120339 	1 	0.0% 	
 
-78223 	1 	0.0% 	
 
-39539 	1 	0.0% 	
 
-24734 	1 	0.0% 	
 
-18582 	1 	0.0% 	
 

Maximum 5 values
Value 	Count 	Frequency (%) 	 
1334218 	1 	0.0% 	
 
2005889 	1 	0.0% 	
 
2240684 	1 	0.0% 	
 
2326381 	1 	0.0% 	
 
2482189 	1 	0.0% 	
 

Year
Constant

This variable is constant and should be ignored for analysis
Constant value 	2017
Correlations
Sample
	Year 	Iowa Code Chapter 	State 	Company Name 	Line of Insurance 	Premiums Written 	Losses Paid 	Taxes Paid 	NAIC Number 	Iowa Company Code
0 	2017 	515.48 	WI 	1st Auto & Casualty Insurance Company 	Farmowners Multiple Peril 	73953 	3900 	754 	44725 	2894
1 	2017 	515.48 	WI 	1st Auto & Casualty Insurance Company 	Homeowners Multiple Peril 	30778 	7500 	314 	44725 	2894
2 	2017 	515.48 	WI 	1st Auto & Casualty Insurance Company 	Other Liability - Occcurence 	50931 	0 	520 	44725 	2894
3 	2017 	515.48 	WI 	1st Auto & Casualty Insurance Company 	Other Private Passenger Auto Liability 	1434294 	826321 	14632 	44725 	2894
4 	2017 	515.48 	WI 	1st Auto & Casualty Insurance Company 	Other Commerical Auto Liability 	72651 	28321


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

df_orig = pd.read_csv("C:/Users/Arun/Desktop/IOWA.csv")

df_orig.info()
df_orig.head()

df = df_orig.copy(deep = True)


pd.isnull(df).any()
pd.isnull(df).sum()

Year                 False
Iowa Code Chapter     True
State                False
Company Name         False
Line of Insurance    False
Premiums Written     False
Losses Paid          False
Taxes Paid           False
NAIC Number          False
Iowa Company Code    False
dtype: bool

Year                  0
Iowa Code Chapter    43
State                 0
Company Name          0
Line of Insurance     0
Premiums Written      0
Losses Paid           0
Taxes Paid            0
NAIC Number           0
Iowa Company Code     0
dtype: int64

df["Iowa Code Chapter"].fillna("Replaced_null", inplace = True)

df['Year'].nunique()
df['Year'].unique()
df['Year'].value_counts()

1

array([2017], dtype=int64)

2017    21156
Name: Year, dtype: int64

#year is not reqd, as it is single value
cols_to_drop = ['Year']
df.drop(cols_to_drop, axis=1, inplace=True)

#Iowa Company Code and NAIC Number are same for each company name, hence can be deleted
cols_to_drop = ['Iowa Company Code', 'NAIC Number']
df.drop(cols_to_drop, axis=1, inplace=True)

df['Iowa Code Chapter'].nunique()
df['Iowa Code Chapter'].unique()
df['Iowa Code Chapter'].value_counts()

4

array(['515.48', '515C', '520', 'Replaced_null'], dtype=object)

515.48           20202
515C               609
520                302
Replaced_null       43
Name: Iowa Code Chapter, dtype: int64

df['State'].nunique()
df['State'].unique()
df['State'].value_counts()

40

array(['WI', 'NY', 'CO', 'PA', 'MD', 'NH', 'MI', 'FL', 'IL', 'IA', 'AZ',
       'IN', 'RI', 'AK', 'NJ', 'OH', 'DE', 'DC', 'TX', 'MO', 'MN', 'CA',
       'NE', 'OK', 'KS', 'NC', 'CT', 'WV', 'OR', 'MA', 'TN', 'SD', 'LA',
       'ND', 'VT', 'GA', 'VA', 'AL', 'NV', 'AR'], dtype=object)
       
       
       Year                 False
Iowa Code Chapter     True
State                False
Company Name         False
Line of Insurance    False
Premiums Written     False
Losses Paid          False
Taxes Paid           False
NAIC Number          False
Iowa Company Code    False
dtype: bool

Year                  0
Iowa Code Chapter    43
State                 0
Company Name          0
Line of Insurance     0
Premiums Written      0
Losses Paid           0
Taxes Paid            0
NAIC Number           0
Iowa Company Code     0
dtype: int64


df["Iowa Code Chapter"].fillna("Replaced_null", inplace = True)

df['Year'].nunique()
df['Year'].unique()
df['Year'].value_counts()

1

array([2017], dtype=int64)

2017    21156
Name: Year, dtype: int64

#year is not reqd, as it is single value
cols_to_drop = ['Year']
df.drop(cols_to_drop, axis=1, inplace=True)

#Iowa Company Code and NAIC Number are same for each company name, hence can be deleted
cols_to_drop = ['Iowa Company Code', 'NAIC Number']
df.drop(cols_to_drop, axis=1, inplace=True)
df['Iowa Code Chapter'].nunique()
df['Iowa Code Chapter'].unique()
df['Iowa Code Chapter'].value_counts()

4

array(['515.48', '515C', '520', 'Replaced_null'], dtype=object)

515.48           20202
515C               609
520                302
Replaced_null       43
Name: Iowa Code Chapter, dtype: int64

df['State'].nunique()
df['State'].unique()
df['State'].value_counts()

40

array(['WI', 'NY', 'CO', 'PA', 'MD', 'NH', 'MI', 'FL', 'IL', 'IA', 'AZ',
       'IN', 'RI', 'AK', 'NJ', 'OH', 'DE', 'DC', 'TX', 'MO', 'MN', 'CA',
       'NE', 'OK', 'KS', 'NC', 'CT', 'WV', 'OR', 'MA', 'TN', 'SD', 'LA',
       'ND', 'VT', 'GA', 'VA', 'AL', 'NV', 'AR'], dtype=object)

IL    2267
OH    2037
IA    1690
WI    1661
CT    1647
PA    1415
NY    1368
TX     938
DE     914
IN     852
NH     822
MN     641
CA     616
MO     575
MI     462
NE     454
NC     395
FL     386
RI     276
NJ     216
KS     169
AZ     149
OK     131
VT     129
MD     120
MA      99
LA      90
WV      87
TN      87
GA      86
SD      52
CO      52
ND      51
OR      47
NV      43
AR      43
AL      43
VA      43
AK       2
DC       1
Name: State, dtype: int64

df['Company Name'].nunique()
df['Company Name'].unique()
df['Company Name'].value_counts()

856

array(['1st Auto & Casualty Insurance Company',
       '21st Century North America Insurance Company',
       '21st Century Pacific Insurance Company',
       '21st Century Premier Insurance Company',
       '21st Century Security Insurance Company',
       'ACA Financial Guaranty Corporation', 'Acadia Insurance Company',
       'Accident Fund General Insurance Company',
       'Accident Fund Insurance Company of America',
       'Accident Fund National Insurance Company',
       'Accredited Surety and Casualty Company, Inc.',
       'ACE American Insurance Company',
       'ACE Fire Underwriters Insurance Company',
       'ACE Property and Casualty Insurance Company',
       'ACIG Insurance Company', 'ACUITY, A Mutual Insurance Company',
       'Addison Insurance Company', 'ADM Insurance Company',
       'Advantage Workers Compensation Insurance Company',
       'Aegis Security Insurance Company',
       'Affiliated FM Insurance Company',
       'Affirmative Direct Insurance Company',
       'AGCS Marine Insurance Company', 'Agri General Insurance Company',
       'AIG Assurance Company', 'AIG Property Casualty Company',
       'AIU Insurance Company', 'Alamance Insurance Company',
       'Alaska National Insurance Company',
       'Alea North America Insurance Company',
       'Allegheny Casualty Company',
       'Allianz Global Risks US Insurance Company',
       'Allied Eastern Indemnity Company',
       'Allied Insurance Company of America',
       'Allied Property and Casualty Insurance Company',
       'Allied World Specialty Insurance Company',
       'Allied World Insurance Company',
       'Allied World National Assurance Company',
       'Allmerica Financial Alliance Insurance Company',
       'Allmerica Financial Benefit Insurance Company',
       'Allstate Fire and Casualty Insurance Company',
       'Allstate Indemnity Company', 'Allstate Insurance Company',
       'Allstate Northbrook Indemnity Company',
       'Allstate Property and Casualty Insurance Company',
       'Allstate Vehicle and Property Insurance Company',
       'Alpha Property & Casualty Insurance Company',
       'Alterra America Insurance Company',
       'Amalgamated Casualty Insurance Company',
       'AMBAC Assurance Corporation', 'AMCO Insurance Company',
       'American Access Casualty Company',
       'American Agri-Business Insurance Company',
       'American Alternative Insurance Corporation',
       'American Automobile Insurance Company',
       'American Bankers Insurance Company of Florida',
       'American Business & Mercantile Insurance Mutual, Inc.',
       'American Casualty Company of Reading Pennsylvania',
       'American Commerce Insurance Company',
       'American Fire and Casualty Company',
       'American Compensation Insurance Company',
       'American Contractors Indemnity Company',
       'American Country Insurance Company',
       'American Economy Insurance Company',
       'American Family Home Insurance Company',
       'American Family Insurance Company',
       'American Family Mutual Insurance Company, S.I.',
       'American Guarantee & Liability Insurance Company',
       'American Insurance Company (The)',
       'American Hallmark Insurance Company of Texas',
       'American Home Assurance Company',
       'American Interstate Insurance Company',
       'American Mercury Insurance Company',
       'American Mining Insurance Company', 'AmGUARD Insurance Company',
       'American Modern Home Insurance Company',
       'American Modern Select Insurance Company',
       'American National General Insurance Company',
       'American National Property and Casualty Company',
       'American Pet Insurance Company',
       'American Physicians Assurance Corporation',
       'American Reliable Insurance Company',
       'American Road Insurance Company (The)',
       'American Security Insurance Company',
       'American Select Insurance Company',
       'American Sentinel Insurance Company',
       'American Service Insurance Company, Inc.',
       'American Southern Home Insurance Company',
       'American Standard Insurance Company of Wisconsin',
       'American States Insurance Company',
       'American Strategic Insurance Corp.',
       'American Summit Insurance Company', 'American Surety Company',
       'American Zurich Insurance Company',
       'Ameriprise Insurance Company', 'Amerisure Insurance Company',
       'Amerisure Mutual Insurance Company',
       'Amerisure Partners Insurance Company', 'AMEX Assurance Company',
       'Amica Mutual Insurance Company',
       'AmTrust Insurance Company of Kansas, Inc.',
       'Ansur America Insurance Company',
       'Applied Underwriters Captive Risk Assurance Company, Inc.',
       'ARAG Insurance Company', 'Arch Indemnity Insurance Company',
       'Arch Insurance Company', 'Arch Mortgage Assurance Company',
       'Arch Mortgage Guaranty Company',
       'Arch Mortgage Insurance Company',
       'Arch Structured Mortgage Insurance Company',
       'Argonaut Great Central Insurance Company',
       'Argonaut Insurance Company', 'Argonaut-Midwest Insurance Company',
       'Armed Forces Insurance Exchange Armed Forces Ins. Corp. A/I/F',
       'Arrowood Indemnity Company',
       'Artisan and Truckers Casualty Company',
       'Ashmere Insurance Company', 'Aspen American Insurance Company',
       'Associated Indemnity Corporation',
       'Association Casualty Insurance Company', 'Assured Guaranty Corp.',
       'Assured Guaranty Municipal Corp.', 'Atain Insurance Company',
       'Atlanta International Insurance Company',
       'AXIS Reinsurance Company', 'Atlantic Specialty Insurance Company',
       'Atlantic States Insurance Company',
       'Atradius Trade Credit Insurance, Inc.',
       'Austin Mutual Insurance Company',
       'Auto Club Group Insurance Company',
       'Auto Club Property-Casualty Insurance Company',
       'Automobile Insurance Company of Hartford, Connecticut (The)',
       'Auto-Owners Insurance Company', 'AVEMCO Insurance Company',
       'AXA Art Insurance Corporation', 'AXA Insurance Company',
       'AXIS Insurance Company', 'Badger Mutual Insurance Company',
       'Balboa Insurance Company', 'Bankers Insurance Company',
       'Bankers Standard Insurance Company',
       'Bar Plan Mutual Insurance Company (The)', 'BCS Insurance Company',
       'Bearing Midwest Casualty Company',
       'Beazley Insurance Company, Inc.', 'Bedivere Insurance Company',
       'Benchmark Insurance Company', 'Berkley Assurance Company',
       'Berkley Insurance Company', 'Berkley National Insurance Company',
       'Berkley Regional Insurance Company',
       'Berkshire Hathaway Assurance Corporation',
       'Berkshire Hathaway Direct Insurance Company',
       'Berkshire Hathaway Homestate Insurance Company',
       'Berkshire Hathaway Specialty Insurance Company',
       'BITCO National Insurance Company',
       'BITCO General Insurance Corporation',
       'Blackboard Insurance Company',
       'BrickStreet Mutual Insurance Company',
       'Bristol West Insurance Company', 'Brookwood Insurance Company',
       'Brotherhood Mutual Insurance Company',
       'Buckeye State Mutual Insurance Company (The)',
       'Build America Mutual Assurance Company',
       'California Casualty General Insurance Company of Oregon',
       'California Casualty Indemnity Exchange, CA Cas Mgmt Co Atty-in-Fact',
       'California Insurance Company', 'Cameron Mutual Insurance Company',
       'Cameron National Insurance Company',
       'CAMICO Mutual Insurance Company', 'Capitol Indemnity Corporation',
       'Capson Physicians Insurance Company',
       'Carolina Casualty Insurance Company',
       'Caterpillar Insurance Company', 'Catlin Indemnity Company',
       'Catlin Insurance Company, Inc.',
       'Celina Mutual Insurance Company', 'Censtat Casualty Company',
       'Central Mutual Insurance Company',
       'Central States Indemnity Co. of Omaha',
       'Centurion Casualty Company', 'Century Indemnity Company',
       'Century-National Insurance Company', 'CGB Insurance Company',
       'Charter Oak Fire Insurance Company (The)',
       'Cherokee Insurance Company', 'Chicago Insurance Company',
       'Chubb Indemnity Insurance Company',
       'Chubb National Insurance Company',
       'Church Insurance Company (The)',
       'Church Mutual Insurance Company',
       'Cincinnati Casualty Company (The)',
       'Cincinnati Indemnity Company (The)',
       'Cincinnati Insurance Company (The)',
       'Citizens Insurance Company of America',
       'Clarendon National Insurance Company',
       'Clear Blue Insurance Company',
       'Clear Spring Property and Casualty Company',
       'Clermont Insurance Company',
       'Coface North America Insurance Company',
       'Columbia National Insurance Company',
       'Colonial American Casualty and Surety Company',
       'Colonial Surety Company', 'Columbia Insurance Company',
       'Columbia Mutual Insurance Company',
       'Commerce and Industry Insurance Company',
       'Commercial Casualty Insurance Company',
       'Continental Casualty Company', 'Compass Insurance Company',
       'Consolidated Insurance Company', 'Consumers Insurance USA, Inc.',
       'Continental Heritage Insurance Company',
       'Continental Indemnity Company',
       'Continental Insurance Company (The)',
       'Continental Western Insurance Company',
       'Contractors Bonding and Insurance Company',
       'COPIC Insurance Company', 'CorePointe Insurance Company',
       'Country Casualty Insurance Company',
       'Crestbrook Insurance Company', 'Country Mutual Insurance Company',
       'Country Preferred Insurance Company',
       'Courtesy Insurance Company', 'Crum & Forster Indemnity Company',
       'CUMIS Insurance Society, Inc.', 'Dairyland Insurance Company',
       'Dakota Truck Underwriters, Risk Administration Services, Inc. A/I/F',
       'Dealers Assurance Company', 'Depositors Insurance Company',
       'Developers Surety and Indemnity Company',
       'Diamond Insurance Company', 'Diamond State Insurance Company',
       'Discover Property & Casualty Insurance Company',
       "Doctors' Company (The), An Interinsurance Exchange, The Doctor's Management",
       'Doctors Direct Insurance, Inc.',
       'Donegal Mutual Insurance Company', 'Dorinco Reinsurance Company',
       'Eastern Advantage Assurance Company',
       'Eastern Alliance Insurance Company',
       'EastGUARD Insurance Company', 'Economy Fire & Casualty Company',
       'Economy Preferred Insurance Company',
       'Economy Premier Assurance Company', 'Electric Insurance Company',
       'EMC Property & Casualty Company', 'EMCASCO Insurance Company',
       'Empire Fire and Marine Insurance Company',
       'Employers Assurance Company',
       'Employers Compensation Insurance Company',
       'Employers Insurance Company of Wausau',
       'Employers Mutual Casualty Company',
       'Employers Preferred Insurance Company',
       'Encompass Indemnity Company', 'Encompass Insurance Company',
       'Encompass Insurance Company of America',
       'Endurance American Insurance Company',
       'Endurance Assurance Corporation', 'Equity Insurance Company',
       'Essent Guaranty, Inc.', 'Essentia Insurance Company',
       'Esurance Insurance Company',
       'Esurance Insurance Company of New Jersey',
       'Esurance Property and Casualty Insurance Company',
       'Euler Hermes North America Insurance Company',
       'Everest Denali Insurance Company',
       'Everest National Insurance Company',
       'Everest Premier Insurance Company', 'Everest Reinsurance Company',
       'Evergreen National Indemnity Company',
       'Everspan Financial Guarantee Corp.',
       'Excess Share Insurance Corporation',
       'Executive Risk Indemnity Inc.', 'Explorer Insurance Company',
       'Factory Mutual Insurance Company',
       'Fair American Insurance and Reinsurance Company',
       'Falls Lake National Insurance Company',
       'Farm Bureau Property & Casualty Insurance Company',
       'Farmers Auto. Ins. Assn. Farmers Automobile Management Corp. A/I/F',
       'Farmers Insurance Company, Inc.',
       'Farmers Insurance Exchange Farmers Underwriters Assn. A/I/F',
       'Farmers Mutual Hail Insurance Company of Iowa',
       'Farmington Casualty Company', 'Farmland Mutual Insurance Company',
       'FCCI Insurance Company', 'Federal Insurance Company',
       'Federated Mutual Insurance Company',
       'Federated Reserve Insurance Company',
       'Federated Rural Electric Insurance Exchange, Fed Rural Elec Mgmt Corp A/I/F',
       'Federated Service Insurance Company',
       'Fidelity and Deposit Company of Maryland',
       'Fidelity and Guaranty Insurance Company',
       'Fidelity and Guaranty Insurance Underwriters, Inc.',
       'Financial Casualty & Surety, Inc.',
       'Financial Pacific Insurance Company',
       'Fire Insurance Exchange Fire Underwriters Assn. A/I/F',
       "Fireman's Fund Insurance Company",
       "Firemen's Insurance Company of Washington, D.C.",
       'First American Property & Casualty Insurance Company',
       'First Chicago Insurance Company',
       'First Colonial Insurance Company',
       'First Dakota Indemnity Company',
       'First Financial Insurance Company',
       'First Guard Insurance Company',
       'First Liberty Insurance Corporation (The)',
       'First National Insurance Company of America',
       'FirstComp Insurance Company', 'Florists Mutual Insurance Company',
       'FMH Ag Risk Insurance Company', 'Foremost Insurance Company',
       'Foremost Property and Casualty Insurance Company',
       'Foremost Signature Insurance Company',
       'Fortress Insurance Company', 'Fortuity Insurance Company',
       'Founders Insurance Company',
       'Frank Winston Crum Insurance Company',
       'Frankenmuth Mutual Insurance Company',
       'Freedom Specialty Insurance Company', 'Fremont Insurance Company',
       'Garrison Property and Casualty Insurance Company',
       'Gateway Insurance Company', 'GEICO Advantage Insurance Company',
       'GEICO Casualty Company', 'GEICO General Insurance Company',
       'GEICO Indemnity Company', 'GEICO Marine Insurance Company',
       'GEICO Secure Insurance Company',
       'General Casualty Company of Wisconsin',
       'General Casualty Insurance Company',
       'General Insurance Company of America',
       'General Reinsurance Corporation',
       'General Security National Insurance Company',
       'General Star National Insurance Company',
       'Generali (United States Branch)', 'Genesis Insurance Company',
       'Genworth Financial Assurance Corporation',
       'Genworth Mortgage Insurance Corporation',
       'Genworth Mortgage Insurance Corporation of North Carolina',
       'GeoVera Insurance Company',
       'Government Employees Insurance Company', 'Granite Re, Inc.',
       'Granite State Insurance Company',
       'Graphic Arts Mutual Insurance Company',
       'Gray Insurance Company (The)',
       'Great American Alliance Insurance Company',
       'Great American Assurance Company',
       'Great American Insurance Company',
       'Great American Insurance Company of New York',
       'Great American Security Insurance Company',
       'Great American Spirit Insurance Company',
       'Great Divide Insurance Company',
       'Great Midwest Insurance Company',
       'Great Northern Insurance Company', 'Great Plains Casualty, Inc.',
       'Great West Casualty Company',
       'Greater New York Mutual Insurance Company',
       'Greenwich Insurance Company', 'Greyhawk Insurance Company',
       'Grinnell Mutual Reinsurance Company',
       'Grinnell Select Insurance Company',
       'Guarantee Company of North America USA (The)',
       'GuideOne America Insurance Company',
       'GuideOne Elite Insurance Company',
       'GuideOne Mutual Insurance Company',
       'GuideOne Specialty Mutual Insurance Company',
       'Hallmark Insurance Company',
       'Hallmark National Insurance Company',
       'Hanover American Insurance Company (The)',
       'Hanover Insurance Company (The)',
       'Harco National Insurance Company',
       'Harleysville Insurance Company',
       'Harleysville Preferred Insurance Company',
       'Harleysville Worcester Insurance Company',
       'Hartford Accident & Indemnity Company',
       'Hartford Casualty Insurance Company',
       'Hartford Fire Insurance Company',
       'Hartford Insurance Company of the Midwest',
       'Hartford Steam Boiler Inspection & Insurance Company (The)',
       'Hartford Steam Boiler Inspection and Insurance Company of Connecticut (The)',
       'Hartford Underwriters Insurance Company',
       'Hastings Mutual Insurance Company',
       'Hawkeye-Security Insurance Company',
       'HDI Global Insurance Company',
       'Heritage Casualty Insurance Company',
       'Heritage Indemnity Company', 'Hiscox Insurance Company Inc.',
       'Home-Owners Insurance Company',
       'Homeowners of America Insurance Company',
       'Homesite Insurance Company',
       'Homesite Insurance Company of the Midwest',
       'Horace Mann Insurance Company',
       'Horace Mann Property & Casualty Insurance Company',
       'Housing Authority Property Insurance, A Mutual Company',
       'IDS Property Casualty Insurance Company',
       'Housing Enterprise Insurance Company, Inc.',
       'Hudson Insurance Company', 'Illinois Casualty Company',
       'Illinois EMCASCO Insurance Company', 'Illinois Insurance Company',
       'Illinois National Insurance Company',
       'Imperium Insurance Company', 'IMT Insurance Company',
       'Indemnity Insurance Company of North America',
       'Independence American Insurance Company',
       'Indiana Insurance Company',
       'Indiana Lumbermens Mutual Insurance Company',
       'Infinity Insurance Company', 'Inland Insurance Company',
       'Integrity Select Insurance Company',
       'Insurance Company of Illinois',
       'Insurance Company of North America',
       'Insurance Company of the State of Pennsylvania (The)',
       'Insurance Company of the West', 'Integon Indemnity Corporation',
       'Integrity Mutual Insurance Company',
       'Integon National Insurance Company',
       'Integrity Property & Casualty Insurance Company',
       'Intrepid Insurance Company',
       'International Fidelity Insurance Company',
       'Iowa American Insurance Company', 'Iowa Mutual Insurance Company',
       'Ironshore Indemnity, Inc.', 'ISMIE Mutual Insurance Company',
       'Jefferson Insurance Company', 'Jewelers Mutual Insurance Company',
       'JM Specialty Insurance Comipany', 'Key Risk Insurance Company',
       'KnightBrook Insurance Company', 'Lafayette Insurance Company',
       'Lamorak Insurance Company', 'Lancer Insurance Company',
       'Le Mars Insurance Company',
       'Lexington National Insurance Corporation',
       'Lexon Insurance Company', 'Liberty Insurance Corporation',
       'Liberty Insurance Underwriters Inc.',
       'Liberty Mutual Fire Insurance Company',
       'Liberty Mutual Insurance Company',
       'Liberty Personal Insurance Company',
       'LM General Insurance Company', 'LM Insurance Corporation',
       'LM Property and Casualty Insurance Company',
       'Lyndon Southern Insurance Company',
       'MAG Mutual Insurance Company',
       'Manufacturers Alliance Insurance Company',
       'Mapfre Insurance Company', 'Markel American Insurance Company',
       'Markel Global Reinsurance Company', 'Markel Insurance Company',
       'Massachusetts Bay Insurance Company',
       'Maxum Casualty Insurance Company', 'MBIA Insurance Corporation',
       'Medical Protective Company (The)', 'Medicus Insurance Company',
       'MEDMARC Casualty Insurance Company',
       'MemberSelect Insurance Company', 'MEMIC Indemnity Company',
       'Mendakota Insurance Company', 'Mendota Insurance Company',
       'Merastar Insurance Company', 'Merchants Bonding Company (Mutual)',
       'Merchants National Bonding, Inc.',
       'Meridian Security Insurance Company',
       'Meritplan Insurance Company', 'Metromile Insurance Company',
       'Metropolitan Casualty Insurance Company',
       'Metropolitan Direct Property and Casualty Insurance Company',
       'Metropolitan General Insurance Company',
       'Metropolitan Group Property and Casualty Insurance Company',
       'Metropolitan Property and Casualty Insurance Company',
       'MFS Mutual Insurance Company', 'MGA Insurance Company, Inc.',
       'MGIC Assurance Corporation', 'MGIC Credit Assurance Corporation',
       'MGIC Indemnity Corporation', 'Miami Mutual Insurance Company',
       'MIC General Insurance Corporation',
       'MIC Property and Casualty Insurance Corporation',
       'Michigan Millers Mutual Insurance Company',
       'Mid-American Fire & Casualty Co.',
       'Mid-Century Insurance Company', 'Mid-Continent Casualty Company',
       'Middlesex Insurance Company', 'Midvale Indemnity Company',
       "Midwest Builders' Casualty Mutual Company",
       'Midwest Employers Casualty Company',
       'Midwest Family Mutual Insurance Company',
       'Midwest Insurance Company', 'Midwestern Indemnity Company (The)',
       'Milbank Insurance Company', 'Milford Casualty Insurance Company',
       'Minnesota Lawyers Mutual Insurance Company',
       'Mitsui Sumitomo Insurance Company of America',
       'Mitsui Sumitomo Insurance USA Inc.', 'MMIC Insurance, Inc.',
       'Mortgage Guaranty Insurance Corporation',
       'Motorists Commercial Mutual Insurance Company',
       'Motors Insurance Corporation', 'Munich Reinsurance America, Inc.',
       'Municipal Assurance Corp.',
       'MutualAid eXchange, MII Management Group, Inc. Atty-in-fact',
       'National American Insurance Company', 'National Casualty Company',
       'National Continental Insurance Company',
       'National General Insurance Company',
       'National Farmers Union Property and Casualty Company',
       'National Fire and Indemnity Exch. John L. Corley Inc. A/I/F',
       'National Fire Insurance Company of Hartford',
       'National General Assurance Company',
       'National General Insurance Online, Inc.',
       'National Indemnity Company',
       'National Interstate Insurance Company',
       'National Indemnity Company of Mid-America',
       'National Indemnity Company of the South',
       'National Insurance Association, The National Corporation A/I/F',
       'National Liability & Fire Insurance Company',
       'National Mortgage Insurance Corporation',
       'National Mutual Insurance Company (The)',
       'National Public Finance Guarantee Corporation',
       'National Surety Corporation',
       'National Specialty Insurance Company',
       'National Trust Insurance Company',
       'National Union Fire Insurance Company of Pittsburgh Pa.',
       'Nationwide Assurance Company',
       'Nationwide Affinity Insurance Company of America',
       'Nationwide Agribusiness Insurance Company',
       'Nationwide General Insurance Company',
       'Nationwide Insurance Company of America',
       'Nationwide Mutual Fire Insurance Company',
       'Nationwide Mutual Insurance Company',
       'Nationwide Property and Casualty Insurance Company',
       'NAU Country Insurance Company', 'Navigators Insurance Company',
       'NCMIC Insurance Company', 'Netherlands Insurance Company (The)',
       'New England Insurance Company',
       'New England Reinsurance Corporation',
       'New Hampshire Insurance Company', 'NGM Insurance Company',
       'New York Marine and General Insurance Company',
       'NORCAL Mutual Insurance Company', 'NorGUARD Insurance Company',
       'North American Elite Insurance Company',
       'North American Specialty Insurance Company',
       'North Pointe Insurance Company',
       'North River Insurance Company (The)',
       'North Star Mutual Insurance Company',
       'Northfield Insurance Company', 'Northland Insurance Company',
       'NorthStone Insurance Company', 'NOVA Casualty Company',
       'Nutmeg Insurance Company', 'Oak River Insurance Company',
       'Oakwood Insurance Company', 'OBI America Insurance Company',
       'OBI National Insurance Company',
       'Occidental Fire & Casualty Company of North Carolina',
       'Odyssey Reinsurance Company', 'OHIC Insurance Company',
       'Ohio Casualty Insurance Company (The)',
       'Ohio Farmers Insurance Company', 'Ohio Indemnity Company',
       'Ohio Mutual Insurance Company', 'Ohio Security Insurance Company',
       'Old United Casualty Company', 'Old Guard Insurance Company',
       'Old Republic General Insurance Corporation',
       'Old Republic Insurance Company',
       'Old Republic Security Assurance Company',
       'Old Republic Surety Company', 'Omni Indemnity Company',
       'Omni Insurance Company', 'OneCIS Insurance Company',
       'Owners Insurance Company', 'Pacific Employers Insurance Company',
       'Pacific Indemnity Company', 'Pacific Specialty Insurance Company',
       'Pacific Star Insurance Company', 'PACO Assurance Company, Inc.',
       'Partner Reinsurance Company of the U.S.',
       'Patriot General Insurance Company',
       'PartnerRe America Insurance Company',
       'PartnerRe Insurance Company of New York',
       'Partners Mutual Insurance Company',
       'Peachtree Casualty Insurance Company',
       'Peak Property and Casualty Insurance Corporation',
       'Peerless Indemnity Insurance Company',
       'Peerless Insurance Company', 'Pekin Insurance Company',
       'Penn Millers Insurance Company', 'Penn-America Insurance Company',
       'Pennsylvania Insurance Company',
       'Pennsylvania Lumbermens Mutual Insurance Company',
       "Pennsylvania Manufacturers' Association Insurance Company",
       'Permanent General Assurance Corporation',
       'Pennsylvania Manufacturers Indemnity Company',
       'Pennsylvania National Mutual Casualty Insurance Company',
       'Permanent General Assurance Corporation of Ohio',
       'Petroleum Casualty Company',
       'Petroleum Marketers Management Insurance Company',
       'Pharmacists Mutual Insurance Company',
       'Philadelphia Indemnity Insurance Company',
       'Phoenix Insurance Company (The)',
       'PinnaclePoint Insurance Company',
       'Pioneer Specialty Insurance Company',
       'Platte River Insurance Company',
       "Plans' Liability Insurance Company", 'Plaza Insurance Company',
       'Plateau Casualty Insurance Company', 'PMI Insurance Co.',
       'Podiatry Insurance Company of America',
       'Praetorian Insurance Company',
       'Preferred Professional Insurance Company',
       'Previsor Insurance Company',
       'Privilege Underwriters Reciprocal Exchange, PURE Risk Mgt, Atty-in-Fact',
       'ProAssurance Casualty Company',
       'ProAssurance Indemnity Company, Inc.',
       'ProCentury Insurance Company',
       'Producers Agriculture Insurance Company',
       'Professional Solutions Insurance Company',
       'Professionals Advocate Insurance Company',
       'Progressive Advanced Insurance Company',
       'Progressive Casualty Insurance Company',
       'Progressive Classic Insurance Company',
       'Progressive Commercial Casualty Company',
       'Progressive Direct Insurance Company',
       'Progressive Max Insurance Company',
       'Progressive Northern Insurance Company',
       'Progressive Northwestern Insurance Company',
       'Progressive Preferred Insurance Company',
       'Progressive Specialty Insurance Company',
       'Progressive Universal Insurance Company',
       'Property and Casualty Insurance Company of Hartford',
       'Property-Owners Insurance Company', 'ProSelect Insurance Company',
       'Protective Insurance Company',
       'Protective Property & Casualty Insurance Company',
       'Providence Washington Insurance Company',
       'QBE Insurance Corporation', 'QBE Reinsurance Corporation',
       'R.V.I. America Insurance Company', 'Radian Guaranty Inc.',
       'Radian Mortgage Assurance Inc.', 'Radian Mortgage Guaranty Inc.',
       'Radnor Specialty Insurance Company', 'Rampart Insurance Company',
       'Redwood Fire and Casualty Insurance Company',
       'Regent Insurance Company', 'Renaissance Reinsurance U.S. Inc.',
       'Republic Indemnity Company of America',
       'Republic Indemnity Company of California',
       'Republic Mortgage Assurance Company',
       'Republic Mortgage Guaranty Insurance Corporation',
       'Repwest Insurance Company', 'Republic Mortgage Insurance Company',
       'Response Insurance Company',
       'Response Worldwide Direct Auto Insurance Company',
       'Response Worldwide Insurance Company',
       'Riverport Insurance Company', 'RLI Insurance Company',
       'Roche Surety and Casualty Company, Inc.',
       'Rockford Mutual Insurance Company',
       'Rockwood Casualty Insurance Company', 'RSUI Indemnity Company',
       'Rural Community Insurance Company',
       'Rural Trust Insurance Company',
       'SAFECO Insurance Company of America',
       'SAFECO Insurance Company of Illinois',
       'Sagamore Insurance Company',
       'SAFECO Insurance Company of Indiana', 'Safeway Insurance Company',
       'Safeco National Insurance Company',
       'Safety First Insurance Company',
       'Safety National Casualty Corporation',
       'Samsung Fire & Marine Insurance Co., Ltd. (U.S. Branch)',
       'San Francisco Reinsurance Company', 'Scor Reinsurance Company',
       'Scottsdale Indemnity Company',
       'Secura Insurance A Mutual Company',
       'SECURA Supreme Insurance Company', 'Securian Casualty Company',
       'Security National Insurance Company', 'Select Insurance Company',
       'Selective Insurance Company of America',
       'Selective Insurance Company of South Carolina',
       'Selective Insurance Company of the Southeast',
       'Seneca Insurance Company, Inc.',
       'Sentinel Insurance Company, Ltd.', 'Sentruity Casualty Company',
       'Sentry Casualty Company', 'Sentry Insurance, a Mutual Company',
       'Sequoia Indemnity Company', 'Sentry Select Insurance Company',
       'Sequoia Insurance Company', 'Service American Indemnity Company',
       'Service Insurance Company', 'SFM Mutual Insurance Company',
       'SFM Safe Insurance Company', 'SFM Select Insurance Company',
       'Shelter General Insurance Company',
       'Shelter Mutual Insurance Company',
       'Sirius America Insurance Company', 'Shelter Reinsurance Company',
       'Society Insurance, a mutual company',
       'Sompo America Fire & Marine Insurance Company',
       'Sompo America Insurance Company',
       'Southern General Insurance Company', 'Southern Insurance Company',
       'Southwest Marine and General Insurance Company',
       'SPARTA Insurance Company', 'Specialty Risk of America',
       'Spinnaker Insurance Company',
       'St. Paul Fire and Marine Insurance Company',
       'St. Paul Mercury Insurance Company',
       'St. Paul Guardian Insurance Company',
       'St. Paul Protective Insurance Company',
       'Standard Fire Insurance Company (The)',
       'Standard Guaranty Insurance Company',
       'Star Casualty Insurance Company', 'Star Insurance Company',
       'StarNet Insurance Company', 'Starr Indemnity & Liability Company',
       'StarStone National Insurance Company',
       'Starr Specialty Insurance Company',
       'State Auto Property & Casualty Insurance Company',
       'State Automobile Mutual Insurance Company',
       'State Farm Fire and Casualty Company',
       'Stonington Insurance Company',
       'State Farm General Insurance Company',
       'State Farm Mutual Automobile Insurance Company',
       'State National Insurance Company, Inc.',
       'Stillwater Insurance Company',
       'Stillwater Property and Casualty Insurance Company',
       'Stonetrust Commercial Insurance Company',
       'Stratford Insurance Company', 'SU Insurance Company',
       'SummitPoint Insurance Company', 'Sun Surety Insurance Company',
       'SureTec Insurance Company',
       'Swiss Reinsurance America Corporation', 'Syncora Guarantee Inc.',
       'T.H.E. Insurance Company', 'TDC National Assurance Company',
       'Teachers Insurance Company', 'Technology Insurance Company, Inc.',
       'TIG Insurance Company', 'Titan Indemnity Company',
       'TNUS Insurance Company',
       'Toa Reinsurance Company of America (The)',
       'Tokio Marine America Insurance Company', 'Topa Insurance Company',
       'Toyota Motor Insurance Company',
       'Trans Pacific Insurance Company', 'Traders Insurance Company',
       'Transamerica Casualty Insurance Company',
       'Transatlantic Reinsurance Company',
       'TransGuard Insurance Company of America, Inc.',
       'Trumbull Insurance Company', 'TravCo Insurance Company',
       'Transportation Insurance Company',
       'Travelers Casualty and Surety Company',
       'Travelers Casualty and Surety Company of America',
       'Travelers Casualty Company (The)',
       'Travelers Casualty Company of Connecticut',
       'Travelers Casualty Insurance Company of America',
       'Travelers Commercial Casualty Company',
       'Travelers Commercial Insurance Company',
       'Travelers Constitution State Insurance Company',
       'Travelers Home and Marine Insurance Company (The)',
       'Travelers Indemnity Company (The)',
       'Travelers Indemnity Company of America (The)',
       'Travelers Indemnity Company of Connecticut (The)',
       'Trustgard Insurance Company',
       'Travelers Personal Insurance Company',
       'Travelers Personal Security Insurance Company',
       'Travelers Property Casualty Company of America',
       'Travelers Property Casualty Insurance Company',
       'Trenwick America Reinsurance Corporation',
       'Triangle Insurance Company, Inc.',
       'Trinity Universal Insurance Company',
       'Tri-State Insurance Company of Minnesota',
       'Triton Insurance Company', 'Triumphe Casualty Company',
       'Truck Insurance Exchange Truck Undrwrtrs. Assn. A/I/F',
       'Twin City Fire Insurance Company',
       'U.S. Specialty Insurance Company',
       'U.S. Underwriters Insurance Company',
       'UFG Specialty Insurance Company', 'Union Insurance Company',
       'Union Insurance Company of Providence',
       'United Casualty and Surety Insurance Company',
       'United Financial Casualty Company',
       'United Fire & Casualty Company',
       'United Guaranty Credit Insurance Company',
       'United Guaranty Mortgage Indemnity Company',
       'United Guaranty Residential Insurance Company',
       'United Ohio Insurance Company',
       'United Guaranty Residential Insurance Company of North Carolina',
       'United Services Automobile Association, USAA Reciprocal, Attorney-in-fact',
       'United States Fire Insurance Company',
       'United States Fidelity and Guaranty Company',
       'United States Liability Insurance Company',
       'United Wisconsin Insurance Company',
       'Unitrin Safeguard Insurance Company',
       'Unitrin Auto and Home Insurance Company',
       'Unitrin Direct Insurance Company',
       'Unitrin Preferred Insurance Company',
       'Universal Fire & Casualty Insurance Company',
       'Universal Surety Company', 'Universal Surety of America',
       'Universal Underwriters Insurance Company',
       'Universal Underwriters of Texas Insurance Company',
       'Valley Forge Insurance Company',
       'USAA Casualty Insurance Company',
       'USAA General Indemnity Company',
       'USPlate Glass Insurance Company',
       'Utica Mutual Insurance Company', 'Vanliner Insurance Company',
       'Vantapro Specialty Insurance Company',
       'Verlan Fire Insurance Company',
       'Victoria Automobile Insurance Company',
       'Victoria Fire & Casualty Company', 'Vigilant Insurance Company',
       'Viking Insurance Company of Wisconsin',
       'Virginia Surety Company, Inc.', 'Wadena Insurance Company',
       'Warner Insurance Company', 'Western General Insurance Company',
       'Washington International Insurance Company',
       'Watford Insurance Company', 'Wausau Business Insurance Company',
       'Wausau Underwriters Insurance Company', 'Wesco Insurance Company',
       'West American Insurance Company',
       'West Bend Mutual Insurance Company',
       'Westchester Fire Insurance Company',
       'Western Agricultural Insurance Company',
       'Western National Assurance Company',
       'Western National Mutual Insurance Company',
       'Western Surety Company', 'Westfield Insurance Company',
       'Westfield National Insurance Company',
       'Westport Insurance Corporation',
       'Williamsburg National Insurance Company',
       'Wilshire Insurance Company',
       'Windhaven National Insurance Company',
       'Work First Casualty Company',
       'Wright National Flood Insurance Company',
       'WRM America Indemnity Company, Inc.',
       'XL Insurance America, Inc.', 'XL Specialty Insurance Company',
       'Yosemite Insurance Company', 'Zale Indemnity Company',
       'Zenith Insurance Company', 'ZNAT Insurance Company',
       'Zurich American Insurance Company',
       'Zurich American Insurance Company of Illinois'], dtype=object)

United Financial Casualty Company                                            43
St. Paul Guardian Insurance Company                                          43
Shelter General Insurance Company                                            43
Pennsylvania Manufacturers' Association Insurance Company                    43
United Fire & Casualty Company                                               43
TDC National Assurance Company                                               43
OBI National Insurance Company                                               43
Travelers Casualty Company (The)                                             43
Wadena Insurance Company                                                     43
Liberty Mutual Fire Insurance Company                                        43
Travelers Commercial Insurance Company                                       43
OneCIS Insurance Company                                                     43
Southwest Marine and General Insurance Company                               43
UFG Specialty Insurance Company                                              43
Syncora Guarantee Inc.                                                       43
Infinity Insurance Company                                                   43
MGIC Indemnity Corporation                                                   43
Illinois Casualty Company                                                    43
SPARTA Insurance Company                                                     43
Iowa Mutual Insurance Company                                                43
Navigators Insurance Company                                                 43
Travelers Casualty Company of Connecticut                                    43
U.S. Underwriters Insurance Company                                          43
United Services Automobile Association, USAA Reciprocal, Attorney-in-fact    43
North Star Mutual Insurance Company                                          43
Nationwide Agribusiness Insurance Company                                    43
Triangle Insurance Company, Inc.                                             43
Ironshore Indemnity, Inc.                                                    43
Scor Reinsurance Company                                                     43
Peerless Insurance Company                                                   43
                                                                             ..
Consumers Insurance USA, Inc.                                                 1
Granite Re, Inc.                                                              1
Eastern Advantage Assurance Company                                           1
Atlanta International Insurance Company                                       1
Affirmative Direct Insurance Company                                          1
Work First Casualty Company                                                   1
GEICO Marine Insurance Company                                                1
First American Property & Casualty Insurance Company                          1
Accident Fund National Insurance Company                                      1
Genworth Financial Assurance Corporation                                      1
Universal Fire & Casualty Insurance Company                                   1
Fidelity and Guaranty Insurance Underwriters, Inc.                            1
Allied Eastern Indemnity Company                                              1
Chubb Indemnity Insurance Company                                             1
Endurance Assurance Corporation                                               1
Encompass Insurance Company                                                   1
Greyhawk Insurance Company                                                    1
Atlantic States Insurance Company                                             1
ARAG Insurance Company                                                        1
Atradius Trade Credit Insurance, Inc.                                         1
Compass Insurance Company                                                     1
Equity Insurance Company                                                      1
Fremont Insurance Company                                                     1
Diamond Insurance Company                                                     1
Colonial American Casualty and Surety Company                                 1
Alea North America Insurance Company                                          1
Everest Premier Insurance Company                                             1
ACA Financial Guaranty Corporation                                            1
Assured Guaranty Municipal Corp.                                              1
Doctors Direct Insurance, Inc.                                                1
Name: Company Name, Length: 856, dtype: int64

df['Line of Insurance'].nunique()

df['Line of Insurance'].unique()

df['Line of Insurance'].nunique()
df['Line of Insurance'].unique()
df['Line of Insurance'].value_counts()

43

array(['Farmowners Multiple Peril', 'Homeowners Multiple Peril',
       'Other Liability - Occcurence',
       'Other Private Passenger Auto Liability',
       'Other Commerical Auto Liability',
       'Private Passenger Physical Damage',
       'Commercial Auto Physical Damage', 'Boiler and Machinery',
       'Group Accident and Health (b)',
       'Guaranteed Renewable Accident and Health (b)',
       'Workers Compensation', 'Financial Guaranty',
       'Commerical Multiple Peril (Non-liability portion)',
       'Commerical Multiple Peril (Liability portion)', 'Inland Marine',
       'Other Liability - Claims Made', 'Surety', 'Fire', 'Allied Lines',
       'Ocean Marine', 'Medical Professional Liability', 'Earthquake',
       'All other Accident and Health (b)', 'Excess Workers Compensation',
       'Products Liability', 'Commercial Auto No Fault (pip)', 'Aircraft',
       'Fidelity', 'Burglary and Theft', 'Credit', 'Warranty',
       'Aggregate Write-In', 'Multiple Peril Crop', 'Federal Flood',
       'Private Passenger Auto No Fault (pip)',
       'Credit Accident and Health (Group and Individual)',
       'Other Accident only', 'Medicare Title XVIII',
       'Federal Employees Health Benefits Program Premium (b)',
       'Mortgage Guaranty',
       'Collectively Renewable Accident and Health (b)',
       'Non-Renewable for Stated Reasons only (b)',
       'Non-Cancelable Accident and Health (b)'], dtype=object)

Other Liability - Occcurence                             639
Workers Compensation                                     622
Inland Marine                                            601
Other Commerical Auto Liability                          594
Commercial Auto Physical Damage                          587
Commerical Multiple Peril (Non-liability portion)        569
Commerical Multiple Peril (Liability portion)            563
Allied Lines                                             559
Fire                                                     549
Private Passenger Physical Damage                        543
Other Private Passenger Auto Liability                   542
Other Liability - Claims Made                            536
Homeowners Multiple Peril                                526
Products Liability                                       520
Boiler and Machinery                                     513
Earthquake                                               511
Surety                                                   506
Fidelity                                                 496
Burglary and Theft                                       489
Group Accident and Health (b)                            470
Aggregate Write-In                                       469
Ocean Marine                                             466
Farmowners Multiple Peril                                462
Medical Professional Liability                           460
Aircraft                                                 453
Credit                                                   453
Excess Workers Compensation                              452
Federal Flood                                            451
Private Passenger Auto No Fault (pip)                    451
Warranty                                                 451
Multiple Peril Crop                                      449
Commercial Auto No Fault (pip)                           446
Mortgage Guaranty                                        440
Financial Guaranty                                       438
Guaranteed Renewable Accident and Health (b)             435
All other Accident and Health (b)                        435
Credit Accident and Health (Group and Individual)        433
Other Accident only                                      432
Collectively Renewable Accident and Health (b)           430
Medicare Title XVIII                                     429
Non-Renewable for Stated Reasons only (b)                429
Federal Employees Health Benefits Program Premium (b)    429
Non-Cancelable Accident and Health (b)                   428
Name: Line of Insurance, dtype: int64

Year 	Iowa Code Chapter 	State 	Company Name 	Line of Insurance 	Premiums Written 	Losses Paid 	Taxes Paid 	NAIC Number 	Iowa Company Code
0 	2017 	515.48 	WI 	1st Auto & Casualty Insurance Company 	Farmowners Multiple Peril 	73953 	3900 	754 	44725 	2894
1 	2017 	515.48 	WI 	1st Auto & Casualty Insurance Company 	Homeowners Multiple Peril 	30778 	7500 	314 	44725 	2894
2 	2017 	515.48 	WI 	1st Auto & Casualty Insurance Company 	Other Liability - Occcurence 	50931 	0 	520 	44725 	2894
3 	2017 	515.48 	WI 	1st Auto & Casualty Insurance Company 	Other Private Passenger Auto Liability 	1434294 	826321 	14632 	44725 	2894
4 	2017 	515.48 	WI 	1st Auto & Casualty Insurance Company 	Other Commerical Auto Liability 	72651 	28321 	741 	44725 	2894

#from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
scId = MinMaxScaler(feature_range=(0,1))
arr_scId = scId.fit_transform(df)
df_scId = pd.DataFrame(arr_scId, columns = df.columns)
df.head()
df_scId.head()
#from sklearn.svm import SVR

 	Iowa Code Chapter 	State 	Company Name 	Line of Insurance 	Premiums Written 	Losses Paid 	Taxes Paid
0 	0 	38 	0 	15 	73953 	3900 	754
1 	0 	38 	0 	23 	30778 	7500 	314
2 	0 	38 	0 	35 	50931 	0 	520
3 	0 	38 	0 	36 	1434294 	826321 	14632
4 	0 	38 	0 	33 	72651 	28321 	741
	Iowa Code Chapter 	State 	Company Name 	Line of Insurance 	Premiums Written 	Losses Paid 	Taxes Paid
0 	0.0 	0.974359 	0.0 	0.357143 	0.003836 	0.003360 	0.046529
1 	0.0 	0.974359 	0.0 	0.547619 	0.003629 	0.003382 	0.046360
2 	0.0 	0.974359 	0.0 	0.833333 	0.003726 	0.003338 	0.046439
3 	0.0 	0.974359 	0.0 	0.857143 	0.010370 	0.008179 	0.051861
4 	0.0 	0.974359 	0.0 	0.785714 	0.003830 	0.003503 	0.046524

#x = df.loc[:, 'Iowa Code Chapter','State', 'Company Name', 'Line of Insurance', 'Losses Paid','Taxes Paid']
x = df.drop('Premiums Written', axis = 1)
y = df['Premiums Written']

type(x)
type(y)

train_x, test_x, train_y, test_y = train_test_split(x,y,test_size = 0.3, random_state = 1)

train_x.shape
test_x.shape
train_y.shape
test_y.shape

(14809, 6)

(6347, 6)

(14809,)

(6347,)

train_x.head()
train_y.head()

 	Iowa Code Chapter 	State 	Company Name 	Line of Insurance 	Losses Paid 	Taxes Paid
6338 	0 	38 	482 	9 	0 	0
16598 	0 	8 	720 	38 	0 	0
13569 	0 	4 	651 	21 	0 	0
9926 	0 	12 	564 	35 	0 	1470
2087 	0 	13 	351 	3 	480 	1

6338          0
16598         0
13569         0
9926     181446
2087          0
Name: Premiums Written, dtype: int64

#create model instance
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm

LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

#fit the model on the training dataset
lm.fit(train_x, train_y)

LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

#fit the model on the training dataset
lm.fit(train_x, train_y)

LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

#predict
predict_test = lm.predict(test_x)

predict_test

array([ -1466.02000019,  19546.5246272 ,  84003.00218244, ...,
        37044.64360802,  39741.47250771, -42648.45299193])
        
        #check the accuracy
#print(regressor_min.score(x_test,y_test))
from sklearn.metrics import r2_score
r2_score_lm = r2_score(test_y,predict_test)
print(r2_score_lm)

0.8665957296753044

#MAE for test data
from sklearn.metrics import mean_absolute_error
mae_test = np.round(mean_absolute_error(test_y, predict_test), 4)
mae_test

145833.3382

from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
lm_lasso=Lasso()
lm_ridge=Ridge()
lm_elastic=ElasticNet()
lm_lasso
lm_ridge
lm_elastic

Lasso(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=None,
   selection='cyclic', tol=0.0001, warm_start=False)

Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
   normalize=False, random_state=None, solver='auto', tol=0.001)

ElasticNet(alpha=1.0, copy_X=True, fit_intercept=True, l1_ratio=0.5,
      max_iter=1000, normalize=False, positive=False, precompute=False,
      random_state=None, selection='cyclic', tol=0.0001, warm_start=False)
      
      Fit a model on train data

lm_lasso.fit(train_x,train_y)
lm_ridge.fit(train_x,train_y)
lm_elastic.fit(train_x,train_y)

Lasso(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=None,
   selection='cyclic', tol=0.0001, warm_start=False)

Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
   normalize=False, random_state=None, solver='auto', tol=0.001)

ElasticNet(alpha=1.0, copy_X=True, fit_intercept=True, l1_ratio=0.5,
      max_iter=1000, normalize=False, positive=False, precompute=False,
      random_state=None, selection='cyclic', tol=0.0001, warm_start=False)
      
      
      
      plt.figure(figsize=(15,10))
ft_importances_lm=pd.Series(lm.coef_,index=x.columns)
ft_importances_lm.plot(kind='barh')
plt.show();

print("RSquare Value for Lasso Regression TEST data is-")
np.round(lm_lasso.score(test_x,test_y)*100,2)
print("RSquare Value for Ridge Regression TEST data is-")
np.round(lm_ridge.score(test_x,test_y)*100,2)
print("RSquare Value for Elastic Net Regression TEST data is-")
np.round(lm_elastic.score(test_x,test_y)*100,2)

RSquare Value for Lasso Regression TEST data is-

86.66

RSquare Value for Ridge Regression TEST data is-

86.66

RSquare Value for Elastic Net Regression TEST data is-

86.66

predict_test_lasso=lm_lasso.predict(test_x)
predict_test_ridge=lm_ridge.predict(test_x)
predict_test_elastic=lm_elastic.predict(test_x)

#print the loss function MSE & MAE

import numpy as np
from sklearn import metrics
print("Lasso Regression Mean Square Error (MSE)for TEST data is-")
np.round(metrics.mean_squared_error(test_y,predict_test_lasso),2)
print("Ridge Regression Mean Square Error (MSE) for TEST data is-")
np.round(metrics.mean_squared_error(test_y,predict_test_ridge),2)
print("ElasticNet Mean Square Error (MSE) for TEST data is-")
np.round(metrics.mean_squared_error(test_y,predict_test_elastic),2)

import numpy as np

from sklearn import metrics

print("Lasso Regression Mean Square Error (MSE)for TEST data is-")

np.round(metrics.mean_squared_error(test_y,predict_test_lasso),2)

print("Ridge Regression Mean Square Error (MSE) for TEST data is-")

np.round(metrics.mean_squared_error(test_y,predict_test_ridge),2)

print("ElasticNet Mean Square Error (MSE) for TEST data is-")

np.round(metrics.mean_squared_error(test_y,predict_test_elastic),2)

Lasso Regression Mean Square Error (MSE)for TEST data is-

2033959223017.76

Ridge Regression Mean Square Error (MSE) for TEST data is-

2033959219262.72

ElasticNet Mean Square Error (MSE) for TEST data is-

2034001211602.7




