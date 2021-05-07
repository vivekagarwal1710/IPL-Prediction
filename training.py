
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals import joblib
import pickle
import pandas as pd
import numpy as np




df = pd.read_csv("all_matches.csv",low_memory=False);

df.drop(['season','legbyes','byes','wides','penalty','other_wicket_type','noballs','other_player_dismissed'],axis=1,inplace=True)

for col in df.columns[df.isna().any()].tolist():
    if col not in ['wicket_type','player_dismissed']:
        df[col].fillna(0.0,inplace=True)

df.drop(df[df['start_date']<'2014'].index,inplace=True)
df.drop(df[df["batting_team"] =='Rising Pune Supergiants' ].index,inplace=True)
df.drop(df[df["batting_team"] =='Rising Pune Supergiant' ].index,inplace=True)
df.drop(df[df["batting_team"]=='Gujarat Lions' ].index,inplace=True)
df.drop(df[df["bowling_team"] =='Rising Pune Supergiants' ].index,inplace=True)
df.drop(df[df["bowling_team"] =='Rising Pune Supergiant' ].index,inplace=True)
df.drop(df[df["bowling_team"]=='Gujarat Lions' ].index,inplace=True)
df['batting_team'] = df['batting_team'].replace(['Delhi Daredevils'],'Delhi Capitals')
df['bowling_team'] = df['bowling_team'].replace(['Delhi Daredevils'],'Delhi Capitals')
df['venue'] = df['venue'].replace(['M Chinnaswamy Stadium'],'M Chinnaswamy Stadium')
df['venue'] = df['venue'].replace(['M.Chinnaswamy Stadium'],'M Chinnaswamy Stadium')
df['venue'] = df['venue'].replace(['Feroz Shah Kotla'],'Arun Jaitley Stadium')
df['venue'] = df['venue'].replace(['MA Chidambaram Stadium, Chepauk, Chennai'],'MA Chidambaram Stadium')
df['venue'] = df['venue'].replace(['Wankhede Stadium, Mumbai'],'Wankhede Stadium')
df['venue'] = df['venue'].replace(['Sardar Patel Stadium, Motera'],'Narendra Modi Stadium')
df['batting_team'] = df['batting_team'].replace(['Kings XI Punjab'],'Punjab Kings')
df['bowling_team'] = df['bowling_team'].replace(['Kings XI Punjab'],'Punjab Kings')
df["venue"] = df["venue"].astype('category')
df["venue_no"] = df["venue"].cat.codes
df["batting_team"] = df["batting_team"].astype('category')
df["batting_no"] = df["batting_team"].cat.codes
df["bowling_team"] = df["bowling_team"].astype('category')
df["bowling_no"] = df["bowling_team"].cat.codes
df.drop(df[df["ball"]>=6].index,inplace=True)
df.drop(df[df['innings']>2].index,inplace=True)

players = set()
for index , row in df.iterrows():
    players.add(row['striker'])
    players.add(row['non_striker'])
    players.add(row['bowler'])

players=list(players)
players.sort()

# converting set to dictionary
players_dict = dict(zip(list(range(len(players))), players))

striker_no=[]
non_striker_no=[]
bowler_no=[]
tot_score1=[]
tot_score2=[]

def getkey(val,dict_name):
    for key, value in dict_name.items():
          if val == value:
              return key

sum1=dict()
sum2=dict()
idx=0
match_id=0
cnt=0


for index,row in df.iterrows(): 

    if(cnt==0):
        match_id=row['match_id']
        sum1[idx]=0
        sum2[idx]=0
        cnt=300

    if row['match_id']!=match_id:
        idx+=1
        sum1[idx]=0
        sum2[idx]=0

    if((row['innings'] == 1) ):
        sum1[idx] += row['runs_off_bat']+row['extras']

    if((row['innings'] == 2) ):
        sum2[idx] += row['runs_off_bat']+row['extras']
    match_id=row['match_id']

cnt=0
first_innings_score=[]
second_innings_score=[]
f=0
for index,row in df.iterrows():
    if f==0:
        match_id=row['match_id']
        f=1
    if match_id!=row['match_id']:
        cnt+=1
    first_innings_score.append(sum1[cnt])
    second_innings_score.append(sum2[cnt])
    striker_no.append(getkey(row['striker'],players_dict))
    non_striker_no.append(getkey(row['non_striker'],players_dict))
    bowler_no.append(getkey(row['bowler'],players_dict))
    
    match_id=row['match_id']

df['striker_no']=striker_no
df['non_striker_no']=non_striker_no
df['bowler_no']=bowler_no
df['first_innings_score']=first_innings_score
df['second_innings_score']=second_innings_score
    
    
    
X = np.matrix(df.iloc[:,[3,14,15,16,17,18,19,19]])

y1= np.array(df.iloc[:,20])
y2= np.array(df.iloc[:,21])

#model1 = DecisionTreeRegressor(random_state=1,max_features=5,max_depth=24)
model1 = DecisionTreeRegressor()       
model1.fit(X,y1)

filename1 = "innings1.joblib"
joblib.dump(model1, filename1)

#model2 = DecisionTreeRegressor(random_state=1,max_features=5,max_depth=24)
model2 = DecisionTreeRegressor()    
model2.fit(X,y2)

filename2 = "innings2.joblib"
joblib.dump(model2, filename2)

teams=dict( enumerate(df['batting_team'].sort_values().unique()))
venuedict=dict( enumerate(df['venue'].sort_values().unique()))

a_file = open("venuedict.pkl", "wb")
pickle.dump(venuedict, a_file)
a_file.close()
a_file = open("teamdict.pkl", "wb")
pickle.dump(teams, a_file)
a_file.close()
a_file = open("playersdict.pkl", "wb")
pickle.dump(players_dict, a_file)
a_file.close()
