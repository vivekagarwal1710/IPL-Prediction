try:
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.tree import DecisionTreeRegressor
    import joblib
    import pickle
    import pandas as pd
    import numpy as np
except:
    def predictRuns(input_test):
        return

def getkey(val,dict_name):
        for key, value in dict_name.items():
            if val == value:
                return key

def predictRuns(input_test):
    try:
        ifi=pd.read_csv(input_test)
        inputvenue=ifi['venue'][0]
        input_innings=ifi['innings'][0]
        a_file = open("teamdict.pkl", "rb")
        teams = pickle.load(a_file)
        a_file.close()
        input_batting=ifi['batting_team'][0]
        input_bowling=ifi['bowling_team'][0]
        batnum=getkey(input_batting,teams)
        bowlnum=getkey(input_bowling,teams)
        
        input_batsmen=ifi['batsmen'][0].split(',')
        input_bowlers=ifi['bowlers'][0].split(',')
        inbatsmen=list()
        inbowlers=list()
        a_file = open("playersdict.pkl", "rb")
        players_dict= pickle.load(a_file)
        a_file.close()
        def getkeyplayer(val):
            for key, value in players_dict.items():
                if val == value:
                    return key
            return 300
        for i in input_batsmen:
            inbatsmen.append(getkeyplayer(i))
        for i in input_bowlers:
            inbowlers.append(getkeyplayer(i))
        
        inbatsmen.sort()
        inbowlers.sort()
        a_file = open("venuedict.pkl", "rb")
        venuedict= pickle.load(a_file)
        a_file.close()
        vnum=getkey(inputvenue,venuedict)
        
        if input_innings==1:
                    model = joblib.load('innings1.joblib')
        elif input_innings==2:
                    model = joblib.load('innings2.joblib')
        y_hats = model.predict([[input_innings,vnum,batnum,bowlnum,inbatsmen[0],inbatsmen[1],inbowlers[0],inbowlers[1]]])
        prediction=round(y_hats[0])
        return prediction
    except:
        return 45