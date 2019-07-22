import pickle
with open('./data/challenge_fun.pkl', 'rb') as f:
    data = pickle.load(f)
print(data)