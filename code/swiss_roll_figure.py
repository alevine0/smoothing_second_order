import sklearn.datasets
import math
import matplotlib
import matplotlib.pyplot as plt
raw, raw_labels = sklearn.datasets.make_swiss_roll(random_state=2,noise=1.0, n_samples = 100)
feats = raw[:,::2]/10.
import pandas as pd
socsv = pd.read_csv('data/certify/swiss_roll/swiss_roll_net/noise_0.50/test/second_order_N_100000000',delimiter='\t')
base = pd.read_csv('data/certify/swiss_roll/swiss_roll_net/noise_0.50/test/N_100000000',delimiter='\t')
ones = (socsv['correct'] & (socsv['label'] == 1)).to_numpy().nonzero()[0]
zeros= (socsv['correct'] & (socsv['label'] == 0)).to_numpy().nonzero()[0]
fails= (1-socsv['correct'] ).to_numpy().nonzero()[0]
fig, ax = plt.subplots()
ax.set_xlim([-2,2])
ax.set_ylim([-2,2])
artists = [plt.Circle(feats[i],socsv['radius'][i],color='lightcoral',fill=False) for i in ones]
[ax.add_artist(a) for a in artists]
artists = [plt.Circle(feats[i],socsv['radius'][i],color='skyblue',fill=False) for i in zeros]
[ax.add_artist(a) for a in artists]
artists = [plt.Circle(feats[i],.02,color='red') for i in ones]
[ax.add_artist(a) for a in artists]
artists = [plt.Circle(feats[i],.02,color='blue') for i in zeros]
[ax.add_artist(a) for a in artists]
artists = [plt.Circle(feats[i],.02,color='black') for i in fails]
[ax.add_artist(a) for a in artists]
fig.savefig('second_order.png')

ones = (base['correct'] & (base['label'] == 1)).to_numpy().nonzero()[0]
zeros = (base['correct'] & (base['label'] == 0)).to_numpy().nonzero()[0]
fails= (1-base['correct'] ).to_numpy().nonzero()[0]
fig, ax = plt.subplots()
ax.set_xlim([-2,2])
ax.set_ylim([-2,2])
artists = [plt.Circle(feats[i],base['radius'][i],color='red',fill=False) for i in ones]
[ax.add_artist(a) for a in artists]
artists = [plt.Circle(feats[i],base['radius'][i],color='blue',fill=False) for i in zeros]
[ax.add_artist(a) for a in artists]
artists = [plt.Circle(feats[i],.02,color='indianred') for i in ones]
[ax.add_artist(a) for a in artists]
artists = [plt.Circle(feats[i],.02,color='steelblue') for i in zeros]
[ax.add_artist(a) for a in artists]
artists = [plt.Circle(feats[i],.02,color='black') for i in fails]
[ax.add_artist(a) for a in artists]
fig.savefig('base_2.png')

onesbase = ones
zerosbase = zeros

ones = (socsv['correct'] & (socsv['label'] == 1)).to_numpy().nonzero()[0]
zeros= (socsv['correct'] & (socsv['label'] == 0)).to_numpy().nonzero()[0]
fails= (1-socsv['correct'] ).to_numpy().nonzero()[0]
fig, ax = plt.subplots()
ax.set_xlim([-2,2])
ax.set_ylim([-2,2])
print(len(ones))
print(len(zeros))

artists = [matplotlib.patches.Wedge(feats[i],socsv['radius'][i],0,360,width= (socsv['radius'][i] - base['radius'][i]),color='lightcoral') for i in ones[::5]]
[ax.add_artist(a) for a in artists]
artists = [matplotlib.patches.Wedge(feats[i],socsv['radius'][i],0,360,width= (socsv['radius'][i] - base['radius'][i]),color='skyblue') for i in zeros[::7]]
[ax.add_artist(a) for a in artists]
artists = [plt.Circle(feats[i],base['radius'][i],color='red',fill=False) for i in ones[::5]]
[ax.add_artist(a) for a in artists]
artists = [plt.Circle(feats[i],base['radius'][i],color='blue',fill=False) for i in zeros[::7]]
[ax.add_artist(a) for a in artists]
artists = [plt.Circle(feats[i],.02,color='indianred') for i in ones]
[ax.add_artist(a) for a in artists]
artists = [plt.Circle(feats[i],.02,color='steelblue') for i in zeros]
[ax.add_artist(a) for a in artists]
artists = [plt.Circle(feats[i],.02,color='red') for i in ones[::5]]
[ax.add_artist(a) for a in artists]
artists = [plt.Circle(feats[i],.02,color='blue') for i in zeros[::7]]
[ax.add_artist(a) for a in artists]
artists = [plt.Circle(feats[i],.02,color='black') for i in fails]
[ax.add_artist(a) for a in artists]
fig.savefig('second_order_2_2.png')

raw, raw_labels = sklearn.datasets.make_swiss_roll(random_state=1,noise=1.0, n_samples = 1000)
feats = raw[:,::2]/10.
labels = (raw_labels > 3*math.pi).astype(int)
fig, ax = plt.subplots()
ax.set_xlim([-2,2])
ax.set_ylim([-2,2])
ones = (labels == 1).nonzero()[0]
zeros = (labels==0).nonzero()[0]
artists = [plt.Circle(feats[i],.02,color='red') for i in ones]
[ax.add_artist(a) for a in artists]
artists = [plt.Circle(feats[i],.02,color='blue') for i in zeros]
[ax.add_artist(a) for a in artists]
fig.savefig('training_2.png')
