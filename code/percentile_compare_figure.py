import math
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# exp_path = 'small_mnist/small_mnist_net'
# exp_name = '7X7 MNIST'
# exp_path = 'small_mnist_10pcs/small_mnist_pca_net'
# exp_name = '10 PCs MNIST'
# exp_path = 'cifar10/resnet110'
# exp_name = 'CIFAR'
exp_path = 'imagenet/resnet50'
exp_name = 'ImageNet'
st_dev = '1.00'
N_exp = 6
method = 'dipole'
method_human = 'Dipole'
# method = 'second_order'
# method_human = 'SoS'
socsv = pd.read_csv('data/certify/'+exp_path+'/noise_'+st_dev+'/test/'+method+'_N_1' + '0'*N_exp,delimiter='\t')
base = pd.read_csv('data/certify/'+exp_path+'/noise_'+st_dev+'/test/N_1' + '0'*N_exp,delimiter='\t')

percentiles = socsv['radius']/base['radius']
total = len(base)
filtered = percentiles.to_numpy()[(socsv['correct'] & base['correct']).to_numpy().nonzero()]

ranges = [0,.75,.85,.95,.99,1.01,1.05,1.15,1.25, math.inf]
bincount = len(ranges) -1
bins = [ ((filtered > ranges[i]) & (filtered <= ranges[i+1])).sum() for i in range(bincount)]
fail_soc = ((1-socsv['correct']) & (base['correct'])).sum()
fail_base = ((socsv['correct']) & (1-base['correct'])).sum()

bins = [fail_soc] + bins + [fail_base]

labels = ['Correct cert.\nfor baseline only',
	'25%+\ndecrease',
	'15%-25%\ndecrease',
	'5%-15%\ndecrease',
	'1%-5%\ndecrease',
	'+/- 1%',
	'1%-5%\nincrease',
	'5%-15%\nincrease',
	'15%-25%\nincrease',
	'25%+\nincrease',
	'Correct cert.\nfor '+method_human+' only']
colors = ['red'] * 5 + ['black'] + ['blue'] * 5
plt.figure(figsize=(12,6))
ax = plt.gca()
print(bins)

ax.bar(labels,bins,color=colors)
plt.setp(ax.get_xticklabels(), rotation=90,fontsize=16)
plt.setp(ax.get_yticklabels(), fontsize=16)

ax.set_ylim([0,max(int(.85*total),max(bins))])
plt.title(method_human+', '+ exp_name+', σ=' + st_dev+ ', N=$10^'+ str(N_exp) + '$',fontsize=20)
plt.tight_layout()
plt.savefig(exp_name +'_'+method+'_sigma_'+ st_dev+ '_N_10_'+ str(N_exp)+ '.png')