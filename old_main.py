import pickle

from segmentation_inventory import predict_sequence as predict_sequence_inventory
from segmentation_agent_pos import predict_sequence as predict_sequence_agent_pos

# ------
#import argparse

#parser = argparse.ArgumentParser()
#parser.add_argument('--key', default=9, type=int)
#parser.add_argument('--example', default=0, type=int)
# parser.add_argument('--inventory', default=False, action="store_true")
# parser.add_argument('--neural-network', default=False, action="store_true")
#args = parser.parse_args()


# ------

def doyourthing(demos, key, example, render=True):
	demo_eg = demos[key][example]
	if render: 
		for s in demo_eg: s.render()
	Q2, seg2 = predict_sequence_agent_pos(demo_eg)
	try:
		Q1, seg1 = predict_sequence_inventory(demo_eg)
		if render: print("Target sketch:{}, \nQ1:{}, Q2:{}".format(key2sketch[key], Q1, Q2))
	except:
		if render: print("Target sketch:{}, \t\tQ2:{}".format(key2sketch[key], Q2))
	#if ( not key2sketch == Q1 ) or ( not key2sketch == Q2 ): 
	#	return example
	
# ------

def main():
	demos = pickle.load(open("../data_psketch/demo_dict.pk", "rb"))
	x = True
	while x:
		for i, demo in enumerate(demos[9]):
			if sum(sum(demo[0].grid[:,:,11])) == 1:
				x = False
				break
	import ipdb; ipdb.set_trace()
	doyourthing(demos, 9, i)
		
# ------

if __name__ == "__main__":
	main()
