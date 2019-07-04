import os
import time
import pickle
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch import optim

import matplotlib.cm as cm
import matplotlib.pyplot as plt

from models.simple_hrl_classic import *
from models.embeddings_model import *
from create_dataset import fullstate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###------------------------------------- Arguments ----------------------------------------###


parser = argparse.ArgumentParser()
parser.add_argument('--resume-experiment', default=False, action="store_true")
parser.add_argument('--rnn', default=False, action="store_true")
parser.add_argument('--single-minibatch', default=False, action="store_true")
parser.add_argument('--test-only', default=False, action="store_true")
parser.add_argument('--use-features', default=False, action="store_true")
parser.add_argument('--num-epochs', default=10, type= int)
parser.add_argument('--num-iter', default=10000, type= int)
parser.add_argument('--minibatch-size', default=16, type= int)
parser.add_argument('--minibatch-size-test', default=64, type= int)
parser.add_argument('--checkpoint-directory', default="../experiments_psketch/hrl/embedding_checkpoint_rnn_1.pt", type=str)
parser.add_argument('--data-directory', default="../data_psketch/", type=str)
parser.add_argument('--key-list', default="-1", type=str)
args = parser.parse_args()


###---------------------------- Getting Skill Trajectories --------------------------------###


def sub_policies(num):
	# These policies are defined in "models.simple_hrl_classic"
	if num == 1:
		return get_wood
	elif num == 2:
		return get_iron
	elif num == 3:
		return get_grass
	elif num == 4:
		return make0
	elif num == 5:
		return make1
	elif num == 6:
		return make2
	elif num == 7:
		return get_gold
	elif num == 8:
		return get_gem

def sample_sequence(Q, start, num_samples=1):
	Ts_list = []
	for _ in range(num_samples):
		Ts = []
		s = start
		Ts.append(s)
		for pi in Q:
			seq = sub_policies(pi)(s)
			if not np.any(seq):
				continue
			for a in seq:
				_, s = s.step(a)
				Ts.append(s)
		Ts_list.append(Ts)
	return Ts_list

def get_segmentation_index(traj):
	inv_change = []
	indices = []
	pr_s = traj[0]
	for ind, s in enumerate(traj[1:]):
		inv = np.where(s.inventory - pr_s.inventory)[0]
		pr_s = s
		if inv.any(): 
			inv_change.append(inv)
			indices.append(ind + 1)
	return indices, inv_change


###--------------------------------- Getting Embeddings ------------------------------------### 


class CurriculumDataset():
	"""docstring for CurriculumDataset"""
	def __init__(self, maps, key_list = [-1], previous_checkpoint = None, max_num_skills = 5, num_demos = 10000, save_dataset = False, save_str=""):
		if previous_checkpoint:
			bytes_in = bytearray(0)
			max_bytes = 2**31 - 1
			input_size = os.path.getsize(previous_checkpoint)
			with open(previous_checkpoint, 'rb') as f_in:
				for _ in range(0, input_size, max_bytes):
					bytes_in += f_in.read(max_bytes)
				dataset = pickle.loads(bytes_in)
			self.padded_demos = dataset["padded_demos"]
			self.demos = dataset["demos"]
			self.labels_detect = dataset["labels_detect"]
			self.labels_predict = dataset["labels_predict"]
			self.flags_detect = dataset["flags_detect"]
			self.flags_predict = dataset["flags_predict"]
			return None
		start = time.clock()
		all_maps = self.get_all_maps(maps)
		if key_list == [-1]:
			key_list = list(range(1,9))
		else:
			assert key_list.__class__ == list
		self.demos = []
		self.labels_detect = []
		self.labels_predict = []
		self.flags_detect = []
		self.flags_predict = []
		for i in range(num_demos):
			num_q = np.random.choice(max_num_skills)
			# Skills, maps and number of skills chosen by random
			Q = np.random.choice(key_list, num_q)
			env_i = np.random.choice(len(all_maps))
			demo_i = []
			label_predict_i = []
			label_detect_i = []
			flag_predict_i = []
			flag_detect_i = []
			for q in Q:
				try:
					demo_q = sample_sequence([q], all_maps[env_i])[0]
					if len(demo_q) > 1:
						demo_i += demo_q
						label_predict_i += [0]*(len(demo_q) - 1) + [q]
						label_detect_i += [0]*(len(demo_q) - 1) + [1]
						flag_predict_i += [0]*(len(demo_q) - 1) + [1]
						flag_detect_i += [1]*(len(demo_q))
				except:
					continue
			if demo_i: 
				self.demos.append(demo_i)
				self.labels_detect.append(label_detect_i)
				self.labels_predict.append(label_predict_i)
				self.flags_detect.append(flag_detect_i)
				self.flags_predict.append(flag_predict_i)
		# Pad the trajectories
		self.create_padding()
		print("Time taken to create dataset {}: {}".format(save_str, time.clock() - start))
		if save_dataset:
			key_str = ""
			for key in key_list:
				key_str += str(key)
			dir_name = args.data_directory + "skill_dict_" + key_str + "_num_" + str(max_num_skills) + "_" + save_str + ".pt"
			max_bytes = 2**31 - 1
			if save_dataset:
				start = time.clock()
				dataset_dict = {"demos": self.demos, "labels_detect": self.labels_detect, "labels_predict": self.labels_predict,\
					"flags_detect": self.flags_detect, "flags_predict": self.flags_predict, "padded_demos": self.padded_demos}
				bytes_out = pickle.dumps(dataset_dict)
				with open(dir_name, "wb") as f_out:
					for idx in range(0, len(bytes_out), max_bytes):
						f_out.write(bytes_out[idx:idx+max_bytes])
				print("Time taken to save the dataset {}: {}".format(save_str, time.clock() - start))


	def get_all_maps(self, train_maps):
		maps = []
		for key in train_maps.keys():
			maps += train_maps[key]
			if key == 7 or key == 8:
				maps += train_maps[key]
		return maps


	def create_padding(self):
		max_length = 0
		for demo in self.demos:
			if len(demo) > max_length:
				max_length = len(demo)
		self.padded_demos = []
		for i in range(len(self.demos)):
			if args.use_features:
				self.padded_demos.append(np.array([s.features() for s in self.demos[i]] + [np.zeros(shape=(self.demos[0][0].features().shape))]*(max_length-len(self.demos[i]))))
			else:
				self.padded_demos.append(np.array([fullstate(s) for s in self.demos[i]] + [np.zeros(shape=(fullstate(self.demos[0][0]).shape))]*(max_length-len(self.demos[i]))))
			self.labels_detect[i] += [0]*(max_length-len(self.demos[i]))
			self.labels_predict[i] += [0]*(max_length-len(self.demos[i]))
			self.flags_detect[i] += [0]*(max_length-len(self.demos[i]))
			self.flags_predict[i] += [0]*(max_length-len(self.demos[i]))


	def sample(self, minibatch_size):
		if minibatch_size > len(self.demos):
			print("Returning the entire dataset")
			indices = np.array(range(len(self.demos)))
		else:
			indices = np.random.choice(range(len(self.demos)), minibatch_size, replace=False)
			
		demos_sample = np.empty((0,) + self.padded_demos[0].shape, float)
		labels_detect_sample = []
		labels_predict_sample = []
		flags_detect_sample = []
		flags_predict_sample = []
		for i in indices:
			try:
				demos_sample = np.append(demos_sample, np.expand_dims(self.padded_demos[i], axis=0), axis=0)
			except:
				print(i, self.padded_demos[i].shape)
				continue
			labels_detect_sample.append(self.labels_detect[i])
			labels_predict_sample.append(self.labels_predict[i])
			flags_detect_sample.append(self.flags_detect[i])
			flags_predict_sample.append(self.flags_predict[i])
		return demos_sample.swapaxes(0,1), np.array(labels_detect_sample), np.array(labels_predict_sample), \
				np.array(flags_detect_sample), np.array(flags_predict_sample)


	def sample_cnn(self, minibatch_size):
		state_diff_sample = []
		labels_detect_sample = []
		labels_predict_sample = []
		flags_detect_sample = []
		flags_predict_sample = []
		
		if minibatch_size > len(self.demos):
			print("Returning the entire dataset")
			indices = np.array(range(len(self.demos)))
		else:
			indices = np.random.choice(range(len(self.demos)), minibatch_size, replace=False)
			
		for i in indices:
			demos_sample_i = self.demos[i]
			labels_detect_i = self.labels_detect[i]
			labels_predict_i = self.labels_predict[i]
			flags_detect_i = self.flags_detect[i]
			flags_predict_i = self.flags_predict[i]
			for it in range(1, len(demos_sample_i)):	
				state_diff_sample.append(fullstate(demos_sample_i[it]) - fullstate(demos_sample_i[it-1]))
				labels_detect_sample.append(labels_detect_i[it])
				labels_predict_sample.append(labels_predict_i[it])
				flags_detect_sample.append(flags_detect_i[it])
				flags_predict_sample.append(flags_predict_i[it])
		return np.array(state_diff_sample), np.array(labels_detect_sample), np.array(labels_predict_sample), \
				np.array(flags_detect_sample), np.array(flags_predict_sample)


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def plot_losses(all_losses, losses_test_end, losses_test_skill, losses_train_end, losses_train_skill, epoch_changes):
	# End losses
	plt.figure(0)
	plt.plot(np.arange(0,len(losses_train_end)*50, 50), smooth(losses_train_end, 5), 'r')
	plt.plot(np.arange(0,len(losses_test_end)*50, 50), smooth(losses_test_end, 5), 'g')
	plt.title("end_losses")
	# Add vertical lines for curriculum changes
	for vline in epoch_changes:
		plt.axvline(x=vline)
	# Skill losses
	plt.figure(1)
	plt.plot(np.arange(0,len(losses_train_skill)*50, 50), smooth(losses_train_skill, 5), 'r')
	plt.plot(np.arange(0,len(losses_test_skill)*50, 50), smooth(losses_test_skill, 5), 'g')
	plt.title("skill_losses")
	# Add vertical lines for curriculum changes
	for vline in epoch_changes:
		plt.axvline(x=vline)
	# Total loss
	plt.figure(2)
	plt.plot(np.arange(0,len(all_losses), 1), smooth(all_losses, 5), 'b-')
	plt.plot(np.arange(0,len(losses_train_end)*50, 50), smooth([sum(x) for x in zip(losses_train_end, losses_train_skill)], 5), 'r')
	plt.plot(np.arange(0,len(losses_test_end)*50, 50), smooth([sum(x) for x in zip(losses_test_end, losses_test_skill)], 5), 'g')
	plt.title("total_loss")
	#Show all
	plt.show()


###----------------------------- Main -------------------------------### 


def main():
	# Define the model and optimizer
	if args.rnn:
		model = EmbeddingRNN(1076, 256, 9, device, args.use_features)
	else:
		model = EmbeddingCNN(1076, 256, 9, device, args.use_features)
	model_optimizer = optim.Adam(model.parameters())
	if torch.cuda.is_available():
		model.cuda()
	# Sorting out the key list
	key_list = [int(it) for it in args.key_list.split(',')]
	# Load previous checkpoint
	epoch_start = 1
	all_losses = []
	test_end_losses = []
	test_skill_losses = []
	train_end_losses = []
	train_skill_losses = []
	if os.path.isfile(args.checkpoint_directory) and (args.resume_experiment or args.test_only):
		print("Loading previous checkpoint")
		checkpoint = torch.load(args.checkpoint_directory, map_location={'cuda:0': 'cpu'})
		model.load_state_dict(checkpoint['model_state_dict'])
		model_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		epoch_start = checkpoint["epoch"]
		all_losses = checkpoint["losses"]
		test_end_losses = checkpoint["test_end_losses"]
		test_skill_losses = checkpoint["test_skill_losses"]
		train_end_losses = checkpoint["train_end_losses"]
		train_skill_losses = checkpoint["train_skill_losses"]
	# If test_only, then we only try to predict the sequence for different keys
	if args.test_only:
		plot_losses(all_losses, test_end_losses, test_skill_losses, train_end_losses, train_skill_losses, np.arange(args.num_epochs)*args.num_iter)
		demo_dict = pickle.load(open(args.data_directory + "demo_dict.pk", "rb"))
		sequence_list = [[1,4], [1,5], [3,6], [3,4], [2,1,6], [1,4,3,5], [1,5,2,4], [1,5,2,5], [2,1,5,7], [1,5,2,4,8]]
		for key, Q in zip(demo_dict.keys(), sequence_list):
			true_positives_total = 0
			false_positives_total = 0
			true_negatives_total = 0
			total_datapoints = 0
			correct_skill_total = 0
			overall_skill_total = 0
			#print("\n\n\n")
			print("Performance of Q:{}".format(Q))
			for _ in range(10):
				demo = np.random.choice(demo_dict[key])
				seg_index_gt, _ = get_segmentation_index(demo)
				Q_pred, seg_index = model.get_prediction(demo)
				true_positives = len(set(seg_index_gt) & set(seg_index))
				false_positives = len(set(seg_index) - set(seg_index_gt))
				true_negatives = len(set(seg_index_gt) - set(seg_index))
				total_datapoints_per_key = len(set(seg_index_gt) | set(seg_index))
				#print("Ground truth || Q:{} \t ind:{}".format(Q, seg_index_gt))
				#print("Prediction || Q:{} \t ind:{}".format(Q_pred, seg_index))
				#print("Fraction || True Positives: {}/{}, False Positives: {}/{}, True Negatives: {}/{}"\
					#.format(true_positives, total_datapoints_per_key, false_positives, total_datapoints_per_key, true_negatives, total_datapoints_per_key))
				#print("Percent || True Positives: {}, False Positives: {}, True Negatives: {}"\
					#.format(true_positives/total_datapoints_per_key*100, false_positives/total_datapoints_per_key*100, true_negatives/total_datapoints_per_key*100))
				true_positives_total += true_positives
				false_positives_total += false_positives
				true_negatives_total += true_negatives
				total_datapoints += total_datapoints_per_key
				correct_skill_per_key = 0
				overall_skill_per_key = true_positives
				for dp in list(set(seg_index_gt) & set(seg_index)):
					ind_gt = seg_index_gt.index(dp)
					ind = seg_index.index(dp)
					if Q[ind_gt] == Q_pred[ind]:
						correct_skill_per_key += 1
					else:
						#print("Predicted skill {} instead of {}".format(Q_pred[ind], Q[ind_gt]))
						pass
				correct_skill_total += correct_skill_per_key
				overall_skill_total += overall_skill_per_key
				#print("Skill prediction accuracy || Fraction:{}/{}, Percent:{}".format(correct_skill_per_key, overall_skill_per_key, \
					#correct_skill_per_key/overall_skill_per_key*100))
				#print("")
			#print("\n\n\n")
			#print("------ Overall Stats --------")
			#print("Fraction || True Positives: {}/{}, False Positives: {}/{}, True Negatives: {}/{}"\
			#			.format(true_positives_total, total_datapoints, false_positives_total, total_datapoints, true_negatives_total, total_datapoints))
			#print("Percent || True Positives: {}, False Positives: {}, True Negatives: {}"\
			#			.format(true_positives_total/total_datapoints*100, false_positives_total/total_datapoints*100, true_negatives_total/total_datapoints*100))
			#print("Skill prediction accuracy || Fraction:{}/{}, Percent:{}".format(correct_skill_total, overall_skill_total, \
			#			correct_skill_total/overall_skill_total*100))
			if overall_skill_total == 0 or total_datapoints == 0:
				print("Something wrong with data, moving on to the next thing")
			else:
				print("True Positives: {}/{} = {:.2f}\nSkill Prediction accuracy: {}/{}={:.2f}\n".format(true_positives_total, total_datapoints, \
					true_positives_total/total_datapoints, correct_skill_total, overall_skill_total, correct_skill_total/overall_skill_total))
		print("Detection loss | Train | Avg (5, 15, 50) == ({:.4f}, {:.4f}, {:.4f})".format(sum(train_end_losses[-5:])/5, \
			sum(train_end_losses[-15:])/15, sum(train_end_losses[-50:])/50))
		print("Detection loss | Validation | Avg (5, 15, 50) == ({:.4f}, {:.4f}, {:.4f})".format(sum(test_end_losses[-5:])/5, \
			sum(test_end_losses[-15:])/15, sum(test_end_losses[-50:])/50))
		import ipdb; ipdb.set_trace()
	# Load maps
	# Curriculum dataset, arguments
	# maps, key_list = -1, max_num_skills = 5, previous_checkpoint = None, num_demos = 10000, save_dataset = False
	if not args.resume_experiment:
		train_maps = pickle.load(open(args.data_directory + "maps.pk", "rb"))
		test_maps = pickle.load(open(args.data_directory + "maps_test.pk", "rb")) 
		dataset_test = CurriculumDataset(test_maps, key_list = key_list, num_demos=3000, save_dataset=True, save_str = "test")
		# Beta-testing the model with a single mini-batch
		if args.single_minibatch:
			dataset = CurriculumDataset(train_maps, key_list = key_list, num_demos=3000, save_dataset=True, save_str = "train_" + str(epoch))
			batch = dataset.sample(args.minibatch_size)
	else:
		dataset_test = CurriculumDataset(None, previous_checkpoint= args.data_directory+"skill_dict_12345678_num_5_test.pk")
		# Beta-testing the model with a single mini-batch
		if args.single_minibatch:
			dataset = CurriculumDataset(None, previous_checkpoint=args.data_directory+"skill_dict_12345678_num_5_train_" + str(epoch) + ".pk")
			batch = dataset.sample(args.minibatch_size)
	# Now, training in a curriculum setting
	for epoch in range(epoch_start, epoch_start + args.num_epochs):
		if not args.single_minibatch:
			if not args.resume_experiment:
				dataset = CurriculumDataset(train_maps, key_list = key_list, num_demos=5000, save_dataset=True, save_str = "train_" + str(epoch))
			else:
				dataset = CurriculumDataset(None, previous_checkpoint=args.data_directory+"skill_dict_12345678_num_5_train_" + str(epoch) + ".pk")
		for _ in range(0, args.num_iter):
			# Are we beta-testing for single minibatch
			if args.single_minibatch:
				loss_end, loss_skill = model.get_loss(*batch)
			else:	
				loss_end, loss_skill = model.get_loss(*dataset.sample(args.minibatch_size))
			# Take a gradient step
			loss = loss_end + loss_skill
			model_optimizer.zero_grad()
			loss.backward()
			model_optimizer.step()
			all_losses.append(loss.item())
			if _ % 50 == 0:
				train_end_losses.append(loss_end.item())
				train_skill_losses.append(loss_skill.item())
				test_end_loss, test_skill_loss = model.get_loss(*dataset_test.sample(args.minibatch_size_test))
				test_end_losses.append(test_end_loss.item())
				test_skill_losses.append(test_skill_loss.item())
				print("Epoch:{}, Iter:{} is train:{:.5f}, test:{:.5f}".format(epoch, _, loss.item(),\
					test_end_loss.item() + test_skill_loss.item()))
				torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
					'optimizer_state_dict': model_optimizer.state_dict(),'losses': all_losses, 
					'train_end_losses': train_end_losses, 'train_skill_losses': train_skill_losses, 
					'test_end_losses': test_end_losses, 'test_skill_losses': test_skill_losses}, args.checkpoint_directory)
	# Plotting stuff
	plot_losses(all_losses, test_end_losses, test_skill_losses, train_end_losses, train_skill_losses, np.arange(args.num_epochs)*args.num_iter)
	import ipdb; ipdb.set_trace()


if __name__ == "__main__":
	main()

