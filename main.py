from agent_level_4 import *
from create_dataset import fullstate

# --------------------------------------- Rule Book ----------------------------------------- #

class RuleBook():
	def __init__(self, agent):
		self.agent = agent
		self.inventory = {"wood": 0, "iron": 0, "grass": 0, "complex": 0}
		self.state_conditions = [ self.in_front, self.inventory]
		self.action_conditions = [ self.use]
		self.rules = [({0: 8}, {0: True}, {0: 0}), ({0: 3}, {0: True}, {0: 3})]
		# Rule 1: Using wood makes it disappear
		# Rule 2: Using workshop does not make it disappear

	def use(self, action):
		if action == 4:
			return True
		else:
			return False

	def in_front(self, state):
		import ipdb; ipdb.set_trace()
		lx, ly = np.where(state[:,:,11] == -1)
		object_in_front = np.where(state[lx,ly] == 1)
		if object_in_front[0]:
			return object_in_front[0]
		else:
			return 0

	def event_predict(self, demo_model, basic_predict):
		rule_array = np.zero((len(demo_model), len(self.rules)))
		for state in demo_model:
			for ir, rule in enumerate(self.rules):
				rule[0].keys()
		return None


grid2num = {}
string_num_dict = { "w0": 3, "w1": 4, "w2": 5, "iron": 6, "grass": 7, "wood": 8, "water": 9, "stone": 10, "gold": 11, "gem": 12 }
num_string_dict = { 3: "w0", 4: "w1", 5: "w2", 6: "iron", 7: "grass", 8: "wood", 9: "water", 10: "stone", 11: "gold", 12: "gem" }		

# --------------------------------------- Main Function ------------------------------------- #


def main():
	demo = pickle.load(open("wood_two_demo.pk", "rb"))
	agent = Agent_Level_4()
	rule_book = RuleBook(agent)
	# Convert into agent readable format
	demo_model = [ fullstate(s) for s in demo ]
	skill_prediction = agent.skill_predict(demo_model)
	event_prediction = rule_book.event_predict(demo_model, skill_prediction)

	import ipdb; ipdb.set_trace()


if __name__ == "__main__":
	main()