# libraries (all Python in-built, so no install required)
import random
import math as math
from itertools import product

import copy

# class for all the allocations stuff
class TAAllocator:
    def __init__(self, tables, seats, people, cats_cluster, cats_diverse, m_data, nallocations, cluster_tables, val_cluster,pareto_prob,n_swap_loops,progress_bar):
        self.pareto_prob = pareto_prob
        self.tables = tables
        self.cluster_tables = cluster_tables
        self.seats = seats
        self.people = people
        self.cats_cluster = cats_cluster
        self.val_cluster = val_cluster
        self.cats_diverse = cats_diverse
        self.m_data = m_data
        self.nallocations = nallocations
        self.n_swap_loops = n_swap_loops
        self.manual_pids = []
        # remove empty seats if assymetrical tables
        self.no_larger_tables = self.m_data%self.tables
        self.no_smaller_tables = self.tables - self.no_larger_tables
        if self.no_larger_tables == 0:
            self.template = [[None for s in range(self.seats)] for r in range(self.no_smaller_tables)]
        else:
            self.template = [[None for s in range(self.seats)] for r in range(self.no_larger_tables)]+[[None for s in range(self.seats-1)] for r in range(self.no_smaller_tables)]
        
        # What are the ideal demographic proportions on a perfectly balanced table for each demographic?
        self.ideal_balance = self.calculate_ideal_balance()
    
    # place manual PIDs
    def set_manually(self, manuals):
        self.manual_pids = [m[0] for m in manuals]
        for pid, t in manuals:
            ss = min(s for s in range(self.seats) if not self.template[t][s])
            self.template[t][ss] = pid

    def run(self,progress_bar):
        n_rounds = self.nallocations
        # save total new meetings, meeting distributions (before and after swaps), average table balance per demog (all before and after running swaps)
        pre_meeting_dist = {}
        post_meeting_dist = {}
        new_meetings_in_round = {}
        pre_balance = {}
        post_balance = {}

        allocations_list = {}

        # initialise previous meetings
        self.previous_meetings = {}
        for i in range(self.m_data):
            for j in range(i + 1, self.m_data):
                pair = (i, j)
                # Initialize the pair in the dictionary if not already present
                if pair not in self.previous_meetings:
                    self.previous_meetings[pair] = 0

        for round_no in range(n_rounds):
            if progress_bar: progress_bar.setValue(round_no+1)
            meetings_previous_round = self.previous_meetings.copy()
            # run the algorithm for a single round of meetings
            round_assign_pre, round_assign_swap, meetings_pre = self.run_round(self.template,self.n_swap_loops)

            # save meetings information
            pre_occurences = {}
            for value in meetings_pre.values():
                pre_occurences[value] = pre_occurences.get(value, 0) + 1
            pre_meeting_dist[round_no] = pre_occurences
            occurences = {}
            for value in self.previous_meetings.values():
                occurences[value] = occurences.get(value, 0) + 1
            post_meeting_dist[round_no] = occurences

            new_meetings = {}
            for pair in self.previous_meetings:
                if self.previous_meetings[pair]-meetings_previous_round[pair] == 1:
                    new_meetings[pair] = self.previous_meetings[pair]
            round_meetings = {}
            for value in new_meetings.values():
                round_meetings[value] = round_meetings.get(value, 0)+1
            new_meetings_in_round[round_no] = round_meetings

            # save demographic information
            pre_demog_evaluations = {}
            for index, table in enumerate(round_assign_pre):
                pre_demog_evaluations[index] = {}
                pre_demog_evaluations[index] = self.evaluate_demographics(
                    round_assign_pre, index)[0]
            pre_balance[round_no] = self.averages_from_evals(
                pre_demog_evaluations)
            post_demog_evaluations = {}
            for index, table in enumerate(round_assign_swap):
                post_demog_evaluations[index] = {}
                post_demog_evaluations[index] = self.evaluate_demographics(
                    round_assign_swap, index)[0]
            post_balance[round_no] = self.averages_from_evals(
                post_demog_evaluations)

            allocations_list[round_no] = round_assign_swap

        return allocations_list, pre_meeting_dist, post_meeting_dist, new_meetings_in_round, pre_balance, post_balance

    # what should the ideal balance of demographics be on a table, given the panel-wide distribution?
    def calculate_ideal_balance(self):
        ideal_balance = {}
        for demog in self.cats_diverse:
            counts = [0] * len(self.cats_diverse[demog])
            for row in self.people:
                for i, category in enumerate(self.cats_diverse[demog]):
                    if row[demog] == category:
                        counts[i] += 1
            ideal_balance[demog] = [count / self.m_data for count in counts]
        return ideal_balance
    
    # result aggregation helper function
    def averages_from_evals(self,evaluations):
        values_per_key = {}
        # Iterate through the nested dictionaries
        for nested_dict in evaluations.values():
            # Iterate through the key-value pairs in each nested dictionary
            for key, value in nested_dict.items():
                # Append the value to the list corresponding to the key
                if key not in values_per_key:
                    values_per_key[key] = []
                values_per_key[key].append(value)
        averages = {}
        for key in values_per_key:
            lst = values_per_key[key]
            averages[key] = sum(lst)/len(lst)
        return averages

    def run_round(self, template, n_swap_loops):
        self.allocations = copy.deepcopy(template)
        # manual pids are already fixed in allocations; randomise the remaining pids
        all_pids = list(range(self.m_data))
        shuffled_pids = all_pids.copy()
        random.shuffle(shuffled_pids)
        # remove any with manual placement
        shuffled_pids = [x for x in shuffled_pids if x not in self.manual_pids]
        # calculate number of clustered tables
        cluster_table_index = list(range(self.cluster_tables))
        # only need clustering if there is a clustering variable selected
        if len(self.cats_cluster)==1:
            cluster_individuals = []
            for index, person in enumerate(self.people):
                if person[next(iter(self.cats_cluster))] == self.val_cluster:
                    cluster_individuals.append(index)
            cluster_individuals = [x for x in cluster_individuals if x not in self.manual_pids]
            # assign clustered individuals to a cluster subset of tables, ensuring each clustering table
            chosen_chair = 0
            # is there enough space on the table for the clustered individuals and manual allocations?
            total_clustering_spaces = sum(self.allocations[index].count(None) for index in cluster_table_index)
            if len(cluster_individuals)>total_clustering_spaces:
                raise ValueError("Too many manual allocations to clustering tables: please reduce manual allocations.")
            for agent in cluster_individuals:
                agent_assigned = 0
                while(agent_assigned == 0):
                    table_no = chosen_chair % len(cluster_table_index)
                    seat_no = math.floor(
                        chosen_chair/len(cluster_table_index) % self.seats)
                    if self.allocations[table_no][seat_no] is None:
                        self.allocations[table_no][seat_no] = agent
                        agent_assigned = 1
                    chosen_chair += 1
        else:
            cluster_individuals = []
        # if not a cluster individual, allocate to the first blank space
        non_cluster_individuals = [x for x in shuffled_pids if x not in cluster_individuals]
        chosen_chair = 0
        for agent in non_cluster_individuals:
            agent_assigned = 0
            while(agent_assigned == 0):
                table_no = chosen_chair % self.tables
                seat_no = math.floor(chosen_chair/self.tables % self.seats)
                if self.allocations[table_no][seat_no] is None:
                    self.allocations[table_no][seat_no] = agent
                    agent_assigned = 1
                chosen_chair += 1

        # search for all pareto improvements - iterate through process n_swap_loops times
        if n_swap_loops == 1:
            pareto_allocations = self.pareto_swaps(shuffled_pids, cluster_individuals, cluster_table_index, self.allocations)
        else:
            pareto_allocations = self.pareto_swaps(shuffled_pids, cluster_individuals, cluster_table_index, self.allocations)
            for swap_round in range(1,n_swap_loops):
                pareto_allocations = self.pareto_swaps(shuffled_pids, cluster_individuals, cluster_table_index, pareto_allocations)
            
        raw_meetings = self.previous_meetings.copy()
        # update previous meetings
        for sublist in pareto_allocations:
            for i in range(len(sublist)):
                for j in range(i + 1, len(sublist)):
                    pair = (min(sublist[i], sublist[j]),
                            max(sublist[i], sublist[j]))
                    # Increment count for the pair in the dictionary
                    self.previous_meetings[pair] += 1
                    
        # also count how many meetings there would have been without swaps
        for sublist in self.allocations:
            for i in range(len(sublist)):
                for j in range(i + 1, len(sublist)):
                    pair = (min(sublist[i], sublist[j]),
                            max(sublist[i], sublist[j]))
                    # Increment count for the pair in the dictionary
                    raw_meetings[pair] += 1

        return self.allocations, pareto_allocations, raw_meetings

    # evaluation of pareto swaps
    def pareto_swaps(self, shuffled_pids, cluster_individuals, cluster_table_index, temp_allocations):
        temp_allocations_update = temp_allocations.copy()
        # input: current allocations; agent properties (global); previous meetings (global)
        # output: allocations with each pareto swap performed
        # evaluate current meetings and current demographics for all tables
        table_meeting_evaluations = {}
        table_demog_evaluations = {}
        for index, table in enumerate(temp_allocations_update):
            table_meeting_evaluations[index] = self.evaluate_meetings(table)
            table_demog_evaluations[index] = {}
            table_demog_evaluations[index] = self.evaluate_demographics(
                temp_allocations_update, index)

        # evaluate each potential swap
        for pid in shuffled_pids:
            # which table contains this agent?
            for index, table in enumerate(temp_allocations_update):
                if pid in table:
                    table_no = index
            # ascertain all potential swaps that are a pareto improvement and adhere to clustering
            pid_info = {key: self.people[pid][key]
                        for key in self.people[pid] if key in self.cats_diverse}
            candidate_demogs = {}
            # possible swaps for someone on this table with this profile:
            for demog in self.cats_diverse:
                # anyone with these demographics represents a pareto improvement
                candidate_demogs[demog] = table_demog_evaluations[table_no][1][demog][pid_info[demog]]
                # if the current value is not in the candidate swaps, then it does not represent a pareto improvement, but it is acceptable to allow for others
            candidate_profiles = self.generate_combinations(
                candidate_demogs, pid_info)
            candidate_swaps = {}
            for profile in candidate_profiles:
                if pid in cluster_individuals:
                    # return all pids on clustering tables (except table_no) with the profile, IF the swap is a pareto improvement for the other table too; return pareto score
                    candidate_swap_tables = [x for x in table_demog_evaluations if (
                        x != table_no) and (x in cluster_table_index)]
                else:
                    # return all pids on any table (except table_no) with the profile, IF the swap is a pareto improvement for the other table too
                    # only tables where the swap would be a pareto improvement
                    candidate_swap_tables = [
                        x for x in table_demog_evaluations if x != table_no]
                for candidate_table in candidate_swap_tables:
                    pareto_score = 0
                    pareto_profile = table_demog_evaluations[candidate_table][1]
                    table_valid = True
                    for index, demog in enumerate(pareto_profile):
                        if pid_info[demog] in pareto_profile[demog][profile[index]]:
                            pareto_score += 1
                        # if not a pareto improvement, or the same, then table not valid
                        elif pid_info[demog] != profile[index]:
                            table_valid = False
                            break
                    if table_valid:
                        # do any table members match this profile?
                        if pid in cluster_individuals:
                            for swap_pid in temp_allocations_update[candidate_table]:
                                if swap_pid not in self.manual_pids:
                                    if tuple(self.people[swap_pid][key] for key in self.people[swap_pid] if key in self.cats_diverse) == profile:
                                        candidate_swaps[swap_pid] = pareto_score + \
                                            candidate_profiles[profile]
                        else:
                            for swap_pid in temp_allocations_update[candidate_table]:
                                if swap_pid not in cluster_individuals:
                                    if swap_pid not in self.manual_pids:
                                        if tuple(self.people[swap_pid][key] for key in self.people[swap_pid] if key in self.cats_diverse) == profile:
                                            candidate_swaps[swap_pid] = pareto_score + \
                                                candidate_profiles[profile]
            # if a pareto improvement is not possible, do not perform a swap
            if len(candidate_swaps) == 0:
                continue
            # what effect will each swap have on meetings?
            candidate_meetings = {}
            for swap in candidate_swaps:
                candidate_meetings[swap] = self.evaluate_swap(
                    pid, swap, temp_allocations_update, table_meeting_evaluations)
            # CALCULATE AND PERFORM OPTIMAL SWAP
            # filter to swaps that either improve meeting scores or make a pareto improvement
            candidate_swaps = {key: value for key, value in candidate_swaps.items() if (
                candidate_swaps[key] > 0) or (candidate_swaps[key] == 0 and candidate_meetings[key] > 0)}
            if len(candidate_swaps) == 0:
                continue
            # for each level of pareto improvement, what is the best improvement in meetings?
            distinct_candidates = {}
            for distinct_value in {value for value in candidate_swaps.values()}:
                distinct_keys = {
                    key for key, value in candidate_swaps.items() if value == distinct_value}
                max_meetings = max(
                    value for key, value in candidate_meetings.items() if key in distinct_keys)
                distinct_candidates.update({key: value for key, value in candidate_swaps.items(
                ) if (value == distinct_value) and (candidate_meetings[key] == max_meetings)})
            distinct_meetings = {key: value for key, value in candidate_meetings.items(
            ) if key in distinct_candidates}
            # if there are ties, randomly keep one
            reverse_mapping = {}
            for key, value in distinct_candidates.items():
                if value not in reverse_mapping:
                    reverse_mapping[value] = []
                reverse_mapping[value].append(key)
            # keep one random key for each unique value
            final_candidates = {}
            for value, keys in reverse_mapping.items():
                final_candidates[random.choice(keys)] = value
            final_meetings = {
                key: value for key, value in distinct_meetings.items() if key in final_candidates}

            # if any swaps are dominated by another swap, remove them
            keys_to_remove = set()
            # Iterate over the keys of both dictionaries
            for key in final_meetings.keys():
                # Check if there exists another key with a larger value in both dictionaries
                if any(final_meetings[other_key] >= final_meetings[key] and final_candidates[other_key] > final_candidates[key] for other_key in final_meetings.keys() if other_key != key):
                    keys_to_remove.add(key)
            # Remove keys from both dictionaries
            for key in keys_to_remove:
                del final_meetings[key]
                del final_candidates[key]

            # Perform probabilistic swaps
            final_swap = self.select_key(final_candidates, final_meetings)
            if final_swap == None:
                continue

            # update tables with final_swap id
            for index, table in enumerate(temp_allocations_update):
                if final_swap in table:
                    swap_table = index
            temp_allocations_update[table_no] = [
                final_swap if x == pid else x for x in temp_allocations_update[table_no]]
            temp_allocations_update[swap_table] = [
                pid if x == final_swap else x for x in temp_allocations_update[swap_table]]

            # only re-evaluate tables where a swap has taken place
            for index in [table_no, swap_table]:
                table_meeting_evaluations[index] = self.evaluate_meetings(
                    temp_allocations_update[index])
                table_demog_evaluations[index] = {}
                table_demog_evaluations[index] = self.evaluate_demographics(
                    temp_allocations_update, index)

            # move on to next PID and loop through

        return temp_allocations_update

    # how to evaluate swaps given pareto scores and meeting scores
    def select_key(self,pareto, meet):
        pareto_copy = pareto.copy()
        meet_copy = meet.copy()
        total_a = sum(pareto_copy.values())
        # Randomly choose between distribution and meetings
        if random.random() < self.pareto_prob:
            if len(pareto_copy) == 1:
                return next(iter(pareto_copy.keys()))
            # calculate cumulative probabilities
            cumulative_prob_a = {}
            cumulative_sum = 0
            for key, value in pareto_copy.items():
                cumulative_sum += value / total_a
                cumulative_prob_a[key] = cumulative_sum
            # Choose key proportional to number of pareto improvements
            rand_num = random.random()
            for key, prob in cumulative_prob_a.items():
                if rand_num <= prob:
                    return key
        else:
            # Choose key proportional to meeting reductions
            meet_copy = {key:value for key,value in meet_copy.items() if meet_copy[key]>=0}
            if len(meet_copy)==0:
                return None
            if len(meet_copy) == 1:
                return next(iter(meet_copy.keys()))
            # calculate cumulative probabilities
            total_b = sum(meet_copy.values())
            cumulative_prob_b = {}
            cumulative_sum = 0
            for key, value in meet_copy.items():
                cumulative_sum += value / total_b
                cumulative_prob_b[key] = cumulative_sum
            rand_num = random.random()
            # do not select a swap that worsens meetings
            for key, prob in cumulative_prob_b.items():
                if rand_num <= prob:
                    return key
                
    # evaluate the quality of a single swap
    def evaluate_swap(self, original_id, swap_id, allocations, table_meeting_evaluations):
        for index, table in enumerate(allocations):
            if swap_id in table:
                swap_table = index
            if original_id in table:
                table_no = index
        # original meeting score: max of sum of table lengths, scaling up large multiple meetings (commented option) or linear meetings
        #original_meetings = sum(2**x for x in table_meeting_evaluations[table_no].values())+sum(2**x for x in table_meeting_evaluations[swap_table].values())
        original_meetings = sum(x for x in table_meeting_evaluations[table_no].values())+sum(x for x in table_meeting_evaluations[swap_table].values())
        original_table = allocations[table_no]
        swap_table = allocations[swap_table]
        original_table_2 = [swap_id if x == original_id else x for x in original_table]
        swap_table_2 = [original_id if x == swap_id else x for x in swap_table]
        meetings_1 = self.evaluate_meetings(original_table_2)
        meetings_2 = self.evaluate_meetings(swap_table_2)
        # new meeting score
        #new_meetings = sum(2**x for x in meetings_1.values())+sum(2**x for x in meetings_2.values())
        new_meetings = sum(x for x in meetings_1.values()) + sum(x for x in meetings_2.values())
        # how has score improved? UPDATED to "how has total meeting number improved, removing geometric focus
        return original_meetings-new_meetings

    # helper function
    def generate_combinations(self, demogs, info):
        # Get the keys (demographics) from the dictionaries
        demographics = list(demogs.keys())
        # Initialize the dictionary to store the combinations and their counts
        combinations_count = {}
        # Generate all possible combinations of values
        for values in product(*[demogs[demographic]+[info[demographic]] for demographic in demographics]):
            combination = tuple(values)
            count = len(demogs) - \
                sum(1 for v in combination if v in info.values())
            combinations_count[combination] = count
        return combinations_count

    # function to evaluate current table based on demographic balance and previous meetings
    def evaluate_meetings(self, table):
        # how many times have individuals met other individuals on the table?
        total_meetings = {}
        for i in range(len(table)):
            for j in range(i + 1, len(table)):
                agent1, agent2 = min(table[i], table[j]), max(
                    table[i], table[j])
                # Sum the values from pairs_dict for the pair of agents
                total_meetings[agent1] = total_meetings.get(
                    agent1, 0) + self.previous_meetings.get((agent1, agent2), 0)
                total_meetings[agent2] = total_meetings.get(
                    agent2, 0) + self.previous_meetings.get((agent1, agent2), 0)
        return(total_meetings)

    # function to evaluate demographic balance on table j compared to overall demographics, and calculate pareto improvement actions
    def evaluate_demographics(self, temp_allocations, table_no):
        table = temp_allocations[table_no]
        # what should the demographic balance be for this table, in a perfect scenario? Saved in ideal_balance
        table_data = {}
        for index in table:
            table_data[index] = self.people[index]
        table_balance = {}
        # for each demographic, what changes would be pareto acceptable?
        table_actions = {}
        table_distances = {}
        table_length = len(table)
        for demog in self.cats_diverse:
            counts = [0] * len(self.cats_diverse[demog])
            for person in table_data.values():
                for i, category in enumerate(self.cats_diverse[demog]):
                    if person.get(demog) == category:
                        counts[i] += 1
            table_balance[demog] = [count / table_length for count in counts]
            
            table_distances[demog] = sum([abs(x-y) for x, y in zip(
                self.ideal_balance[demog], table_balance[demog])])/len(self.ideal_balance[demog])
            # acceptable pareto improvements for a table: (a) if a reduction/increase by one still rounds to balance;
            # (b) if a reduction/increase by one moves the table closer to the ideal balance
            table_actions[demog] = self.evaluate_actions(
                self.ideal_balance[demog], table_balance[demog], self.cats_diverse[demog], len(table))
        return table_distances, table_actions

    # assess pareto improvement outcomes on a table
    def evaluate_actions(self, ideal_dist, table_dist, cat_labels, table_size):
        # what can we sub out an agent with value X for to lead to a pareto improvement?
        table_discrepancies = [y-x for y, x in zip(table_dist, ideal_dist)]
        actions = {}
        # what pareto improvement actions can be taken given the ideal split and the table split?
        for index, label in enumerate(cat_labels):
            actions_for_label = []
            if table_dist[index] > ideal_dist[index]:
                # there are too many of this label on the table - a reduction is always a pareto improvement
                # what can we increase to fund this?
                for a, b in zip(table_discrepancies, cat_labels):
                    if a < 0:
                        actions_for_label.append(b)
            actions[label] = actions_for_label
        return actions
