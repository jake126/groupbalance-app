import random
import math as math

from org.sortition.groupselect.allocator.TAAllocator import TAAllocator

class TAAllocationsManager:
    def __init__(self, ctx):
        self.ctx = ctx

    def run(self, progress_bar=None):
        app_data = self.ctx.app_data

        
        tables = app_data.settings['tables']
        cluster_tables = app_data.settings['cluster_tables']
        m_data = app_data.m_data
        # seats calculable from tables and data size:
        seats = math.ceil(m_data/tables)
        val_cluster = app_data.settings['val_cluster']
        
        
        if(tables*seats < len(app_data.peopledata_vals)):
            raise Exception("Error: Not enough space!", "There's not enough space! Please increase the number of groups or number of people per group!")

        if cluster_tables>=tables:
            raise Exception("Error: More tables assigned for clustering than there are total tables! Please either reduce cluster tables or increase total tables")
        
        # advanced settings (or defaults if not set)
        pareto_prob = app_data.settings['pareto_prob']
        seed = app_data.settings['seed']

        try:
            random.seed(seed)
        except:
            raise Exception("Error: Random seed incorrect!", "There was a problem setting the random seed. Please check your input!")

        nallocations = app_data.settings['nallocations']
        
        if (nallocations < 1):
            raise Exception("Error: Wrong allocation number!", "The number of computed allocations must at least be 1!")

        peopledata_vals_used = [{} for i in range(app_data.m_data)]

        for i in range(app_data.m_data):
            for j in app_data.order_cluster+app_data.order_diverse:
                peopledata_vals_used[i][j] = self.ctx.app_data_manager.load_details(i, j)

        
        order_cluster_dict = self.ctx.app_data_manager.get_fields_cluster_dict()
        order_diverse_dict = self.ctx.app_data_manager.get_fields_diverse_dict()

        #if not order_diverse_dict:
        #    raise Exception("Error: One diversification field required!", "You have to set at least one field that is used to diversify people across groups.")
            
        if len(order_cluster_dict)>1:
            raise Exception("Error: Only one cluster field permitted. Please reduce the number of cluster fields.")
        
        # how many agents have cluster value?
        if len(order_cluster_dict)==1:
            cluster_key = next(iter(order_cluster_dict))
            no_cluster_agents = sum(1 for person in peopledata_vals_used if person[cluster_key]==val_cluster)
            # what is the minimum number of tables required for clustering?
            min_cluster_tables = math.ceil(no_cluster_agents/seats)
            # warning if insufficient space for clustering individuals
            if cluster_tables<min_cluster_tables:
                raise Exception("Error: There is not enough space on clustering tables to fit the "+str(no_cluster_agents)+" clustered individuals. Please increase Clustered Groups (recommended minimum: "+str(min_cluster_tables+1)+")")
            if cluster_tables == min_cluster_tables != 0:
                # print a warning message with output
                self.ctx.app_data.settings["cluster_tables_required"] = min_cluster_tables
            else:
                # no additional warning message - reset to default
                self.ctx.app_data.settings["cluster_tables_required"] = 0 
                
        # how many iterations of pareto swaps do we want?        
        n_swap_loops = int(app_data.settings['swap_rounds'])
        if n_swap_loops<1:
            raise Exception("Error: at least one round of meeting optimization must be specified (in *advanced settings*)")
        
        allocator = TAAllocator(tables, seats, peopledata_vals_used, order_cluster_dict, order_diverse_dict,m_data,nallocations,cluster_tables,val_cluster,pareto_prob,n_swap_loops,progress_bar)

        manuals = app_data.manuals
        
        if any([len([m[0] for m in manuals if m[1] == t]) > seats for t in range(tables)]):
            raise Exception("Error: Too many manuals!", "You allocated too many people manually to one group.")
        allocator.set_manually(manuals)
        
        
        n_results = allocator.run(progress_bar)
        
        allocations = []
        for result in n_results[0]:
            allocations.append(n_results[0][result])

        allocation_group_outcome = allocations
        
        # calculate minimum possible repeated meetings between two rounds (for logic of this calculation, see https://hackmd.io/oZ5MdQ9oR7KeajwNTA4vWw)
        # note that this may be an underestimate with clustering
        d_mult = m_data // (tables**2)
        L_R = ((tables**2) * 0.5 * d_mult * (d_mult-1)) + d_mult * (m_data % (tables**2))
        min_duplicates = max(0, L_R)
        # what is the optimal number of new pairs generated in the first round?
        optimal_pairs = 0
        for table in allocations[0]:
            n = len(table)
            optimal_pairs+=n*(n-1)//2
        
        total_possible_pairs = 0
        for round_no in range(nallocations):
            if round_no==0:
                # no restrictions on repeating pairs
                total_possible_pairs+=optimal_pairs
            else:
                total_possible_pairs+=optimal_pairs-min_duplicates
        # calculate total pairs in sample
        total_pairs = m_data*(m_data-1)//2
        
        if 0 in n_results[1][nallocations-1]:
            allocation_group_links_pp = (total_pairs - n_results[2][nallocations-1][0])/m_data
        else:
            # all pairs have met
            allocation_group_links_pp = total_pairs
            
        # maximum links from round 0 to 1 are a function of table size and number of tables
        allocation_group_links_pp_max = min(total_pairs,total_possible_pairs)/m_data

        

        self.ctx.app_data.results = allocation_group_outcome

        self.links = allocation_group_links_pp
        self.links_rel = allocation_group_links_pp/allocation_group_links_pp_max
