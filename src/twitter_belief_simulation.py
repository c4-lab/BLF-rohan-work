from dataclasses import dataclass
import pandas as pd
import numpy as np
from numpy.linalg import norm
from scipy.stats import truncnorm
import typing
import time
from math import floor, exp
from collections import OrderedDict
import json
import random


@dataclass
class Environment:
    ht_creation_rate: int  #Number of hashtags created each timestep
    belief_dimension: int  #Underlying size of the vector space
    ht_recency_window: int  #Length of time since last mention before HT is removed from population
    rt_recency_window: int  #Length of time since last tweet may be retweeted
    init_agent_population: int #Number of agents
    init_ht_population: int #Number of hashtags 
    agent_turnover: float  #[0..1] Likelihood any given agent will be recycled in a population
    post_probability: typing.Tuple[float,float] #[mu,sigma] parameters for truncated normal distribution governing post prob
    agent_belief_swap: float #[0..1]  Probability agent will adopt the anchor belief of someone they retweet
    rt_probability: typing.Tuple[float,float] #[mu,sigma] parameters for truncated normal distribution governing rt prob
    integration_latitude: typing.Tuple[float,float] #[mu,sigma] parameters for trunc norm distribution governing diversity tolerance
    min_match: float #Minimum match for HT to be considered
    ht_count: typing.Tuple[int,float,int,int,float] #[mu,sigma_pop,low,high,sigma_ind] number of hashtags per tweet
    num_clusters: int # Number of belief clusters
    cluster_dispersion: float #Std deviation of noisy clusters for initialization
    max_time: 1000

    
    def initialize(self):
        self.clusters = [_generate_belief(self.belief_dimension) for _ in range(self.num_clusters)]
        self.sim_time = 0
        self.feed = []
        self.hts =  self._init_hts()
        self.agents = [Agent(self) for _ in range(self.init_agent_population)]
        self.original_posts = 0
        self.retweet_posts = 0
        self.belief_df = pd.DataFrame(columns=["Agent","Latitude","InitBelief","Time","Belief","HTMatch","RTMatch","Distance"])
        self.post_df = pd.DataFrame(columns = ["PostId","Agent","Time","Hashtags","RetweetId","RetweetAuthor"])
        self.feed_indices = []


    def _init_hts(self) -> OrderedDict:
        pool = OrderedDict()
        for i in range(self.init_ht_population):
            ht = Hashtag(self)
            pool[ht] = ht
        return pool
        
    def sample_noisy_cluster(self,integration_latitude):
        m = 1-((integration_latitude+1) / 2)
        ix = random.choice(range(len(self.clusters)))
        return ix,_noisy_centroid(self.clusters[ix],self.cluster_dispersion*m)
    
    def sample_post_prob(self):
        return _truncnorm(self.post_probability[0],self.post_probability[1])
    
    def sample_rt_prob(self):
        return _truncnorm(self.rt_probability[0],self.rt_probability[1])

    def sample_integration_latitude(self):
        return _truncnorm(self.integration_latitude[0],self.integration_latitude[1],low=-1,high=1)

    #TODO Fix this so that we have an integral number of hashtags to post
    def sample_ht_count(self):
        return _truncnorm(self.ht_count[0],self.ht_count[1],self.ht_count[2],self.ht_count[3])

    def get_ht_std(self):
        return self.ht_count[4]

    def tick(self) -> bool:
        self.prune()
        self.sim_time+=1
        data = []
        for a in self.agents:
            dict = {
                "Agent":a.id,
                "Latitude":a.integration_latitude,
                "InitBelief":a.initial_belief_id,
                "Time":self.sim_time,
                "Belief":a.current_belief,
                "HTMatch":a.ht_match_score,
                "RTMatch":a.rt_match_score,
                "Distance":a.distance

            }
            data.append(dict)
        self.belief_df = pd.concat([self.belief_df,pd.DataFrame(data,columns=["Agent","Latitude","InitBelief","Time","Belief","HTMatch","RTMatch","Distance"])],ignore_index=True,sort=False)
        return self.sim_time <= self.max_time

    def make_post(self,p:'Post'):
        self.feed.append(p)
        df = self.post_df
        df.loc[len(df)] = [p.id,p.author.id,self.sim_time,[h.id for h in p.hashtags],
            p.post.id if p.is_retweet() else None, p.post.author.id if p.is_retweet() else None]
        if p.is_retweet():
            self.retweet_posts+=1
        else:
            self.original_posts+=1

    def get_hashtag_pool(self):
        return list(self.hts.keys())
    
    def get_feed(self):
        return list(self.feed)

    
    
    def use_hashtag(self,ht:'Hashtag'):
        self.hts.move_to_end(ht)
    
    def prune(self):
        
        while (self.feed and self.sim_time-self.feed[0].post_time > self.rt_recency_window):
            self.feed.pop(0)

        x = next(iter(self.hts.keys()))
        while self.hts and self.sim_time-x.use_time > self.ht_recency_window:
            self.hts.popitem(0)
            x = next(iter(self.hts.keys()))
        

    def expand_ht_pool(self):
        for i in range(self.ht_creation_rate):
            h = Hashtag(self)
            self.hts[h]=h

    def recycle_agents(self):
        it_agents = list(self.agents)
        for a in it_agents:
            if _roll(self.agent_turnover):
                self.agents.remove(a)
                self.agents.append(Agent(self)) 

    def inspect(self):
        print(f"{self.sim_time} steps")
        print(f"Original posts: {self.original_posts} Retweet posts: {self.retweet_posts}")
        print(f"Pool size = {len(self.hts)}")
        print(f"Feed size = {len(self.feed)}")
        print(f"Num agents = {len(self.agents)}")

    def save(self):
        self.belief_df.to_csv("simulation_beliefs.csv")
        self.post_df.to_csv("simulation_posts.csv")
        
             
class Agent:

    def __init__(
        self,
        env: Environment,
    ):
        self.id = _generate_guid("Agent")
        self.integration_latitude = env.sample_integration_latitude()
        self.initial_belief_id,self.current_belief = env.sample_noisy_cluster(self.integration_latitude)
        self.post_prob = env.sample_post_prob()
        self.rt_prob = env.sample_rt_prob()
       
        self.ht_count = env.sample_ht_count()
        self.environment = env
        self.distance = 0
        self.ht_match_score = 0
        self.rt_match_score = 0
    
    def post(self):

        all_hts = []
        if _roll(self.post_prob):
            # TODO:  Need to fix this up to make sure we get a decent integer
            num_hts = floor(_truncnorm(self.ht_count,self.environment.get_ht_std(),
                self.environment.ht_count[2],self.environment.ht_count[3]))
            hts = self.environment.get_hashtag_pool()
            ht_match = [self._calculate_match(ht.belief) for ht in hts]
            self.ht_match_score = np.mean(ht_match)
            content = self._generate_candidates(hts,ht_match,num_hts)
            if len(content) > 0:
                self.environment.make_post(Post(self.environment,self,hashtags = content)) 
                all_hts = all_hts + content.tolist()
        
        if _roll(self.rt_prob):
            rts = self.environment.get_feed()
            rt_match = [self._calculate_match(*[h.belief for h in rt.hashtags]) for rt in rts]
            self.rt_match_score = np.mean(rt_match)
            content = self._generate_candidates(rts,rt_match)
            if len(content) > 0:
                self.environment.make_post(Post(self.environment,self,post = content[0]))
                all_hts = all_hts + content[0].hashtags.tolist()
                if _roll(self.environment.agent_belief_swap):
                    self.initial_belief_id = content[0].author.initial_belief_id

        if len(all_hts) > 0:
            n_belief = np.mean(np.array([ht.belief for ht in all_hts]),axis=0)
            n_belief = np.mean(np.array([self.environment.clusters[self.initial_belief_id],self.current_belief,n_belief]),axis=0)
            self.distance = 1 - _cosine_similarity(self.current_belief,n_belief)
            self.current_belief = n_belief
           

    def _generate_candidates(self,candidates:typing.List, matches:typing.List, num: int = 1):
        pairs = list(filter(lambda x: x[1] > self.environment.min_match,zip(candidates,matches)))
        #print(f"Latitude {self.integration_latitude} Filtered {len(candidates)-len(pairs)}")
        if pairs:
            c,p = list(zip(*pairs))
            p = np.array(p)/sum(p)
            return np.random.choice(c,min(num,len(c)),replace=False,p=p)
        else:
            return []

    def _calculate_match(self,*args):
        sim = np.mean([_cosine_similarity(self.current_belief,x) for x in args])
        return _power_scale(sim,self.integration_latitude)

        
class Hashtag:

    def __init__(
        self,
        environment: Environment
    ):
        self.id = _generate_guid("HT")
        self.environment = environment
        self.use_time = environment.sim_time
        self.belief = _generate_belief(environment.belief_dimension)
    
    def __repr__(self):
        return self.id

class Post:

    def __init__(
        self,
        environment: Environment,
        author: Agent,
        hashtags: typing.List[Hashtag] = None,
        post: 'Post' = None
    ):
        self.id = _generate_guid("POST")
        self.environment = environment
        self.author = author
        self.post = post
        self.post_time = environment.sim_time
        if self.post:
            self.hashtags = post.hashtags.copy()
        elif hashtags is not None:
            self.hashtags = hashtags.copy()
        else:
            raise ValueError("post and hashtags parameters cannot both be empty")

        for h in self.hashtags:
            environment.use_hashtag(h)
    
    def is_retweet(self):
        return self.post is not None

    def __repr__(self):
        return f"{self.author.id}:{self.hashtags}:{self.post.author.id if self.is_retweet() else '[orig]'}"
    


guid_map = None    

def run_simulation(config_file:str = None):
    global guid_map
    if not config_file:
        config_file = "./sim_config.json"
    with open(config_file) as f:
        config = json.load(f)
    guid_map = {}
    env = Environment(**config)
    env.initialize()
    print(f"Will run {env.max_time} steps")
    try:
        while (env.tick()):
            
            for a in env.agents:
                a.post()
            env.recycle_agents()
            env.expand_ht_pool()
            
            if (env.sim_time % 10 == 0):
                print("#",end="")
            else:
                print(".",end="")
    except KeyboardInterrupt:
        print("Gracefully exiting...")
    finally:
        env.save()


            


def _power_scale(x:float,rho:float):
    return x**exp(rho)

def _cosine_similarity(a,b):
    return(np.dot(a, b)/(norm(a)*norm(b)))

def _roll(probability:float):
    return np.random.rand() <= probability

def _truncnorm(mu,sig,low=0,high=1):
        a, b = (low - mu) / sig, (high - mu) / sig
        return truncnorm.rvs(a,b,loc = mu, scale=sig)

def _noisy_centroid(v:np.array, std:float):
    return np.array([_truncnorm(x,std) for x in v])

def _generate_guid(label:str ):
    if label not in guid_map:
        guid_map[label] = 0
    next = guid_map[label]
    guid_map[label]+=1
    return f"{label}:{next}"

def _generate_belief(nbits:int):
    return(np.random.rand(nbits))

