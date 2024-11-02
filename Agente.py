import gymnasium as gym
from agent import Agente
import numpy as np

if _name_ == '_main_':
    env = gym.make('LunarLander-v2', render_mode='human')
    env = gym.make('LunarLander-v2')
    agent = Agent(
        gamma=0.99,          
        epsilon=1.0, 
        lr=0.0001,             
        input_dims=[8], 
        batch_size=64,       
        n_actions=4,
        max_mem_size=100000, 
        eps_end=0.01, 
        eps_dec=1e-4,        
        target_update=100    
    )

    agent.load_checkpoint()

    scores, eps_history = [], []
    n_games = 500000
    biggest_score = 0
    
    for i in range(n_games):
        score = 0
        
        done = False
        observation = env.reset()[0]  
        observation = np.array(observation, dtype=np.float32)  
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            observation_ = np.array(observation_, dtype=np.float32) 
            done = terminated or truncated
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
            

        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])
        if scores[i] >= biggest_score:
                biggest_score = scores[i]

        print(f'episode {i}, score {score:.2f}, average score {avg_score:.2f}, epsilon {agent.epsilon:.2f}, best score: {biggest_score:.2f}')

        if i % 100 == 0:
            agent.save_checkpoint()
