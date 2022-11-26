import numpy as np
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
from env import Env
from torch.distributions import Categorical
import torch


def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

# agent_num = [2, 4, 8, 16, 32]
# action_space = [15, 31, 63, 127, 255, 511, 1023] = 7
# observation_space = cw, 전송성공수, 전송실패수 = 3
# state_space = observation_space * agent_num
# reward = 전송성공비율 * fairness (미정)

def simulation(agent_num):
    # env = Env(agent_num)
    actor_dims = [4 for _ in range(agent_num)]
    critic_dims = sum(actor_dims)

    action_space = 7

    maddpg_agents = MADDPG(actor_dims, critic_dims, agent_num, action_space,
                           fc1=256, fc2=256,
                           alpha=0.00025, beta=0.0005, scenario="test",
                           chkpt_dir='/Users/mobicom/maddpg/')
    memory = MultiAgentReplayBuffer(100000, critic_dims, actor_dims,
                                    action_space, agent_num, batch_size=512)

    PRINT_INTERVAL = 10
    EPISODE_NUM = 100000
    MAX_TIME = 60000000 # 300000000us
    # MAX_TIME = 1000000 # 300000000us
    DONE = [False]*agent_num
    total_steps = 1
    score_history = []
    best_score = 0

    cw_arr = [15, 31, 63, 127, 255, 511, 1023]
    epi_th = []
    epi_fai = []

    epi_ratio = []

    for epi_num in range(EPISODE_NUM):
        score = 0
        env = Env(agent_num)
        step_th = []
        step_fai = []
        # 첫 observation 설정
        station_actions = [0 for _ in range(agent_num)]
        station_backoff = np.array([np.random.randint(16) for i in range(agent_num)])

        obs, reward, _, step_time, add_packet, step_success = env.csma_ca(station_actions, station_backoff, agent_num)
        total_success = step_success
        total_packet = add_packet

        epi_time = 50
        # epi 시작
        while epi_time <= MAX_TIME:
            # print(epi_time)
            # acgent action 결정
            actions_prob = maddpg_agents.choose_action(obs)
            for i in range(agent_num):
                prob = torch.tensor(actions_prob[i])
                agent_step_action = Categorical(prob).sample().item()
                station_actions[i] = agent_step_action
                station_backoff[i] = np.random.randint(cw_arr[station_actions[i]] + 1)

            # step
            obs_, reward, throughput, step_time, add_packet, step_success = env.csma_ca(station_actions, station_backoff, agent_num)

            # step 결과 저장
            step_th.append(throughput)

            total_success += step_success
            total_packet += add_packet

            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

            memory.store_transition(obs, state, actions_prob, reward, obs_, state_, DONE)

            # print(total_steps)
            if total_steps % 30 == 0:
                maddpg_agents.learn(memory)

            obs = obs_

            score += sum(reward)
            total_steps += 1
            epi_time += step_time

        # print(sa)
        total_packet -= add_packet
        epi_ratio.append(np.round(total_success/total_packet, 5))
        score_history.append(np.round(score, 2))
        avg_score = np.mean(score_history[-10:])
        avg_score *= 1.0
        if avg_score > best_score:
            maddpg_agents.save_checkpoint()
            best_score = avg_score

        epi_th.append(np.round(np.mean(step_th), 3))
        # epi_fai.append(np.round(np.mean(step_fai[-10:]), 4))

        # if avg_score > best_score:
            # maddpg_agents.save_checkpoint()
            # best_score = avg_score
        if epi_num % PRINT_INTERVAL == 0 and epi_num > 0:
            # print(epi_th)
            print(epi_ratio)
            print(score_history)
            print('episode', epi_num, 'average score {:.1f}'.format(avg_score))
            print(epi_ratio[-1])
            # print(epi_ratio[-1])
            print('-----------')


    print(epi_th)
    print(score_history)
    return epi_th, score_history


if __name__ == '__main__':
    agent_nums = [i+1 for i in range(100)]

    th_array = []
    re_array = []

    for agent_num in agent_nums:
        th, reward = simulation(agent_num)

        th_array.append(th)
        re_array.append(reward)

    print(th_array)
    print(re_array)