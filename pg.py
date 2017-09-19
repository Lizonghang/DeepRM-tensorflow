from network import Network
import environment
import parameters
import job_distribution

"""
Problem: Fall into the local minimum
"""

params = parameters.Parameters()
params.compute_dependent_parameters()

learner = Network(
    params.network_input_height,
    params.network_input_width,
    params.network_output_dim,
    params.lr,
    params.discount,
    params.decay,
    params.epsilon
)

for dynamic_env_time in range(50):

    nw_len_seqs, nw_size_seqs = job_distribution.generate_sequence_work(params)

    env = environment.Env(
        params,
        nw_len_seqs=nw_len_seqs,
        nw_size_seqs=nw_size_seqs,
        render=False,
        repre='image',
        end='no_new_job'
    )

    env_max_rewards_sum = -100

    for i_episode in range(1000):
        obs = []
        actions = []
        rewards = []

        env.reset()

        observation = env.observe()

        for iter in range(200):

            action = learner.choose_action(observation)

            observation_, reward, done, info = env.step(action)

            obs.append(observation)
            actions.append(action)
            rewards.append(reward)

            if done:

                rewards_sum = sum(rewards)

                if rewards_sum > env_max_rewards_sum:  env_max_rewards_sum = rewards_sum

                print("training on env:", dynamic_env_time, " episode:", i_episode, " env max reward sum:", env_max_rewards_sum, "  reward sum:", rewards_sum)

                learner.learn(obs, actions, rewards)

                break

            observation = observation_
