from network import Network
import environment
import parameters
import job_distribution

params = parameters.Parameters()
params.compute_dependent_parameters()

learner = Network(
    params.network_input_height,
    params.network_input_width,
    params.network_output_dim,
    params.lr,
    params.reward_decay,
    params.epsilon
)
max_reward = -100
for dynamic_env_time in range(50):

    nw_len_seqs, nw_size_seqs = job_distribution.generate_sequence_work(params, seed=42)

    env = environment.Env(
        params,
        nw_len_seqs=nw_len_seqs,
        nw_size_seqs=nw_size_seqs,
        render=False,
        repre='image',
        end='no_new_job'
    )

    for i_episode in range(10000):
        obs = []
        actions = []
        rewards = []

        env.reset()

        observation = env.observe()

        for iter in range(2000):

            action = learner.choose_action(observation)

            observation_, reward, done, info = env.step(action)

            obs.append(observation)
            actions.append(action)
            rewards.append(reward)

            if done:

                rewards_sum = sum(rewards)

                running_reward = running_reward * 0.9 + rewards_sum * 0.1 if 'running_reward' in globals() else rewards_sum

                # print("training on env:", dynamic_env_time, " episode:", i_episode, "  reward:", int(running_reward))
                if running_reward > max_reward:  max_reward = running_reward
                # learner.learn(obs, actions, rewards)

                break

            observation = observation_

    print max_reward
