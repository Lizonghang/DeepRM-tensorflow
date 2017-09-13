from network import Network
import numpy as np
import job_distribution
import environment
import time


def get_entropy(act_prob):
    entropy = -np.sum(act_prob * np.log(act_prob))
    if np.isnan(entropy):
        entropy = 0
    return entropy


def get_traj(agent, env, episode_max_length):
    """
    Run agent-environment loop for one whole episode (trajectory)
    Return dictionary of results
    """
    env.reset()

    obs = []
    actions = []
    rewards = []
    entropy = []
    info = []

    ob = env.observe()

    for _ in xrange(episode_max_length):

        action, act_prob = agent.choose_action(ob)

        ob_, reward, done, info = env.step(action, repeat=True)

        obs.append(ob)
        actions.append(action)
        rewards.append(reward)
        entropy.append(get_entropy(act_prob))

        if done:  break

        ob = ob_

    return {
        'obs': np.array(obs),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'entropy': entropy,
        'info': info
    }


def concatenate_all_ob(trajs, params):
    timesteps_total = 0
    for i in xrange(len(trajs)):
        timesteps_total += len(trajs[i]['rewards'])

    all_ob = np.zeros((timesteps_total, 1, params.network_input_height, params.network_input_width), dtype=np.float32)
    timesteps = 0
    for i in xrange(len(trajs)):
        for j in xrange(len(trajs[i]['rewards'])):
            all_ob[timesteps, 0, :, :] = trajs[i]['obs'][j]
    return all_ob


def calc_discount_sum(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    assert x.ndim >= 1
    out = np.zeros(len(x))
    out[-1] = x[-1]
    for i in reversed(xrange(len(x) - 1)):
        out[i] = x[i] + gamma * out[i + 1]
    return out


def process_all_info(trajs):
    enter_time = []
    finish_time = []
    job_len = []

    for traj in trajs:
        enter_time.append(np.array([traj['info'].record[i].enter_time for i in xrange(len(traj['info'].record))]))
        finish_time.append(np.array([traj['info'].record[i].finish_time for i in xrange(len(traj['info'].record))]))
        job_len.append(np.array([traj['info'].record[i].len for i in xrange(len(traj['info'].record))]))

    enter_time = np.concatenate(enter_time)
    finish_time = np.concatenate(finish_time)
    job_len = np.concatenate(job_len)
    return enter_time, finish_time, job_len


def get_traj_worker(pg_learner, env, params):
    trajs = []
    for i in xrange(params.num_seq_per_batch):
        traj = get_traj(pg_learner, env, params.episode_max_length)
        trajs.append(traj)
    all_obs = concatenate_all_ob(trajs, params)

    # Compute discounted sums of rewards
    rets = [calc_discount_sum(traj['rewards'], params.discount) for traj in trajs]
    maxlen = max(len(ret) for ret in rets)
    padded_rets = [np.concatenate([ret, np.zeros(maxlen - len(ret))]) for ret in rets]

    # Compute time-dependent baseline
    baseline = np.mean(padded_rets, axis=0)

    # Compute advantage function
    advs = [ret - baseline[:len(ret)] for ret in rets]
    all_advs = np.concatenate(advs)
    all_actions = np.concatenate([traj['actions'] for traj in trajs])

    all_eprews = np.array([calc_discount_sum(traj["rewards"], params.discount)[0] for traj in trajs])  # episode total rewards
    all_eplens = np.array([len(traj["rewards"]) for traj in trajs])  # episode lengths

    # All Job Stat
    enter_time, finish_time, job_len = process_all_info(trajs)
    finished_idx = (finish_time >= 0)
    all_slowdown = (finish_time[finished_idx] - enter_time[finished_idx]) / job_len[finished_idx]

    all_entropy = np.concatenate([traj["entropy"] for traj in trajs])

    return {
        'all_obs': all_obs,
        'all_actions': all_actions,
        'all_advs': all_advs,
        'all_eprews': all_eprews,
        'all_eplens': all_eplens,
        'all_slowdown': all_slowdown,
        'all_entropy': all_entropy
    }


def concatenate_all_ob_across_examples(all_obs, params):
    num_ex = len(all_obs)
    total_samp = 0
    for i in xrange(num_ex):
        total_samp += all_obs[i].shape[0]

    all_obs_contact = np.zeros((total_samp, 1, params.network_input_height, params.network_input_width), dtype=np.float32)

    total_samp = 0
    for i in xrange(num_ex):
        prev_samp = total_samp
        total_samp += all_obs[i].shape[0]
        all_obs_contact[prev_samp: total_samp, :, :, :] = all_obs[i]
    return all_obs_contact


def launch(params, render=False, repre='image', end='no_new_job'):
    print "Preparing for envs and workers."

    pg_learners = []
    envs = []

    nw_len_seqs, nw_size_seqs = job_distribution.generate_sequence_work(params, seed=42)

    for ex in xrange(params.num_ex):
        env = environment.Env(params, nw_len_seqs=nw_len_seqs, nw_size_seqs=nw_size_seqs, render=False, repre=repre, end=end)
        env.seq_no = ex
        envs.append(env)

    for ex in xrange(params.batch_size + 1):  # last worker for updating the parameters
        pg_learner = Network(params.network_input_height,
                             params.network_input_width,
                             params.network_output_dim,
                             params.lr,
                             params.reward_decay,
                             params.epsilon)
        pg_learners.append(pg_learner)

    print "Start training."

    timer_start = time.time()

    for iter in xrange(1, params.num_epochs):

        learner_and_env_idxs = []
        result = []

        ex_indices = range(params.num_ex)
        np.random.shuffle(ex_indices)

        all_eprews = []
        grad_and_vars_all = []
        eprews = []
        eplens = []
        all_slowdown = []
        all_entropy = []

        ex_counter = 0
        for ex in xrange(params.num_ex):
            ex_idx = ex_indices[ex]
            learner_and_env_idxs.append([ex_counter, ex_idx])
            ex_counter += 1

            if ex_counter > params.batch_size or ex == params.num_ex - 1:
                ex_counter = 0

                for idx in learner_and_env_idxs:
                    result.append(get_traj_worker(pg_learners[idx[0]], envs[idx[1]], params))

                learner_and_env_idxs = []  # clear

                all_obs = concatenate_all_ob_across_examples([r["all_obs"] for r in result], params)
                all_actions = np.concatenate([r["all_actions"] for r in result])
                all_advs = np.concatenate([r["all_advs"] for r in result])

                # Do policy gradient update step, using the first agent
                # put the new parameter in the last 'worker', then propagate the update at the end
                learner = pg_learners[params.batch_size]
                grad_and_vars = learner.get_gradient(all_obs, all_actions, all_advs)
                grad_and_vars_all.append(grad_and_vars)

                all_eprews.extend([r["all_eprews"] for r in result])

                eprews.extend(np.concatenate([r["all_eprews"] for r in result]))  # episode total rewards
                eplens.extend(np.concatenate([r["all_eplens"] for r in result]))  # episode lengths

                all_slowdown.extend(np.concatenate([r["all_slowdown"] for r in result]))
                all_entropy.extend(np.concatenate([r["all_entropy"] for r in result]))

        # assemble gradients
        grad_and_vars = grad_and_vars_all[0]
        for i in xrange(1, len(grad_and_vars_all)):
            for j in xrange(len(grad_and_vars)):
                grad_and_vars[j] += grad_and_vars_all[i][j]

        # propagate network parameters to others
        learner = pg_learners[params.batch_size]
        learner.apply_gradient(grad_and_vars)

        net_params = learner.get_params()
        for i in xrange(params.batch_size):
            pg_learners[i].set_params(net_params)

        timer_end = time.time()

        print "-----------------"
        print "Iteration: \t %i" % iter
        # print "NumTrajs: \t %i" % len(eprews)
        # print "NumTimesteps: \t %i" % np.sum(eplens)
        print "MaxReward: \t %s" % np.average([np.max(rew) for rew in all_eprews])
        print "MeanReward: \t %s +- %s" % (np.mean(eprews), np.std(eprews))
        print "MeanSlowdown: \t %s" % np.mean(all_slowdown)
        # print "MeanLen: \t %s +- %s" % (np.mean(eplens), np.std(eplens))
        # print "MeanEntropy \t %s" % (np.mean(all_entropy))
        # print "Elapsed time\t %s" % (timer_end - timer_start), "seconds"
        print "-----------------"
