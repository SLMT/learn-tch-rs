
use gym::{GymClient, SpaceTemplate, Environment, SpaceData};
use tch::nn::{self, VarStore, Optimizer, Sequential, Adam, OptimizerConfig, Linear};
use tch::{Device, Tensor};
use tch::Kind::{Float, Int64};

// Iteartions
const NUM_ENVS: i64 = 25;
const NUM_STEPS: i64 = 4;
const MAX_UPDATES: i64 = 10000;
const EVAL_UPDATE: i64 = 100;

// Reward
const REWARD_TARGET: f64 = 450.0;
const REWARD_GAMMA: f64 = 0.99;

// Loss term
const VALUE_LOSS_WEIGHT: f64 = 0.5;
const ENTROPY_WEIGHT: f64 = 0.01;

struct ActorCriticModel {
    shared: Sequential,
    actor: Linear,
    critic: Linear,
}

impl ActorCriticModel {
    fn new(p: &nn::Path, input_size: i64, num_actions: i64) -> ActorCriticModel {
        let shared = nn::seq()
        .add(nn::linear(p / "lin1", input_size, 32, Default::default()))
        .add_fn(|xs| xs.relu());
        let actor = nn::linear(p / "alin", 32, num_actions, Default::default());
        let critic = nn::linear(p / "clin", 32, 1, Default::default());
        ActorCriticModel { shared, actor, critic }
    }

    fn forward(&self, x: &Tensor) -> (Tensor, Tensor) {
        let x = x.apply(&self.shared);
        (x.apply(&self.actor), x.apply(&self.critic))
    }

    fn forward_actor(&self, x: &Tensor) -> Tensor {
        let x = x.apply(&self.shared);
        x.apply(&self.actor)
    }

    fn forward_critic(&self, x: &Tensor) -> Tensor {
        let x = x.apply(&self.shared);
        x.apply(&self.critic)
    }
}

struct Agent {
    vs: VarStore,
    opt: Option<Optimizer>,
    model: Option<ActorCriticModel>,
    input_size: i64
}

impl Agent {
    fn new(input_size: i64, num_actions: i64) -> Agent {
        let mut agent = Agent {
            vs: VarStore::new(Device::cuda_if_available()),
            // vs: VarStore::new(Device::Cpu),
            opt: None,
            model: None,
            input_size
        };
        agent.opt = Some(Adam::default().build(&agent.vs, 1e-2).unwrap());
        agent.model = Some(
            ActorCriticModel::new(&agent.vs.root(), input_size, num_actions)
        );
        agent
    }

    fn act(&self, observations: &Tensor) -> Tensor {
        // Ensure using the same device
        let obs = observations.to_device(self.vs.device());

        // Model inference the action distribution
        let probs = tch::no_grad(
            || self.model.as_ref().unwrap()
                .forward_actor(&obs).softmax(-1, Float)
        );

        // Sample actions
        probs.multinomial(1, true).squeeze_dim(-1)
    }

    fn critic(&self, observations: &Tensor) -> Tensor {
        // Ensure using the same device
        let obs = observations.to_device(self.vs.device());

        // Model inference the critic
        tch::no_grad(
            || self.model.as_ref().unwrap()
                .forward_critic(&obs)
        )
    }

    fn update(&mut self, states: &Tensor, actions: &Tensor, critics: &Tensor,
            batch_size: i64) -> f64 {
        // Ensure using the same device
        let states = states.to_device(self.vs.device());
        let actions = actions.to_device(self.vs.device());
        let critics = critics.to_device(self.vs.device());

        // Flatten the tensors
        let states = states.view([batch_size, self.input_size]);
        let actions = actions.view([batch_size, 1]);
        let critics = critics.view([batch_size, 1]);

        // println!("S: {}", states.to_string(100).unwrap());
        // println!("A: {}", actions.to_string(100).unwrap());
        // println!("C: {}", critics.to_string(100).unwrap());

        // Forward pass
        let (f_acts, f_cris) = self.model.as_ref().unwrap().forward(&states);

        // Value loss
        let advantages = critics - f_cris;
        let value_loss = (&advantages * &advantages).mean(Float);

        // Action loss
        let log_probs = f_acts.log_softmax(-1, Float);
        let action_log_probs = log_probs.gather(1, &actions, false);
        let action_loss = -(&advantages * &action_log_probs).mean(Float);

        // Entropy for exploration
        let probs = f_acts.softmax(-1, Float);
        let entropy = -(log_probs * probs).sum_dim_intlist(&[-1], false, Float).mean(Float);

        // Sum the loss and backward pass
        let loss = value_loss * VALUE_LOSS_WEIGHT +
                action_loss - entropy * ENTROPY_WEIGHT;
        self.opt.as_mut().unwrap().backward_step_clip(&loss, 0.5);

        f64::from(loss)
    }

    fn device(&self) -> Device {
        self.vs.device()
    }
}

fn get_input_size(env: &Environment) -> i64 {
    if let SpaceTemplate::BOX { high: _, low: _, shape } = env.observation_space() {
        shape.iter().product::<usize>() as i64
    } else {
        panic!("the observation space is not what we expected");
    }
}

fn get_action_count(env: &Environment) -> i64 {
    if let SpaceTemplate::DISCRETE { n } = env.action_space() {
        *n as i64
    } else {
        panic!("the action space is not what we expected");
    }
}

fn observation_into_tensor(observation: SpaceData) -> Tensor {
    let obs = observation.get_box().expect("unable to get state");
    let obs = obs.as_slice().unwrap();
    let obs: Vec<f32> = obs.iter().map(|val| *val as f32).collect();
    Tensor::of_slice(&obs)
}

fn action_tensor_to_vector(actions: &Tensor) -> Vec<usize> {
    Vec::<i64>::from(actions).into_iter().map(|a| a as usize).collect()
}

fn main() {
    // Setup Environment
    let gym = GymClient::default();
    let mut envs = Vec::new();
    for _ in 0 .. NUM_ENVS {
        envs.push(gym.make("CartPole-v1"));
    }
    let eval_env = gym.make("CartPole-v1");
    let input_size = get_input_size(&envs[0]);
    let action_count = get_action_count(&envs[0]);
    
    println!("Observation space: {:?}", input_size);
    println!("Action space: {:?}", action_count);
    
    // Setup Agent
    let mut agent = Agent::new(input_size, action_count);

    // Setup batch memory
    let device = agent.device();
    let batch_state = Tensor::zeros(&[NUM_STEPS, NUM_ENVS, input_size], (Float, device));
    let batch_action = Tensor::zeros(&[NUM_STEPS, NUM_ENVS], (Int64, device));
    let batch_critic = Tensor::zeros(&[NUM_STEPS, NUM_ENVS], (Float, device));

    // Get initial observations
    let observations = Tensor::zeros(&[NUM_ENVS, input_size], (Float, device));
    for env_idx in 0 .. NUM_ENVS {
        let obs = envs[env_idx as usize].reset().expect("unable to reset");
        let obs = observation_into_tensor(obs);
        observations.get(env_idx).copy_(&obs);
    }

    // Update iteration
    let mut total_rewards = vec![0.0; NUM_ENVS as usize];
    let mut finished_rewards: Vec<f64> = vec![];
    let mut loss_vals: Vec<f64> = vec![];
    for up_idx in 1..=MAX_UPDATES {

        // Step iteration
        for step_idx in 0 .. NUM_STEPS {
            // Record the observations
            batch_state.get(step_idx).copy_(&observations);

            // Get actions
            let actions = agent.act(&observations);
            let action_ids = action_tensor_to_vector(&actions);

            // React to each environment
            let mut rewards: Vec<f64> = Vec::new();
            let mut reward_masks: Vec<f64> = Vec::new();
            for env_idx in 0 .. NUM_ENVS {
                let action = gym::SpaceData::DISCRETE(action_ids[env_idx as usize]);
                let state = envs[env_idx as usize]
                    .step(&action).expect("unable to take an action");

                // Record reward
                total_rewards[env_idx as usize] += state.reward;
                rewards.push(state.reward);
                reward_masks.push(if state.is_done { 0.0 } else { 1.0 });

                // Get the next observation
                let next_obs = if state.is_done {
                    finished_rewards.push(total_rewards[env_idx as usize]);
                    total_rewards[env_idx as usize] = 0.0;
                    envs[env_idx as usize].reset().expect("unable to reset")
                } else {
                    state.observation
                };
                let next_obs = observation_into_tensor(next_obs);
                observations.get(env_idx).copy_(&next_obs);
            }

            // Compute critics
            let rewards = Tensor::of_slice(&rewards).to_device(device);
            let reward_masks = Tensor::of_slice(&reward_masks).to_device(device);
            let next_state_critics = agent.critic(&observations).view([NUM_ENVS]);
            let critics = rewards + REWARD_GAMMA * reward_masks * next_state_critics;

            // Record the information to the batch
            batch_action.get(step_idx).copy_(&actions);
            batch_critic.get(step_idx).copy_(&critics);
        }

        // Early stop
        if up_idx % EVAL_UPDATE == 0 {
            let avg_rewards = finished_rewards.iter().sum::<f64>() /
                finished_rewards.len() as f64;
            if avg_rewards > REWARD_TARGET {
                println!("The average reward {:.2} reaches the target {:.2} at \
                    iteration {}. Stop here.",
                    avg_rewards,
                    REWARD_TARGET,
                    up_idx
                );
                break;
            }
        }

        // Update the agent
        let loss = agent.update(
            &batch_state,
            &batch_action,
            &batch_critic,
            NUM_ENVS * NUM_STEPS
        );
        loss_vals.push(loss);

        // Evaluation
        if up_idx % EVAL_UPDATE == 0 {
            let avg_rewards = finished_rewards.iter().sum::<f64>() /
                finished_rewards.len() as f64;
            let avg_loss = loss_vals.iter().sum::<f64>() /
                loss_vals.len() as f64;
            println!("Iteration: {}, finished episodes: {}, avg. rewards: {:.2}, \
                avg. loss: {:.2}, evaluation reward: {}",
                up_idx,
                finished_rewards.len(),
                avg_rewards,
                avg_loss,
                evaluate(&agent, &eval_env)
            );
            finished_rewards.clear();
            loss_vals.clear();
        }
    }

    // Final evaluation
    println!("Running the final evaluation...");
    let reward = evaluate(&agent, &eval_env);
    println!("Final reward: {}", reward);
}

fn evaluate(agent: &Agent, env: &Environment) -> f64 {
    let obs = env.reset().expect("unable to reset");
    let mut obs = observation_into_tensor(obs);
    let mut total_reward = 0.0;
    loop {
        // React to the observation
        let actions = agent.act(&obs);
        let action_ids = action_tensor_to_vector(&actions);
        let action = gym::SpaceData::DISCRETE(action_ids[0]);
        let state = env.step(&action).expect("unable to take an action");

        // Update the observation
        obs = observation_into_tensor(state.observation.clone());

        // Record the reward
        total_reward += state.reward;

        // Render
        env.render();

        // Check if it finishes
        if state.is_done {
            break;
        }
    }
    total_reward
}
