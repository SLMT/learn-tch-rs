
use gym::{GymClient, SpaceTemplate, Environment};
use tch::nn::{self, VarStore, Optimizer, Module, Sequential, Adam, OptimizerConfig};
use tch::{Device, Tensor};
use tch::Kind::Float;

const TOTAL_EPISODE: usize = 10000;
const UPDATE_EPISODE: usize = 100;
const EVAL_EPISODE: usize = 500;

struct Agent {
    vs: VarStore,
    opt: Option<Optimizer>,
    model: Option<Sequential>,
    action_count: i64,
}

impl Agent {
    fn new(input_size: i64, action_count: i64) -> Agent {
        let mut agent = Agent {
            vs: VarStore::new(Device::cuda_if_available()),
            // vs: VarStore::new(Device::Cpu),
            opt: None,
            model: None,
            action_count
        };
        agent.opt = Some(Adam::default().build(&agent.vs, 1e-2).unwrap());
        agent.model = Some(
            build_model(&agent.vs.root(), input_size, action_count)
        );

        agent
    }

    fn act(&self, observation: &Tensor) -> usize {
        // Model inference the action distribution
        let probs = tch::no_grad(||
            self.model.as_ref().unwrap().forward(observation).softmax(0, Float)
        );
        
        // Sample an action
        let action: i64 = probs.multinomial(1, true).into();
        action as usize
    }

    fn device(&self) -> Device {
        self.vs.device()
    }

    fn update(&mut self, steps: Vec<Step>) {
        // Prepare data as tensors
        let step_count = steps.len() as i64;
        let actions: Vec<i64> = steps.iter()
            .map(|s| s.action as i64).collect();
        let actions = Tensor::of_slice(&actions)
            .to_device(self.device()).unsqueeze(1);
        let action_masks = Tensor::zeros(
            &[step_count, self.action_count],
            (Float, self.device())
        ).scatter_value(1, &actions, 1.0);
        let rewards: Vec<f64> = steps.iter()
            .map(|s| s.reward).collect();
        let rewards = Tensor::of_slice(&rewards)
            .to_device(self.device()).to_kind(Float);
        // Note that this code comsumes 'steps' since we use 'into_iter()'
        // so it must be put at the last line.
        let observations: Vec<Tensor> = steps.into_iter()
            .map(|s| s.observation).collect();
        let observations = Tensor::stack(&observations, 0);

        // Calculate loss
        let logits = observations.apply(self.model.as_ref().unwrap());
        let log_probs = action_masks * logits.log_softmax(1, Float);
        let log_probs = log_probs.sum_dim_intlist(&[1], false, Float);
        let loss = -(rewards * log_probs).mean(Float);

        // Backward pass
        self.opt.as_mut().unwrap().backward_step(&loss);

    }
}

#[derive(Debug)]
struct Step {
    observation: Tensor,
    reward: f64,
    action: usize
}

fn build_model(p: &nn::Path, input_size: i64, action_count: i64) -> Sequential {
    nn::seq()
        .add(nn::linear(p / "lin1", input_size, 32, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(p / "lin2", 32, action_count, Default::default()))
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

fn observation_into_tensor(observation: &[f64], device: Device) -> Tensor {
    let obs: Vec<f32> = observation.iter().map(|val| *val as f32).collect();
    Tensor::of_slice(&obs).to_device(device)
}

fn accmulate_rewards(trajectory: &mut [Step]) {
    let mut acc_reward: f64 = 0.0;
    for step in trajectory.iter_mut().rev() {
        acc_reward += step.reward;
        step.reward = acc_reward;
    }
}

fn main() {
    // Setup Environment
    let gym = GymClient::default();
    let env = gym.make("CartPole-v1");
    let input_size = get_input_size(&env);
    let action_count = get_action_count(&env);
    
    println!("Observation space: {:?}", input_size);
    println!("Action space: {:?}", action_count);

    // Setup Agent
    let mut agent = Agent::new(input_size, action_count);

    // Episode iteration
    let mut reward_sum = 0.0;
    let mut best_reward = 0.0;
    let mut steps = Vec::new();
    for episode in 1..=TOTAL_EPISODE {
        // Reset the environment
        let obs = env.reset().expect("unable to reset");
        let obs = obs.get_box().expect("unable to get state");

        // Step iteration
        let mut trajectory = Vec::new();
        loop {
            // React to the observation
            let obs = observation_into_tensor(obs.as_slice().unwrap(), agent.device());
            let action_id = agent.act(&obs);
            let action = gym::SpaceData::DISCRETE(action_id);
            let state = env.step(&action).expect("unable to take an action");

            // Record this step
            trajectory.push(Step {
                observation: obs,
                action: action_id,
                reward: state.reward
            });

            // Render
            if episode % EVAL_EPISODE == 0 {
                env.render();
            }

            // Check if it finishes
            if state.is_done {
                break;
            }
        }

        // Record the result
        let total_reward: f64 = trajectory.iter().map(|s| s.reward).sum();
        reward_sum += total_reward;
        if total_reward > best_reward {
            best_reward = total_reward;
        }

        // Record this trajectory
        accmulate_rewards(&mut trajectory);
        steps.append(&mut trajectory);

        // Print the result
        if episode % EVAL_EPISODE == 0 {
            let avg_reward = reward_sum / EVAL_EPISODE as f64;
            println!("Episode {} finished. Avg. reward: {}, best reward: {}.",
                episode, avg_reward, best_reward);
            reward_sum = 0.0;
            best_reward = 0.0;
        }

        // Update the agent's model
        if episode % UPDATE_EPISODE == 0 {
            agent.update(steps);
            steps = Vec::new();
        }
    }
}
