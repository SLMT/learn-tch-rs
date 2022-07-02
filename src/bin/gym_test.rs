use std::io::{self, Write};

use gym::GymClient;

fn main() {
    let gym = GymClient::default();
    let env = gym.make("CartPole-v1");

    for episode in 0..10 {
        env.reset().expect("unable to reset");

        print!("Simulating episode {} ...", episode);
        io::stdout().flush().unwrap();

        for _ in 0..100 {
            let action = env.action_space().sample();
            let state = env.step(&action).expect("unable to take an action");
            env.render();
            if state.is_done {
                break;
            }
        }

        println!("finished.")
    }
}
