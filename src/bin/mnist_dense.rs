use tch::{vision::mnist, nn::{VarStore, Module, self, OptimizerConfig}};

const IMAGE_DIM: i64 = 784;
const HIDDEN_SIZE: i64 = 128;
const LABELS: i64 = 10;

fn main() {
    // Setup
    let mnist = mnist::load_dir("data//mnist").unwrap();
    let var_store = VarStore::new(tch::Device::Cpu);
    let network = neural_network(&var_store.root());
    let mut opt = nn::Adam::default().build(&var_store, 1e-3).unwrap();

    // Training
    for epoch in 1..200 {
        // Backprop
        let loss = network
            .forward(&mnist.train_images)
            .cross_entropy_for_logits(&mnist.train_labels);
        opt.backward_step(&loss);

        // Test
        let test_accuracy = network
            .forward(&mnist.test_images)
            .accuracy_for_logits(&mnist.test_labels);
        println!(
            "epoch: {:4} train loss: {:8.5} test acc: {:5.2}%",
            epoch,
            f64::from(&loss),
            100. * f64::from(&test_accuracy),
        )
    }
}

fn neural_network(vs: &nn::Path) -> impl Module {
    nn::seq()
        .add(
            nn::linear(
                vs / "layer1", 
                IMAGE_DIM, 
                HIDDEN_SIZE, 
                Default::default()
            )
        )
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs, HIDDEN_SIZE, LABELS, Default::default()))
}
