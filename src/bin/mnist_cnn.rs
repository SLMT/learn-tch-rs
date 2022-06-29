use tch::{vision::mnist, nn::{VarStore, Module, self, OptimizerConfig}};

fn main() {
    // Setup
    let mnist = mnist::load_dir("data//mnist").unwrap();
    let var_store = VarStore::new(tch::Device::cuda_if_available());
    let network = neural_network(&var_store.root());
    let mut opt = nn::Adam::default().build(&var_store, 1e-4).unwrap();

    // Transfer the test images to the main device
    let test_images = mnist.test_images.to_device(var_store.device());
    let test_labels = mnist.test_labels.to_device(var_store.device());

    // Training
    for epoch in 1..100 {
        // Backprop
        for (image_batch, label_batch) in mnist.train_iter(256).shuffle().to_device(var_store.device()) {
            let loss = network
                .forward(&image_batch)
                .cross_entropy_for_logits(&label_batch);
            opt.backward_step(&loss);
        }

        // Test
        let test_accuracy = network
            .forward(&test_images)
            .accuracy_for_logits(&test_labels);
        println!(
            "epoch: {:4} test acc: {:5.2}%",
            epoch,
            100. * f64::from(&test_accuracy),
        )
    }
}

fn neural_network(vs: &nn::Path) -> impl Module {
    nn::seq()
        .add_fn(|xs| xs.view([-1, 1, 28, 28]))
        .add(nn::conv2d(vs / "c1", 1, 32, 5, Default::default()))
        .add_fn(|xs| xs.max_pool2d_default(2))
        .add(nn::conv2d(vs / "c2", 32, 64, 5, Default::default()))
        .add_fn(|xs| xs.max_pool2d_default(2))
        .add_fn(|xs| xs.flat_view())
        .add(nn::linear(vs / "f1", 1024, 1024, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs / "f2", 1024, 10, Default::default()))
}
