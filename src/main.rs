mod io;
mod models;
mod training;

use std::path::Path;

use burn::tensor::Device;

use io::mnist::{INPUT_DIM, MnistLoaderConfig, build_loaders};

use models::{cnnvae::CNNVAE, vae::VAE};

use training::{TrainingConfig, train};

const SEED: u64 = 42;

const HIDDEN_DIM: usize = 512;
const LATENT_DIM: usize = 16;

fn main() {
    let architecture = std::env::args().nth(1).unwrap_or_else(|| "vae".to_owned());

    if matches!(architecture.as_str(), "-h" | "--help") {
        println!("Uso:");
        println!("  cargo run --release -- vae");
        println!("  cargo run --release -- cnn");
        return;
    }

    if !matches!(architecture.as_str(), "vae" | "cnn") {
        eprintln!("Arquitectura desconocida: {architecture}");
        eprintln!("Opciones válidas: vae, cnn");
        std::process::exit(2);
    }

    let train_device = Device::metal(Default::default()).autodiff();

    let eval_device = train_device.clone().without_autodiff();

    let loader_config = MnistLoaderConfig {
        batch_size: 256,
        num_workers: 2,
        validation_size: 6_000,
        seed: SEED,
    };

    let training_config = TrainingConfig {
        latent_dim: LATENT_DIM,
        max_epochs: 100,

        initial_lr: 1e-3,
        min_lr: 1e-5,
        lr_factor: 0.5,
        lr_patience: 5,

        early_stop_patience: 15,
        min_delta: 0.05,

        beta: 1.0,
        kl_warmup_epochs: 5,

        validation_mc_samples: 3,
        test_mc_samples: 5,

        generation_seed: SEED + 1,
    };

    let loaders = build_loaders(&loader_config, &train_device, &eval_device);

    // Semilla antes de inicializar los pesos.
    train_device.seed(SEED);

    match architecture.as_str() {
        "vae" => {
            println!(
                "VAE denso: input={INPUT_DIM}, \
                 hidden={HIDDEN_DIM}, latent={LATENT_DIM}"
            );

            let model = VAE::new(INPUT_DIM, HIDDEN_DIM, LATENT_DIM, &train_device);

            train(
                model,
                &train_device,
                &eval_device,
                &loaders,
                &training_config,
                Path::new("artifacts/vae"),
            );
        }

        "cnn" => {
            println!(
                "CNN-VAE: canales=[32, 64], \
                 latent={LATENT_DIM}"
            );

            let model = CNNVAE::new(LATENT_DIM, &train_device);

            train(
                model,
                &train_device,
                &eval_device,
                &loaders,
                &training_config,
                Path::new("artifacts/cnn"),
            );
        }

        _ => unreachable!(),
    }
}
