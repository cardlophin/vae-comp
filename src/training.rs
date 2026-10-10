use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};

use burn::{
    data::dataloader::DataLoader,
    optim::{AdamConfig, GradientsParams},
    tensor::{Device, Distribution, Tensor, activation},
};

use crate::{
    io::{
        self,
        mnist::{INPUT_DIM, MnistBatch, MnistLoaders},
    },
    models::VariationalModel,
};

pub struct TrainingConfig {
    pub latent_dim: usize,
    pub max_epochs: usize,

    pub initial_lr: f64,
    pub min_lr: f64,
    pub lr_factor: f64,
    pub lr_patience: usize,

    pub early_stop_patience: usize,
    pub min_delta: f64,

    pub beta: f32,
    pub kl_warmup_epochs: usize,

    pub validation_mc_samples: usize,
    pub test_mc_samples: usize,

    pub generation_seed: u64,
}

struct VaeLoss {
    total: Tensor<1>,
    reconstruction: Tensor<1>,
    kl: Tensor<1>,
}

fn vae_loss(
    logits: Tensor<2>,
    target: Tensor<2>,
    mu: Tensor<2>,
    logvar: Tensor<2>,
    beta: f32,
) -> VaeLoss {
    let batch_size = target.dims()[0] as f32;

    let log_p = activation::log_sigmoid(logits.clone());

    let log_one_minus_p = activation::log_sigmoid(logits.neg());

    let reconstruction = (target.clone() * log_p + (target.neg() + 1.0) * log_one_minus_p)
        .neg()
        .sum()
        / batch_size;

    let mu_squared = mu.clone() * mu;

    let kl = (mu_squared + logvar.clone().exp() - logvar - 1.0).sum() * 0.5 / batch_size;

    let total = reconstruction.clone() + kl.clone() * beta;

    VaeLoss {
        total,
        reconstruction,
        kl,
    }
}

#[derive(Clone, Copy)]
struct LossSummary {
    total: f64,
    reconstruction: f64,
    kl: f64,
}

#[derive(Default)]
struct LossAccumulator {
    sums: Option<Tensor<1>>,
    samples: usize,
}

impl LossAccumulator {
    fn update(&mut self, loss: &VaeLoss, batch_size: usize) {
        let reconstruction = loss.reconstruction.clone().without_autodiff();

        let kl = loss.kl.clone().without_autodiff();

        let values = Tensor::cat(vec![reconstruction, kl], 0) * batch_size as f32;

        self.sums = Some(match self.sums.take() {
            Some(previous) => previous + values,
            None => values,
        });

        self.samples += batch_size;
    }

    fn finish(self, beta: f32) -> LossSummary {
        assert!(self.samples > 0, "No se procesaron muestras");

        let values = self
            .sums
            .expect("No hay métricas acumuladas")
            .try_into_vec_as::<f32>()
            .expect("No se pudieron leer las métricas");

        let reconstruction = values[0] as f64 / self.samples as f64;

        let kl = values[1] as f64 / self.samples as f64;

        let total = reconstruction + beta as f64 * kl;

        assert!(
            total.is_finite() && reconstruction.is_finite() && kl.is_finite(),
            "Pérdidas no finitas: \
             total={total}, BCE={reconstruction}, KL={kl}"
        );

        LossSummary {
            total,
            reconstruction,
            kl,
        }
    }
}

fn evaluate<M: VariationalModel>(
    model: &M,
    loader: &dyn DataLoader<MnistBatch>,
    beta: f32,
    mc_samples: usize,
) -> LossSummary {
    assert!(mc_samples > 0);

    let mut metrics = LossAccumulator::default();

    for batch_result in loader.iter() {
        let batch = batch_result.expect("Error cargando evaluación");

        let images = batch.images;
        let batch_size = images.dims()[0];

        for _ in 0..mc_samples {
            let (logits, mu, logvar) = model.forward_logits(images.clone());

            let loss = vae_loss(logits, images.clone(), mu, logvar, beta);

            metrics.update(&loss, batch_size);
        }
    }

    metrics.finish(beta)
}

fn beta_for_epoch(epoch: usize, config: &TrainingConfig) -> f32 {
    if config.kl_warmup_epochs == 0 {
        config.beta
    } else {
        config.beta * (epoch as f32 / config.kl_warmup_epochs as f32).min(1.0)
    }
}

pub fn train<M: VariationalModel>(
    mut model: M,
    train_device: &Device,
    eval_device: &Device,
    loaders: &MnistLoaders,
    config: &TrainingConfig,
    output_dir: &Path,
) {
    assert!(config.max_epochs > 0);
    assert!(config.latent_dim > 0);
    assert!(config.initial_lr >= config.min_lr);
    assert!(config.min_lr > 0.0);
    assert!(config.lr_factor > 0.0 && config.lr_factor < 1.0);
    assert!(config.lr_patience > 0);
    assert!(config.early_stop_patience > 0);
    assert!(config.min_delta >= 0.0);
    assert!(config.beta >= 0.0);
    assert!(config.kl_warmup_epochs <= config.max_epochs);

    std::fs::create_dir_all(output_dir).expect("No se pudo crear el directorio de resultados");

    let mut optimizer = AdamConfig::new().init();
    let mut learning_rate = config.initial_lr;

    let history_file =
        File::create(output_dir.join("history.csv")).expect("No se pudo crear el historial");

    let mut history = BufWriter::new(history_file);

    writeln!(
        history,
        "epoch,lr,beta,train_total,train_bce,train_kl,\
         val_total,val_bce,val_kl,train_seconds,epoch_seconds"
    )
    .expect("Error escribiendo la cabecera");

    let mut best_model: Option<M> = None;
    let mut best_loss = f64::INFINITY;
    let mut best_epoch = 0;

    let mut control_best = f64::INFINITY;
    let mut bad_epochs = 0;
    let mut lr_bad_epochs = 0;

    println!(
        "Train={}, val={}, test={}",
        loaders.train_size, loaders.validation_size, loaders.test_size,
    );

    for epoch in 1..=config.max_epochs {
        train_device
            .sync()
            .expect("Error sincronizando el dispositivo");

        let epoch_start = Instant::now();
        let train_start = Instant::now();

        let epoch_lr = learning_rate;
        let beta = beta_for_epoch(epoch, config);
        let mut metrics = LossAccumulator::default();

        for batch_result in loaders.train.iter() {
            let batch = batch_result.expect("Error cargando train");

            let images = batch.images;
            let batch_size = images.dims()[0];

            let (logits, mu, logvar) = model.forward_logits(images.clone());

            let loss = vae_loss(logits, images, mu, logvar, beta);

            metrics.update(&loss, batch_size);

            let gradients = loss.total.backward();

            let gradients = GradientsParams::from_grads(gradients, &model);

            model = optimizer.step(epoch_lr, model, gradients);
        }

        let train_metrics = metrics.finish(beta);

        train_device.sync().expect("Error sincronizando train");

        let train_seconds = train_start.elapsed().as_secs_f64();

        let evaluation_model = model.valid();

        let validation_metrics = evaluate(
            &evaluation_model,
            loaders.validation.as_ref(),
            config.beta,
            config.validation_mc_samples,
        );

        eval_device.sync().expect("Error sincronizando validación");

        let can_select = epoch >= config.kl_warmup_epochs;

        if can_select && validation_metrics.total < best_loss {
            best_loss = validation_metrics.total;
            best_epoch = epoch;
            best_model = Some(evaluation_model);
        }

        let epoch_seconds = epoch_start.elapsed().as_secs_f64();

        println!(
            "Epoch {epoch:03}/{} | \
             lr={epoch_lr:.6} beta={beta:.2} | \
             train={:.4} BCE={:.4} KL={:.4} | \
             val={:.4} BCE={:.4} KL={:.4} | \
             train={train_seconds:.2}s \
             total={epoch_seconds:.2}s",
            config.max_epochs,
            train_metrics.total,
            train_metrics.reconstruction,
            train_metrics.kl,
            validation_metrics.total,
            validation_metrics.reconstruction,
            validation_metrics.kl,
        );

        writeln!(
            history,
            "{epoch},{epoch_lr:.8},{beta:.4},\
             {:.6},{:.6},{:.6},\
             {:.6},{:.6},{:.6},\
             {train_seconds:.4},{epoch_seconds:.4}",
            train_metrics.total,
            train_metrics.reconstruction,
            train_metrics.kl,
            validation_metrics.total,
            validation_metrics.reconstruction,
            validation_metrics.kl,
        )
        .expect("Error escribiendo el historial");

        history.flush().expect("Error guardando el historial");

        if !can_select {
            continue;
        }

        if validation_metrics.total < control_best - config.min_delta {
            control_best = validation_metrics.total;
            bad_epochs = 0;
            lr_bad_epochs = 0;
        } else {
            bad_epochs += 1;
            lr_bad_epochs += 1;

            if lr_bad_epochs >= config.lr_patience {
                let next_lr = (learning_rate * config.lr_factor).max(config.min_lr);

                if next_lr < learning_rate {
                    println!(
                        "Reduciendo lr: \
                         {learning_rate:.6} -> {next_lr:.6}"
                    );
                }

                learning_rate = next_lr;
                lr_bad_epochs = 0;
            }
        }

        if bad_epochs >= config.early_stop_patience {
            println!(
                "Parada temprana: {bad_epochs} épocas \
                 sin mejora significativa"
            );
            break;
        }
    }

    let best_model = best_model.expect("No se seleccionó ningún modelo");

    println!(
        "Mejor época: {best_epoch}; \
         val={best_loss:.4}"
    );

    model
        .save_file(output_dir.join("last.bpk"))
        .expect("Error guardando el último modelo");

    best_model
        .clone()
        .save_file(output_dir.join("best.bpk"))
        .expect("Error guardando el mejor modelo");

    // Test únicamente después de seleccionar el modelo.
    let test_metrics = evaluate(
        &best_model,
        loaders.test.as_ref(),
        config.beta,
        config.test_mc_samples,
    );

    println!(
        "Test | total={:.4} BCE={:.4} KL={:.4}",
        test_metrics.total, test_metrics.reconstruction, test_metrics.kl,
    );

    std::fs::write(
        output_dir.join("test.csv"),
        format!(
            "best_epoch,total,bce,kl\n\
             {best_epoch},{:.6},{:.6},{:.6}\n",
            test_metrics.total, test_metrics.reconstruction, test_metrics.kl,
        ),
    )
    .expect("Error guardando las métricas de test");

    eval_device.seed(config.generation_seed);

    let z: Tensor<2> = Tensor::random(
        [64, config.latent_dim],
        Distribution::Normal(0.0, 1.0),
        eval_device,
    );

    let generated = best_model.decode(z);
    let first = generated.clone().slice([0..1, 0..INPUT_DIM]);

    io::save_mnist_grid(generated, 8, 4, output_dir.join("generated-grid.png"))
        .expect("Error guardando la cuadrícula");

    io::save_mnist_grid(first, 1, 8, output_dir.join("generated-single.png"))
        .expect("Error guardando el dígito");

    println!("Resultados guardados en {}", output_dir.display());
}
