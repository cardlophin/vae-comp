use burn::{
    module::Module,
    nn::{
        Linear, LinearConfig, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
    },
    tensor::{Device, Distribution, Tensor, activation},
};

const IMAGE_SIZE: usize = 28;
const INPUT_DIM: usize = IMAGE_SIZE * IMAGE_SIZE;

const CHANNELS_1: usize = 32;
const CHANNELS_2: usize = 64;
const FEATURE_SIZE: usize = 7;

const FLAT_FEATURES: usize = CHANNELS_2 * FEATURE_SIZE * FEATURE_SIZE;

#[derive(Module, Debug)]
pub struct CNNVAE {
    // Encoder
    enc_conv1: Conv2d,
    enc_conv2: Conv2d,

    fc_mu: Linear,
    fc_logvar: Linear,

    // Decoder
    dec_linear: Linear,
    dec_conv1: ConvTranspose2d,
    dec_conv2: ConvTranspose2d,
}

impl CNNVAE {
    pub fn new(d_latent: usize, device: &Device) -> Self {
        assert!(d_latent > 0);

        Self {
            enc_conv1: Conv2dConfig::new([1, CHANNELS_1], [4, 4])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),

            enc_conv2: Conv2dConfig::new([CHANNELS_1, CHANNELS_2], [4, 4])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),

            fc_mu: LinearConfig::new(FLAT_FEATURES, d_latent).init(device),

            fc_logvar: LinearConfig::new(FLAT_FEATURES, d_latent).init(device),

            dec_linear: LinearConfig::new(d_latent, FLAT_FEATURES).init(device),

            dec_conv1: ConvTranspose2dConfig::new([CHANNELS_2, CHANNELS_1], [4, 4])
                .with_stride([2, 2])
                .with_padding([1, 1])
                .init(device),

            dec_conv2: ConvTranspose2dConfig::new([CHANNELS_1, 1], [4, 4])
                .with_stride([2, 2])
                .with_padding([1, 1])
                .init(device),
        }
    }

    pub fn encode(&self, x: Tensor<2>) -> (Tensor<2>, Tensor<2>) {
        let [batch_size, input_dim] = x.dims();

        assert_eq!(
            input_dim, INPUT_DIM,
            "El modelo espera imágenes MNIST de 784 píxeles",
        );

        let x: Tensor<4> = x.reshape([batch_size, 1, IMAGE_SIZE, IMAGE_SIZE]);

        let h = activation::relu(self.enc_conv1.forward(x));

        let h = activation::relu(self.enc_conv2.forward(h));

        let h: Tensor<2> = h.reshape([batch_size, FLAT_FEATURES]);

        // Sin activación en las cabezas gaussianas.
        let mu = self.fc_mu.forward(h.clone());
        let logvar = self.fc_logvar.forward(h);

        (mu, logvar)
    }

    fn reparametrize(&self, mu: Tensor<2>, logvar: Tensor<2>) -> Tensor<2> {
        let std = (logvar * 0.5).exp();

        let eps: Tensor<2> =
            Tensor::random(std.shape(), Distribution::Normal(0.0, 1.0), &std.device());

        mu + std * eps
    }

    pub fn decode_logits(&self, z: Tensor<2>) -> Tensor<2> {
        let batch_size = z.dims()[0];

        let h = activation::relu(self.dec_linear.forward(z));

        let h: Tensor<4> = h.reshape([batch_size, CHANNELS_2, FEATURE_SIZE, FEATURE_SIZE]);

        let h = activation::relu(self.dec_conv1.forward(h));

        // Última capa sin ReLU ni sigmoid.
        let logits = self.dec_conv2.forward(h);

        assert_eq!(
            logits.dims(),
            [batch_size, 1, IMAGE_SIZE, IMAGE_SIZE],
            "El decoder no ha reconstruido la resolución esperada",
        );

        logits.reshape([batch_size, INPUT_DIM])
    }

    pub fn decode(&self, z: Tensor<2>) -> Tensor<2> {
        activation::sigmoid(self.decode_logits(z))
    }

    pub fn forward_logits(&self, x: Tensor<2>) -> (Tensor<2>, Tensor<2>, Tensor<2>) {
        let (mu, logvar) = self.encode(x);

        let z = self.reparametrize(mu.clone(), logvar.clone());

        let logits = self.decode_logits(z);

        (logits, mu, logvar)
    }

    pub fn forward(&self, x: Tensor<2>) -> (Tensor<2>, Tensor<2>, Tensor<2>) {
        let (logits, mu, logvar) = self.forward_logits(x);

        (activation::sigmoid(logits), mu, logvar)
    }
}
