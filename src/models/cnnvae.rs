use crate::models::utils::{RELU_XAVIER_GAIN, SIGMOID_XAVIER_GAIN, linear};
use burn::{
    module::{
        Initializer::{KaimingUniform, XavierUniform},
        Module,
    },
    nn::{
        Linear, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
    },
    tensor::{Device, Distribution, Tensor, activation},
};

// Every convolution halves (or doubles) the spatial resolution.
const KERNEL_SIZE: usize = 4;
const STRIDE: usize = 2;
const PADDING: usize = 1;

#[derive(Module, Debug)]
pub struct CNNVAE {
    image_size: usize,
    d_channels_2: usize,
    feature_size: usize,

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
    pub fn new(
        d_input: usize,
        d_channels_1: usize,
        d_channels_2: usize,
        d_latent: usize,
        device: &Device,
    ) -> Self {
        let image_size = d_input.isqrt();

        assert_eq!(
            image_size * image_size,
            d_input,
            "Input must be a square image"
        );
        assert_eq!(image_size % 4, 0, "Image size must be divisible by 4");
        assert!(d_channels_1 > 0 && d_channels_2 > 0 && d_latent > 0);

        let feature_size = image_size / 4;
        let flat_features = d_channels_2 * feature_size * feature_size;

        let conv = |d_in, d_out| {
            Conv2dConfig::new([d_in, d_out], [KERNEL_SIZE, KERNEL_SIZE])
                .with_stride([STRIDE, STRIDE])
                .with_padding(PaddingConfig2d::Explicit(
                    PADDING, PADDING, PADDING, PADDING,
                ))
                .with_initializer(XavierUniform {
                    gain: RELU_XAVIER_GAIN,
                })
                .init(device)
        };

        // Burn only provides fan_in to transposed convolutions, so Xavier is
        // not available; use Kaiming with the same gains instead.
        let conv_transpose = |d_in, d_out, gain| {
            ConvTranspose2dConfig::new([d_in, d_out], [KERNEL_SIZE, KERNEL_SIZE])
                .with_stride([STRIDE, STRIDE])
                .with_padding([PADDING, PADDING])
                .with_initializer(KaimingUniform {
                    gain,
                    fan_out_only: false,
                })
                .init(device)
        };

        Self {
            image_size,
            d_channels_2,
            feature_size,

            enc_conv1: conv(1, d_channels_1),
            enc_conv2: conv(d_channels_1, d_channels_2),

            fc_mu: linear(flat_features, d_latent, true, RELU_XAVIER_GAIN, device),
            fc_logvar: linear(flat_features, d_latent, true, RELU_XAVIER_GAIN, device),

            dec_linear: linear(d_latent, flat_features, true, RELU_XAVIER_GAIN, device),
            dec_conv1: conv_transpose(d_channels_2, d_channels_1, RELU_XAVIER_GAIN),
            dec_conv2: conv_transpose(d_channels_1, 1, SIGMOID_XAVIER_GAIN),
        }
    }

    pub fn encode(&self, x: Tensor<2>) -> (Tensor<2>, Tensor<2>) {
        let [batch_size, input_dim] = x.dims();

        assert_eq!(
            input_dim,
            self.image_size * self.image_size,
            "Unexpected input size for this model",
        );

        let x: Tensor<4> = x.reshape([batch_size, 1, self.image_size, self.image_size]);

        let h = activation::relu(self.enc_conv1.forward(x));

        let h = activation::relu(self.enc_conv2.forward(h));

        let h: Tensor<2> = h.flatten(1, 3);

        // No activation on the Gaussian heads.
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

        let h: Tensor<4> = h.reshape([
            batch_size,
            self.d_channels_2,
            self.feature_size,
            self.feature_size,
        ]);

        let h = activation::relu(self.dec_conv1.forward(h));

        // Last layer without ReLU or sigmoid.
        let logits = self.dec_conv2.forward(h);

        assert_eq!(
            logits.dims(),
            [batch_size, 1, self.image_size, self.image_size],
            "Decoder did not reconstruct the expected resolution",
        );

        logits.reshape([batch_size, self.image_size * self.image_size])
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
