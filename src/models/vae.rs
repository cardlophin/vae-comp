use crate::models::utils::{RELU_XAVIER_GAIN, SIGMOID_XAVIER_GAIN, linear};
use burn::{
    module::Module,
    nn::Linear,
    tensor::{Device, Distribution, Tensor, activation},
};

#[derive(Module, Debug)]
pub struct VAE {
    fc1: Linear,

    fc_mu: Linear,
    fc_logvar: Linear,

    fc2: Linear,
    fc3: Linear,
}

impl VAE {
    pub fn new(d_input: usize, d_hidden: usize, d_latent: usize, device: &Device) -> Self {
        VAE {
            fc1: linear(d_input, d_hidden, true, RELU_XAVIER_GAIN, device),
            fc_mu: linear(d_hidden, d_latent, true, RELU_XAVIER_GAIN, device),
            fc_logvar: linear(d_hidden, d_latent, true, RELU_XAVIER_GAIN, device),
            fc2: linear(d_latent, d_hidden, true, RELU_XAVIER_GAIN, device),
            fc3: linear(d_hidden, d_input, true, SIGMOID_XAVIER_GAIN, device),
        }
    }

    fn encode(&self, x: Tensor<2>) -> (Tensor<2>, Tensor<2>) {
        let h = activation::relu(self.fc1.forward(x));
        (self.fc_mu.forward(h.clone()), self.fc_logvar.forward(h))
    }

    fn reparametrize(&self, mu: Tensor<2>, logvar: Tensor<2>) -> Tensor<2> {
        let std = logvar.mul_scalar(0.5).exp();
        let eps: Tensor<2> =
            Tensor::random(std.shape(), Distribution::Normal(0.0, 1.0), &std.device());
        mu + std * eps
    }

    pub fn decode_logits(&self, z: Tensor<2>) -> Tensor<2> {
        let h = activation::relu(self.fc2.forward(z));

        self.fc3.forward(h)
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
