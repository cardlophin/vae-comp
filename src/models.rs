pub mod cnnvae;
pub mod utils;
pub mod vae;

use burn::{module::Module, tensor::Tensor};

use cnnvae::CNNVAE;
use vae::VAE;

pub trait VariationalModel: Module {
    fn forward_logits(&self, x: Tensor<2>) -> (Tensor<2>, Tensor<2>, Tensor<2>);

    fn decode(&self, z: Tensor<2>) -> Tensor<2>;
}

impl VariationalModel for VAE {
    fn forward_logits(&self, x: Tensor<2>) -> (Tensor<2>, Tensor<2>, Tensor<2>) {
        VAE::forward_logits(self, x)
    }

    fn decode(&self, z: Tensor<2>) -> Tensor<2> {
        VAE::decode(self, z)
    }
}

impl VariationalModel for CNNVAE {
    fn forward_logits(&self, x: Tensor<2>) -> (Tensor<2>, Tensor<2>, Tensor<2>) {
        CNNVAE::forward_logits(self, x)
    }

    fn decode(&self, z: Tensor<2>) -> Tensor<2> {
        CNNVAE::decode(self, z)
    }
}
