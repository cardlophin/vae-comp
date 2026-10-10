use burn::{module::Initializer::XavierUniform, nn::Linear, nn::LinearConfig, tensor::Device};

use std::f64::consts::SQRT_2;

pub const RELU_XAVIER_GAIN: f64 = SQRT_2;
pub const SIGMOID_XAVIER_GAIN: f64 = 1.0_f64;

pub fn linear(
    d_input: usize,
    d_output: usize,
    bias: bool,
    xavier_gain: f64,
    device: &Device,
) -> Linear {
    LinearConfig::new(d_input, d_output)
        .with_bias(bias)
        .with_initializer(XavierUniform { gain: xavier_gain })
        .init(device)
}
