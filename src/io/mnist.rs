use std::sync::Arc;

use burn::{
    data::{
        dataloader::{DataLoader, DataLoaderBuilder, batcher::Batcher},
        dataset::{
            Dataset, DatasetError,
            transform::ShuffledDataset,
            vision::{MnistDataset, MnistItem},
        },
    },
    tensor::{Device, Tensor, TensorData},
};

pub const INPUT_DIM: usize = 784;

#[derive(Clone, Debug)]
pub struct MnistBatch {
    pub images: Tensor<2>,
}

#[derive(Clone, Debug, Default)]
pub struct MnistBatcher;

impl Batcher<MnistItem, MnistBatch> for MnistBatcher {
    fn batch(&self, items: Vec<MnistItem>, device: &Device) -> MnistBatch {
        let batch_size = items.len();
        let mut pixels = Vec::with_capacity(batch_size * INPUT_DIM);

        for item in items {
            pixels.extend(
                item.image
                    .into_iter()
                    .flatten()
                    .map(|pixel| pixel as f32 / 255.0),
            );
        }

        let data = TensorData::new(pixels, [batch_size, INPUT_DIM]);

        MnistBatch {
            images: Tensor::from_data(data, device),
        }
    }
}

type ShuffledMnist = ShuffledDataset<MnistDataset, MnistItem>;

struct MnistSplit {
    source: Arc<ShuffledMnist>,
    start: usize,
    end: usize,
}

impl Dataset<MnistItem> for MnistSplit {
    fn get(&self, index: usize) -> Result<MnistItem, DatasetError> {
        assert!(index < self.len(), "Index out of MNIST split bounds");

        self.source.get(self.start + index)
    }

    fn len(&self) -> usize {
        self.end - self.start
    }
}

pub struct MnistLoaderConfig {
    pub batch_size: usize,
    pub num_workers: usize,
    pub validation_size: usize,
    pub seed: u64,
}

pub struct MnistLoaders {
    pub train: Arc<dyn DataLoader<MnistBatch>>,
    pub validation: Arc<dyn DataLoader<MnistBatch>>,
    pub test: Arc<dyn DataLoader<MnistBatch>>,

    pub train_size: usize,
    pub validation_size: usize,
    pub test_size: usize,
}

pub fn build_loaders(
    config: &MnistLoaderConfig,
    train_device: &Device,
    eval_device: &Device,
) -> MnistLoaders {
    assert!(config.batch_size > 0);
    assert!(config.validation_size > 0);

    let source = Arc::new(ShuffledDataset::new(MnistDataset::train(), config.seed));

    let total_size = source.len();

    assert!(
        config.validation_size < total_size,
        "Validation size must be smaller than the dataset"
    );

    let train_size = total_size - config.validation_size;

    let train_dataset = MnistSplit {
        source: source.clone(),
        start: 0,
        end: train_size,
    };

    let validation_dataset = MnistSplit {
        source,
        start: train_size,
        end: total_size,
    };

    let test_dataset = MnistDataset::test();
    let test_size = test_dataset.len();

    let train = DataLoaderBuilder::new(MnistBatcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .set_device(train_device.clone())
        .build(train_dataset);

    let validation = DataLoaderBuilder::new(MnistBatcher)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .set_device(eval_device.clone())
        .build(validation_dataset);

    let test = DataLoaderBuilder::new(MnistBatcher)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .set_device(eval_device.clone())
        .build(test_dataset);

    MnistLoaders {
        train,
        validation,
        test,
        train_size,
        validation_size: config.validation_size,
        test_size,
    }
}
