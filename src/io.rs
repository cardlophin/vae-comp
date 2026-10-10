pub mod mnist;
use std::path::Path;

use burn::tensor::Tensor;
use image::{
    GrayImage, ImageResult, Luma,
    imageops::{FilterType, resize},
};

const DIGIT_WIDTH: usize = 28;
const DIGIT_HEIGHT: usize = 28;
const DIGIT_PIXELS: usize = DIGIT_WIDTH * DIGIT_HEIGHT;

pub fn save_mnist_grid(
    images: Tensor<2>,
    columns: usize,
    scale: u32,
    path: impl AsRef<Path>,
) -> ImageResult<()> {
    let [num_images, num_pixels] = images.dims();

    assert!(num_images > 0, "No hay imágenes para guardar");
    assert!(columns > 0, "El número de columnas debe ser positivo");
    assert!(scale > 0, "La escala debe ser positiva");

    assert_eq!(
        num_pixels, DIGIT_PIXELS,
        "Cada imagen debe contener 784 píxeles",
    );

    let data = images.into_data();

    let pixels = data
        .try_to_vec::<f32>()
        .expect("No se pudieron leer los píxeles del tensor");

    assert!(
        pixels.iter().all(|value| value.is_finite()),
        "Las imágenes contienen valores NaN o infinitos",
    );

    let columns = columns.min(num_images);
    let rows = num_images.div_ceil(columns);

    // Separación y margen exterior, en píxeles antes de ampliar.
    let padding = 2_usize;

    let grid_width = columns * DIGIT_WIDTH + (columns + 1) * padding;

    let grid_height = rows * DIGIT_HEIGHT + (rows + 1) * padding;

    let mut grid = GrayImage::new(grid_width as u32, grid_height as u32);

    for image_index in 0..num_images {
        let column = image_index % columns;
        let row = image_index / columns;

        let offset_x = padding + column * (DIGIT_WIDTH + padding);

        let offset_y = padding + row * (DIGIT_HEIGHT + padding);

        let start = image_index * DIGIT_PIXELS;

        for y in 0..DIGIT_HEIGHT {
            for x in 0..DIGIT_WIDTH {
                let value = pixels[start + y * DIGIT_WIDTH + x];

                let intensity = (value.clamp(0.0, 1.0) * 255.0).round() as u8;

                grid.put_pixel(
                    (offset_x + x) as u32,
                    (offset_y + y) as u32,
                    Luma([intensity]),
                );
            }
        }
    }

    if scale == 1 {
        grid.save(path)
    } else {
        let enlarged = resize(
            &grid,
            grid.width() * scale,
            grid.height() * scale,
            FilterType::Nearest,
        );

        enlarged.save(path)
    }
}
