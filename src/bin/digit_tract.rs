use std::{cmp::Ordering, time::Instant};

use tract_onnx::prelude::*;

fn main() -> TractResult<()> {
    dotenv::dotenv().ok();
    env_logger::init();
    let instant = Instant::now();

    //====
    let model = tract_onnx::onnx()
        .model_for_path("model/mnist-model.onnx")?
        .into_optimized()?
        .into_runnable()?;

    let img = image::open("res/digit_example.png").unwrap().to_luma8();
    let resized = image::imageops::resize(&img, 28, 28, ::image::imageops::FilterType::Triangle);
    let img: Tensor = tract_ndarray::Array4::from_shape_fn((1, 1, 28, 28), |(_, c, y, x)| {
        resized[(x as _, y as _)][c] as f32
    })
    .into();

    let result = model.run(tvec!(img.into()))?;
    let value = result[0]
        .to_array_view::<f32>()?
        .into_iter()
        .enumerate()
        .max_by(|x, y| {
            if x.1 > y.1 {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        })
        .unwrap();

    assert_eq!(value.0, 5);
    log::info!("Result digit is {}", value.0);

    //====
    log::info!("Compute time : {} secs", instant.elapsed().as_secs_f64());
    Ok(())
}
