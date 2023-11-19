use image::{imageops::FilterType, Pixel};
use std::{cmp::Ordering, collections::HashMap, time::Instant};
use wonnx::Session;

async fn run() {
    log::info!("run");

    let mut input = HashMap::new();

    // Load image
    let img = image::open("./res/digit_example.png")
        .unwrap()
        .resize_exact(28, 28, FilterType::Nearest)
        .to_luma8();
    let dim = img.dimensions();
    assert_eq!(dim, (28, 28));

    // into ndarray
    let img = ndarray::Array::from_shape_fn((1, 1, 28, 28), |(_, _, j, i)| {
        let val = img.get_pixel(i as u32, j as u32).channels()[0];
        val as f32 / 255.0
    });

    // insert input
    input.insert("Input3".into(), img.as_slice().unwrap().into());

    // load the onnx session model
    match Session::from_path("model/mnist-model.onnx").await {
        Ok(session) => {
            log::info!("session ready");
            match session.run(&input).await {
                Ok(output) => {
                    //
                    let (_, logits) = output.into_iter().next().unwrap();
                    let logits: Vec<f32> = logits.try_into().unwrap();
                    let result = logits
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

                    assert_eq!(result.0, 5);
                    log::info!("Result digit is {}", result.0);
                }
                Err(err) => {
                    log::error!("Run model error - {:?}", err);
                }
            }
        }
        Err(err) => {
            log::error!("Session error - {:?}", err);
        }
    }
}

fn main() {
    dotenv::dotenv().ok();
    env_logger::init();
    let instant = Instant::now();
    pollster::block_on(run());

    log::info!("Compute time : {} secs", instant.elapsed().as_secs_f64());
}
