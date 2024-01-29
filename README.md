# ONNX Model in Rust

to Testing an ONNX model in Rust using `wonnx` and `tract` library. 

## Model

downloaded from [ONNX MNIST Model](https://github.com/onnx/models/tree/main/vision/classification/mnist).


## Using wonnx (0.4)

[wonnx GitHub](https://github.com/webonnx/wonnx)


```bash
cargo run --bin digit_wonnx
```


Encountered a runtime error when loading the model with version 0.5 (latest). 


## Using tract

[tract GitHub](https://github.com/sonos/tract).

```bash
cargo run --bin digit_tract
```
