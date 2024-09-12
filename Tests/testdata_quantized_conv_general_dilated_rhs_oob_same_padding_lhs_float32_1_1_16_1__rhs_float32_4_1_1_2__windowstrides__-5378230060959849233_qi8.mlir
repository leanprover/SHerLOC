"builtin.module"() ({
  "func.func"() <{function_type = () -> tensor<i1>, sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[[1.94757128], [0.731765032], [-0.726273656], [-5.65812302], [-0.322147697], [2.76183558], [4.334000e+00], [0.198797315], [0.552351534], [-3.67483807], [1.00321078], [1.13255763], [1.41218019], [2.263990e+00], [-3.61878276], [2.55396557]]]]> : tensor<1x1x16x1xf32>}> : () -> tensor<1x1x16x1xf32>
    %1 = "stablehlo.constant"() <{value = dense<[[[[-2.90031719, 3.83649588]]], [[[3.26177287, 6.146370e-01]]], [[[2.2365787, -2.09627247]]], [[[-2.90434122, -0.155085653]]]]> : tensor<4x1x1x2xf32>}> : () -> tensor<4x1x1x2xf32>
    %2 = "stablehlo.constant"() <{value = dense<[[[[0.992761671, 0.615122914], [0.731918394, 0.451609224], [0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00], [0.992761671, 0.615122914], [0.992761671, 0.615122914], [0.19855234, 0.124581859], [5.528320e-01, 0.33870694], [0.000000e+00, 0.000000e+00], [0.992761671, 0.615122914], [0.992761671, 0.615122914], [0.992761671, 0.615122914], [0.992761671, 0.615122914], [0.000000e+00, 0.000000e+00], [0.992761671, 0.615122914]]]]> : tensor<1x1x16x2xf32>}> : () -> tensor<1x1x16x2xf32>
    %3 = "stablehlo.uniform_quantize"(%1) : (tensor<4x1x1x2xf32>) -> tensor<4x1x1x2x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>
    %4 = "stablehlo.uniform_quantize"(%0) : (tensor<1x1x16x1xf32>) -> tensor<1x1x16x1x!quant.uniform<i8:f32, 0.0039188104517319626:-128>>
    %5 = "stablehlo.convolution"(%4, %3) <{batch_group_count = 1 : i64, dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>, feature_group_count = 1 : i64, padding = dense<[[1, 2], [0, 0]]> : tensor<2x2xi64>}> : (tensor<1x1x16x1x!quant.uniform<i8:f32, 0.0039188104517319626:-128>>, tensor<4x1x1x2x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>) -> tensor<1x1x16x2x!quant.uniform<i32:f32, 1.5350925350904778E-5>>
    %6 = "stablehlo.uniform_quantize"(%5) : (tensor<1x1x16x2x!quant.uniform<i32:f32, 1.5350925350904778E-5>>) -> tensor<1x1x16x2x!quant.uniform<i8:f32, 0.0038931830256592995:-128>>
    %7 = "stablehlo.uniform_dequantize"(%6) : (tensor<1x1x16x2x!quant.uniform<i8:f32, 0.0038931830256592995:-128>>) -> tensor<1x1x16x2xf32>
    %8 = "stablehlo.custom_call"(%2, %7) <{call_target_name = "check.eq"}> : (tensor<1x1x16x2xf32>, tensor<1x1x16x2xf32>) -> tensor<i1>
    "func.return"(%8) : (tensor<i1>) -> ()
  }) : () -> ()
}) {jax.uses_shape_polymorphism = true} : () -> ()

