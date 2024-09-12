"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x5xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<4x6xf32>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x5xf32>
    %4 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %5 = "stablehlo.reduce_window"(%2, %4) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %6 = "stablehlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%6) : (tensor<f32>) -> ()
    }) : (tensor<4x6xf32>, tensor<f32>) -> tensor<3x5xf32>
    "stablehlo.custom_call"(%5, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x5xf32>, tensor<3x5xf32>) -> ()
    "func.return"(%5) : (tensor<3x5xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-6.80302143, 0.321375608, -1.86038101, 3.62991023, 0.598879099, -5.04817581], [-0.74092257, -0.301740915, -2.58586526, 6.663420e-01, 2.39137483, 6.18894672], [-4.40329695, 1.96387827, 2.2082417, -2.70424533, 1.05929911, -0.497613579], [3.7442131, -2.94033289, -4.3009696, -1.19836605, 1.62041414, -0.197622195]]> : tensor<4x6xf32>}> : () -> tensor<4x6xf32>
    "func.return"(%1) : (tensor<4x6xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x5xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-6.52430916, -3.42661142, 0.850006103, 8.28650569, 5.13102484], [-2.48208237, 2.28451395, -1.41552687, 2.41277075, 10.1420069], [-0.635538578, -2.0691824, -4.99533939, -0.222898126, 2.98447728]]> : tensor<3x5xf32>}> : () -> tensor<3x5xf32>
    "func.return"(%0) : (tensor<3x5xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

