"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x5xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<4x6xf32>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x5xf32>
    %4 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %5 = "stablehlo.reduce_window"(%2, %4) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %6 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%6) : (tensor<f32>) -> ()
    }) : (tensor<4x6xf32>, tensor<f32>) -> tensor<3x5xf32>
    "stablehlo.custom_call"(%5, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x5xf32>, tensor<3x5xf32>) -> ()
    "func.return"(%5) : (tensor<3x5xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[0.577798367, 0.523132384, -1.81298876, 6.48733997, 0.210517034, 1.69092274], [-4.87022209, 1.46585453, 5.6482358, -4.29839659, -0.252917409, -2.678330e+00], [-0.246699423, -6.12239218, 0.689703226, -4.10430193, -2.13349247, 1.53750014], [-2.85476828, -0.966081857, 0.410659164, -2.26567984, -1.87033808, -4.98556089]]> : tensor<4x6xf32>}> : () -> tensor<4x6xf32>
    "func.return"(%1) : (tensor<4x6xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x5xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1.46585453, 5.6482358, 6.48733997, 6.48733997, 1.69092274], [1.46585453, 5.6482358, 5.6482358, 1.000000e+00, 1.53750014], [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.53750014]]> : tensor<3x5xf32>}> : () -> tensor<3x5xf32>
    "func.return"(%0) : (tensor<3x5xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

