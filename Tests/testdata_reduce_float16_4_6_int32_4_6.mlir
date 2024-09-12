"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> (tensor<6xf16>, tensor<6xi32>), res_attrs = [{jax.result_info = "[0]", mhlo.layout_mode = "default"}, {jax.result_info = "[1]", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x6xf16>, tensor<4x6xi32>)
    %5:2 = "func.call"() <{callee = @expected}> : () -> (tensor<6xf16>, tensor<6xi32>)
    %6 = "stablehlo.constant"() <{value = dense<3.000000e+00> : tensor<f16>}> : () -> tensor<f16>
    %7 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
    %8:2 = "stablehlo.reduce"(%4#0, %4#1, %6, %7) <{dimensions = array<i64: 0>}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<i32>, %arg2: tensor<f16>, %arg3: tensor<i32>):
      %9 = "stablehlo.maximum"(%arg0, %arg2) : (tensor<f16>, tensor<f16>) -> tensor<f16>
      %10 = "stablehlo.minimum"(%arg1, %arg3) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "stablehlo.return"(%9, %10) : (tensor<f16>, tensor<i32>) -> ()
    }) : (tensor<4x6xf16>, tensor<4x6xi32>, tensor<f16>, tensor<i32>) -> (tensor<6xf16>, tensor<6xi32>)
    "stablehlo.custom_call"(%8#0, %5#0) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<6xf16>, tensor<6xf16>) -> ()
    "stablehlo.custom_call"(%8#1, %5#1) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<6xi32>, tensor<6xi32>) -> ()
    "func.return"(%8#0, %8#1) : (tensor<6xf16>, tensor<6xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x6xf16>, tensor<4x6xi32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %2 = "stablehlo.constant"() <{value = dense<[[-7.534170e-01, 3.000000e+00, -3.408200e+00, 8.002920e-01, 4.058590e+00, 5.179690e+00], [1.445310e+00, -4.501950e-01, -6.801760e-01, 1.175780e+00, -2.705080e-01, 1.590820e+00], [4.644530e+00, -1.593750e+00, 3.220700e+00, 1.577150e+00, -1.478520e+00, 1.500240e-01], [-3.230470e+00, -4.367190e+00, 3.647460e-01, 1.127930e+00, 3.892580e+00, 2.025390e+00]]> : tensor<4x6xf16>}> : () -> tensor<4x6xf16>
    %3 = "stablehlo.constant"() <{value = dense<[[-2, 1, 1, -4, 2, 0], [2, 3, 0, 6, 5, -1], [3, -5, 0, 2, -3, 0], [0, -4, 2, 1, -2, 2]]> : tensor<4x6xi32>}> : () -> tensor<4x6xi32>
    "func.return"(%2, %3) : (tensor<4x6xf16>, tensor<4x6xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<6xf16>, tensor<6xi32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[4.644530e+00, 3.000000e+00, 3.220700e+00, 3.000000e+00, 4.058590e+00, 5.179690e+00]> : tensor<6xf16>}> : () -> tensor<6xf16>
    %1 = "stablehlo.constant"() <{value = dense<[-2, -5, 0, -4, -3, -1]> : tensor<6xi32>}> : () -> tensor<6xi32>
    "func.return"(%0, %1) : (tensor<6xf16>, tensor<6xi32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

