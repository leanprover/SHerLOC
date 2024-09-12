"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> (tensor<6xbf16>, tensor<6xi32>), res_attrs = [{jax.result_info = "[0]", mhlo.layout_mode = "default"}, {jax.result_info = "[1]", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x6xbf16>, tensor<4x6xi32>)
    %5:2 = "func.call"() <{callee = @expected}> : () -> (tensor<6xbf16>, tensor<6xi32>)
    %6 = "stablehlo.constant"() <{value = dense<3.000000e+00> : tensor<bf16>}> : () -> tensor<bf16>
    %7 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
    %8:2 = "stablehlo.reduce"(%4#0, %4#1, %6, %7) <{dimensions = array<i64: 0>}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<i32>, %arg2: tensor<bf16>, %arg3: tensor<i32>):
      %9 = "stablehlo.maximum"(%arg0, %arg2) : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      %10 = "stablehlo.minimum"(%arg1, %arg3) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "stablehlo.return"(%9, %10) : (tensor<bf16>, tensor<i32>) -> ()
    }) : (tensor<4x6xbf16>, tensor<4x6xi32>, tensor<bf16>, tensor<i32>) -> (tensor<6xbf16>, tensor<6xi32>)
    "stablehlo.custom_call"(%8#0, %5#0) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<6xbf16>, tensor<6xbf16>) -> ()
    "stablehlo.custom_call"(%8#1, %5#1) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<6xi32>, tensor<6xi32>) -> ()
    "func.return"(%8#0, %8#1) : (tensor<6xbf16>, tensor<6xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x6xbf16>, tensor<4x6xi32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %2 = "stablehlo.constant"() <{value = dense<[[2.334590e-03, -2.871090e-01, 1.953130e+00, 6.625000e+00, -3.234380e+00, 6.054690e-01], [2.031250e+00, -2.859380e+00, -4.593750e+00, -5.156250e+00, 2.109380e+00, -6.484380e-01], [6.062500e+00, -2.015630e+00, -5.062500e+00, 1.335940e+00, 3.015630e+00, 8.632810e-01], [2.734380e+00, 1.742190e+00, -5.812500e+00, 1.669920e-01, -3.062500e+00, -1.718750e+00]]> : tensor<4x6xbf16>}> : () -> tensor<4x6xbf16>
    %3 = "stablehlo.constant"() <{value = dense<[[0, 0, 0, 6, 4, 1], [3, -1, -3, -1, 0, 0], [5, 1, 6, 1, 4, 2], [0, 1, 0, -4, 2, -2]]> : tensor<4x6xi32>}> : () -> tensor<4x6xi32>
    "func.return"(%2, %3) : (tensor<4x6xbf16>, tensor<4x6xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<6xbf16>, tensor<6xi32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[6.062500e+00, 3.000000e+00, 3.000000e+00, 6.625000e+00, 3.015630e+00, 3.000000e+00]> : tensor<6xbf16>}> : () -> tensor<6xbf16>
    %1 = "stablehlo.constant"() <{value = dense<[0, -1, -3, -4, 0, -2]> : tensor<6xi32>}> : () -> tensor<6xi32>
    "func.return"(%0, %1) : (tensor<6xbf16>, tensor<6xi32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

