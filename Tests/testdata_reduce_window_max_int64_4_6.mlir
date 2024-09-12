"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x5xi64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<4x6xi64>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x5xi64>
    %4 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %5 = "stablehlo.reduce_window"(%2, %4) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %6 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%6) : (tensor<i64>) -> ()
    }) : (tensor<4x6xi64>, tensor<i64>) -> tensor<3x5xi64>
    "stablehlo.custom_call"(%5, %3) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<3x5xi64>, tensor<3x5xi64>) -> ()
    "func.return"(%5) : (tensor<3x5xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xi64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-1, -5, 0, 1, -2, 0], [3, 1, -2, 4, 0, 1], [0, 0, -1, 4, 1, -7], [3, -3, -3, -2, 0, -5]]> : tensor<4x6xi64>}> : () -> tensor<4x6xi64>
    "func.return"(%1) : (tensor<4x6xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x5xi64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[3, 1, 4, 4, 1], [3, 1, 4, 4, 1], [3, 1, 4, 4, 1]]> : tensor<3x5xi64>}> : () -> tensor<3x5xi64>
    "func.return"(%0) : (tensor<3x5xi64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

