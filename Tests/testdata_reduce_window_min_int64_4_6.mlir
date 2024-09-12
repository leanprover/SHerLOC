"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x5xi64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<4x6xi64>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x5xi64>
    %4 = "stablehlo.constant"() <{value = dense<9223372036854775807> : tensor<i64>}> : () -> tensor<i64>
    %5 = "stablehlo.broadcast_in_dim"(%4) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<i64>
    %6 = "stablehlo.reduce_window"(%2, %5) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %7 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%7) : (tensor<i64>) -> ()
    }) : (tensor<4x6xi64>, tensor<i64>) -> tensor<3x5xi64>
    "stablehlo.custom_call"(%6, %3) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<3x5xi64>, tensor<3x5xi64>) -> ()
    "func.return"(%6) : (tensor<3x5xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xi64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-2, 0, 0, 1, -2, 2], [1, -5, 0, 1, -3, -3], [-4, 2, -1, -4, 2, 0], [-2, 0, -5, 2, -5, 0]]> : tensor<4x6xi64>}> : () -> tensor<4x6xi64>
    "func.return"(%1) : (tensor<4x6xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x5xi64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-5, -5, 0, -3, -3], [-5, -5, -4, -4, -3], [-4, -5, -5, -5, -5]]> : tensor<3x5xi64>}> : () -> tensor<3x5xi64>
    "func.return"(%0) : (tensor<3x5xi64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

