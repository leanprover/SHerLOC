"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x5xi32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<4x6xi32>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x5xi32>
    %4 = "stablehlo.constant"() <{value = dense<2147483647> : tensor<i32>}> : () -> tensor<i32>
    %5 = "stablehlo.broadcast_in_dim"(%4) <{broadcast_dimensions = array<i64>}> : (tensor<i32>) -> tensor<i32>
    %6 = "stablehlo.reduce_window"(%2, %5) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %7 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "stablehlo.return"(%7) : (tensor<i32>) -> ()
    }) : (tensor<4x6xi32>, tensor<i32>) -> tensor<3x5xi32>
    "stablehlo.custom_call"(%6, %3) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<3x5xi32>, tensor<3x5xi32>) -> ()
    "func.return"(%6) : (tensor<3x5xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xi32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[0, -1, 1, 1, 2, 0], [1, 0, 0, 0, -1, -1], [0, -4, -1, 0, -4, -4], [-2, -1, 0, 5, 3, 2]]> : tensor<4x6xi32>}> : () -> tensor<4x6xi32>
    "func.return"(%1) : (tensor<4x6xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x5xi32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-1, -1, 0, -1, -1], [-4, -4, -1, -4, -4], [-4, -4, -1, -4, -4]]> : tensor<3x5xi32>}> : () -> tensor<3x5xi32>
    "func.return"(%0) : (tensor<3x5xi32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

