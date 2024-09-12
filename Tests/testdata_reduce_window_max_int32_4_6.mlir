"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x5xi32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<4x6xi32>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x5xi32>
    %4 = "stablehlo.constant"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
    %5 = "stablehlo.reduce_window"(%2, %4) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %6 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "stablehlo.return"(%6) : (tensor<i32>) -> ()
    }) : (tensor<4x6xi32>, tensor<i32>) -> tensor<3x5xi32>
    "stablehlo.custom_call"(%5, %3) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<3x5xi32>, tensor<3x5xi32>) -> ()
    "func.return"(%5) : (tensor<3x5xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xi32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-3, 1, -2, -1, 3, -1], [-4, -3, -1, 1, 0, -2], [-1, 0, -3, -5, 0, 3], [6, 1, 0, 0, 2, -5]]> : tensor<4x6xi32>}> : () -> tensor<4x6xi32>
    "func.return"(%1) : (tensor<4x6xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x5xi32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1, 1, 1, 3, 3], [1, 1, 1, 1, 3], [6, 1, 1, 2, 3]]> : tensor<3x5xi32>}> : () -> tensor<3x5xi32>
    "func.return"(%0) : (tensor<3x5xi32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

