"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4xi32>, res_attrs = [{jax.result_info = "[0]", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "func.call"() <{callee = @inputs}> : () -> tensor<4x6xi32>
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4xi32>
    %5 = "stablehlo.constant"() <{value = dense<3> : tensor<i32>}> : () -> tensor<i32>
    %6 = "stablehlo.reduce"(%3, %5) <{dimensions = array<i64: 1>}> ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "stablehlo.return"(%7) : (tensor<i32>) -> ()
    }) : (tensor<4x6xi32>, tensor<i32>) -> tensor<4xi32>
    "stablehlo.custom_call"(%6, %4) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<4xi32>, tensor<4xi32>) -> ()
    "func.return"(%6) : (tensor<4xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xi32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[2, 3, 7, 2, -1, -7], [5, 0, 2, -4, -3, 1], [-1, 7, 0, -1, 3, -3], [-4, 2, -3, -7, 0, -1]]> : tensor<4x6xi32>}> : () -> tensor<4x6xi32>
    %2 = "stablehlo.constant"() <{value = dense<[[2, 3, 0, 0, 0, 4], [7, 2, -5, -1, 1, 2], [0, 0, 0, 0, 4, -6], [-1, 0, -7, 2, 3, 0]]> : tensor<4x6xi32>}> : () -> tensor<4x6xi32>
    "func.return"(%1) : (tensor<4x6xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4xi32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[9, 4, 8, -10]> : tensor<4xi32>}> : () -> tensor<4xi32>
    "func.return"(%0) : (tensor<4xi32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

