"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> (tensor<6xi1>, tensor<6xi32>), res_attrs = [{jax.result_info = "[0]", mhlo.layout_mode = "default"}, {jax.result_info = "[1]", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x6xi1>, tensor<4x6xi32>)
    %5:2 = "func.call"() <{callee = @expected}> : () -> (tensor<6xi1>, tensor<6xi32>)
    %6 = "stablehlo.constant"() <{value = dense<true> : tensor<i1>}> : () -> tensor<i1>
    %7 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
    %8:2 = "stablehlo.reduce"(%4#0, %4#1, %6, %7) <{dimensions = array<i64: 0>}> ({
    ^bb0(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i1>, %arg3: tensor<i32>):
      %9 = "stablehlo.maximum"(%arg0, %arg2) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %10 = "stablehlo.minimum"(%arg1, %arg3) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "stablehlo.return"(%9, %10) : (tensor<i1>, tensor<i32>) -> ()
    }) : (tensor<4x6xi1>, tensor<4x6xi32>, tensor<i1>, tensor<i32>) -> (tensor<6xi1>, tensor<6xi32>)
    "stablehlo.custom_call"(%8#0, %5#0) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<6xi1>, tensor<6xi1>) -> ()
    "stablehlo.custom_call"(%8#1, %5#1) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<6xi32>, tensor<6xi32>) -> ()
    "func.return"(%8#0, %8#1) : (tensor<6xi1>, tensor<6xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x6xi1>, tensor<4x6xi32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %2 = "stablehlo.constant"() <{value = dense<true> : tensor<4x6xi1>}> : () -> tensor<4x6xi1>
    %3 = "stablehlo.constant"() <{value = dense<[[-3, 0, -2, 0, 0, -2], [1, 10, 5, 0, 4, 1], [0, 3, 0, 3, 0, -1], [2, 8, 0, -2, 3, -2]]> : tensor<4x6xi32>}> : () -> tensor<4x6xi32>
    "func.return"(%2, %3) : (tensor<4x6xi1>, tensor<4x6xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<6xi1>, tensor<6xi32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<true> : tensor<6xi1>}> : () -> tensor<6xi1>
    %1 = "stablehlo.constant"() <{value = dense<[-3, 0, -2, -2, 0, -2]> : tensor<6xi32>}> : () -> tensor<6xi32>
    "func.return"(%0, %1) : (tensor<6xi1>, tensor<6xi32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

