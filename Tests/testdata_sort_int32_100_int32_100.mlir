"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> (tensor<100xi32>, tensor<100xi32>), res_attrs = [{jax.result_info = "[0]", mhlo.layout_mode = "default"}, {jax.result_info = "[1]", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<100xi32>, tensor<100xi32>)
    %5:2 = "func.call"() <{callee = @expected}> : () -> (tensor<100xi32>, tensor<100xi32>)
    %6:2 = "stablehlo.sort"(%4#0, %4#1) <{dimension = 0 : i64, is_stable = true}> ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
      %7 = "stablehlo.compare"(%arg2, %arg3) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %8 = "stablehlo.compare"(%arg0, %arg1) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %9 = "stablehlo.compare"(%arg0, %arg1) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %10 = "stablehlo.and"(%9, %7) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %11 = "stablehlo.or"(%8, %10) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      "stablehlo.return"(%11) : (tensor<i1>) -> ()
    }) : (tensor<100xi32>, tensor<100xi32>) -> (tensor<100xi32>, tensor<100xi32>)
    "stablehlo.custom_call"(%6#0, %5#0) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<100xi32>, tensor<100xi32>) -> ()
    "stablehlo.custom_call"(%6#1, %5#1) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<100xi32>, tensor<100xi32>) -> ()
    "func.return"(%6#0, %6#1) : (tensor<100xi32>, tensor<100xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<100xi32>, tensor<100xi32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %2 = "stablehlo.constant"() <{value = dense<[0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0]> : tensor<100xi32>}> : () -> tensor<100xi32>
    %3 = "stablehlo.constant"() <{value = dense<[-1, 0, 0, 3, -1, -7, -5, -2, 0, -3, -2, -2, -1, 2, -2, 2, -2, 0, 0, 0, -2, 4, 2, 0, -3, -1, -1, 0, 0, -2, -2, 1, 2, -1, 1, -4, 0, -3, 0, -1, 0, -2, -2, -5, 0, 0, -5, 3, 0, 0, 3, 2, -2, -3, -4, -1, 2, -1, -5, 1, -4, 0, -3, 0, -2, 2, 1, 3, 5, 0, -1, -4, 0, 3, -1, 1, 0, 2, 3, -1, 2, 0, 0, -4, 3, 0, -1, 0, 6, 2, 1, 0, -4, -1, 0, 2, -3, 0, 3, -1]> : tensor<100xi32>}> : () -> tensor<100xi32>
    "func.return"(%2, %3) : (tensor<100xi32>, tensor<100xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<100xi32>, tensor<100xi32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]> : tensor<100xi32>}> : () -> tensor<100xi32>
    %1 = "stablehlo.constant"() <{value = dense<[-5, -5, -4, -4, -4, -3, -3, -3, -2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 4, 5, -7, -5, -5, -4, -4, -4, -3, -3, -3, -2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 6]> : tensor<100xi32>}> : () -> tensor<100xi32>
    "func.return"(%0, %1) : (tensor<100xi32>, tensor<100xi32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

