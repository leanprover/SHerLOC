"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x7xui32>, res_attrs = [{jax.result_info = "[0]", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<5x7xui32>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<5x7xui32>
    %4 = "stablehlo.sort"(%2) <{dimension = 0 : i64}> ({
    ^bb0(%arg0: tensor<ui32>, %arg1: tensor<ui32>):
      %5 = "stablehlo.compare"(%arg0, %arg1) <{compare_type = #stablehlo<comparison_type UNSIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
      "stablehlo.return"(%5) : (tensor<i1>) -> ()
    }) : (tensor<5x7xui32>) -> tensor<5x7xui32>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x7xui32>, tensor<5x7xui32>) -> ()
    "func.return"(%4) : (tensor<5x7xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x7xui32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[1, 5, 6, 0, 0, 0, 1], [4, 1, 5, 0, 0, 2, 2], [1, 4, 2, 0, 5, 0, 5], [3, 4, 3, 0, 4, 4, 1], [0, 3, 3, 2, 2, 1, 0]]> : tensor<5x7xui32>}> : () -> tensor<5x7xui32>
    "func.return"(%1) : (tensor<5x7xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x7xui32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[0, 1, 2, 0, 0, 0, 0], [1, 3, 3, 0, 0, 0, 1], [1, 4, 3, 0, 2, 1, 1], [3, 4, 5, 0, 4, 2, 2], [4, 5, 6, 2, 5, 4, 5]]> : tensor<5x7xui32>}> : () -> tensor<5x7xui32>
    "func.return"(%0) : (tensor<5x7xui32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

