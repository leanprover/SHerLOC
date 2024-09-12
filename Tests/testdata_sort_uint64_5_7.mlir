"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x7xui64>, res_attrs = [{jax.result_info = "[0]", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<5x7xui64>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<5x7xui64>
    %4 = "stablehlo.sort"(%2) <{dimension = 0 : i64}> ({
    ^bb0(%arg0: tensor<ui64>, %arg1: tensor<ui64>):
      %5 = "stablehlo.compare"(%arg0, %arg1) <{compare_type = #stablehlo<comparison_type UNSIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<ui64>, tensor<ui64>) -> tensor<i1>
      "stablehlo.return"(%5) : (tensor<i1>) -> ()
    }) : (tensor<5x7xui64>) -> tensor<5x7xui64>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x7xui64>, tensor<5x7xui64>) -> ()
    "func.return"(%4) : (tensor<5x7xui64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x7xui64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[3, 1, 1, 1, 0, 1, 1], [0, 2, 0, 1, 2, 4, 5], [0, 1, 2, 0, 5, 1, 2], [1, 4, 2, 1, 2, 0, 4], [0, 2, 1, 2, 3, 3, 3]]> : tensor<5x7xui64>}> : () -> tensor<5x7xui64>
    "func.return"(%1) : (tensor<5x7xui64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x7xui64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[0, 1, 0, 0, 0, 0, 1], [0, 1, 1, 1, 2, 1, 2], [0, 2, 1, 1, 2, 1, 3], [1, 2, 2, 1, 3, 3, 4], [3, 4, 2, 2, 5, 4, 5]]> : tensor<5x7xui64>}> : () -> tensor<5x7xui64>
    "func.return"(%0) : (tensor<5x7xui64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

