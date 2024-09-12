"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x7xi64>, res_attrs = [{jax.result_info = "[0]", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<5x7xi64>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<5x7xi64>
    %4 = "stablehlo.sort"(%2) <{dimension = 0 : i64}> ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %5 = "stablehlo.compare"(%arg0, %arg1) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
      "stablehlo.return"(%5) : (tensor<i1>) -> ()
    }) : (tensor<5x7xi64>) -> tensor<5x7xi64>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x7xi64>, tensor<5x7xi64>) -> ()
    "func.return"(%4) : (tensor<5x7xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x7xi64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[0, 1, 1, -3, -3, -2, 0], [4, -1, 3, 0, 4, 0, -3], [1, 0, 0, 0, 0, 2, 3], [-3, -4, -4, -2, 0, 3, 3], [0, 5, 2, -2, 0, -2, 0]]> : tensor<5x7xi64>}> : () -> tensor<5x7xi64>
    "func.return"(%1) : (tensor<5x7xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x7xi64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-3, -4, -4, -3, -3, -2, -3], [0, -1, 0, -2, 0, -2, 0], [0, 0, 1, -2, 0, 0, 0], [1, 1, 2, 0, 0, 2, 3], [4, 5, 3, 0, 4, 3, 3]]> : tensor<5x7xi64>}> : () -> tensor<5x7xi64>
    "func.return"(%0) : (tensor<5x7xi64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

