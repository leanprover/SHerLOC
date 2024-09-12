"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x7xi32>, res_attrs = [{jax.result_info = "[0]", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<5x7xi32>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<5x7xi32>
    %4 = "stablehlo.sort"(%2) <{dimension = 0 : i64}> ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %5 = "stablehlo.compare"(%arg0, %arg1) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i32>, tensor<i32>) -> tensor<i1>
      "stablehlo.return"(%5) : (tensor<i1>) -> ()
    }) : (tensor<5x7xi32>) -> tensor<5x7xi32>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x7xi32>, tensor<5x7xi32>) -> ()
    "func.return"(%4) : (tensor<5x7xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x7xi32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[2, 4, -2, 4, -4, 0, -2], [0, -2, 2, 0, -1, 4, 0], [0, 3, 0, -3, 1, -2, -1], [-1, 0, 0, 0, 0, 3, 2], [2, -5, -3, -1, 0, 2, -2]]> : tensor<5x7xi32>}> : () -> tensor<5x7xi32>
    "func.return"(%1) : (tensor<5x7xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x7xi32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-1, -5, -3, -3, -4, -2, -2], [0, -2, -2, -1, -1, 0, -2], [0, 0, 0, 0, 0, 2, -1], [2, 3, 0, 0, 0, 3, 0], [2, 4, 2, 4, 1, 4, 2]]> : tensor<5x7xi32>}> : () -> tensor<5x7xi32>
    "func.return"(%0) : (tensor<5x7xi32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

