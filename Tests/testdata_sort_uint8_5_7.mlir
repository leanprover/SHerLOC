"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x7xui8>, res_attrs = [{jax.result_info = "[0]", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<5x7xui8>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<5x7xui8>
    %4 = "stablehlo.sort"(%2) <{dimension = 0 : i64}> ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      %5 = "stablehlo.compare"(%arg0, %arg1) <{compare_type = #stablehlo<comparison_type UNSIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<ui8>, tensor<ui8>) -> tensor<i1>
      "stablehlo.return"(%5) : (tensor<i1>) -> ()
    }) : (tensor<5x7xui8>) -> tensor<5x7xui8>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x7xui8>, tensor<5x7xui8>) -> ()
    "func.return"(%4) : (tensor<5x7xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x7xui8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[5, 4, 2, 2, 3, 3, 0], [2, 2, 0, 1, 1, 4, 0], [0, 0, 1, 0, 2, 4, 3], [2, 0, 1, 2, 0, 5, 0], [1, 4, 4, 0, 5, 1, 0]]> : tensor<5x7xui8>}> : () -> tensor<5x7xui8>
    "func.return"(%1) : (tensor<5x7xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x7xui8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[0, 0, 0, 0, 0, 1, 0], [1, 0, 1, 0, 1, 3, 0], [2, 2, 1, 1, 2, 4, 0], [2, 4, 2, 2, 3, 4, 0], [5, 4, 4, 2, 5, 5, 3]]> : tensor<5x7xui8>}> : () -> tensor<5x7xui8>
    "func.return"(%0) : (tensor<5x7xui8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

