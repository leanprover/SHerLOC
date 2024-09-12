"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x3xui16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %5:4 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x3xi32>, tensor<2x3xui16>, tensor<2x3xui16>, tensor<2x3xui16>)
    %6 = "func.call"() <{callee = @expected}> : () -> tensor<2x3xui16>
    %7 = "stablehlo.constant"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
    %8 = "stablehlo.broadcast_in_dim"(%7) <{broadcast_dimensions = array<i64>}> : (tensor<i32>) -> tensor<2x3xi32>
    %9 = "stablehlo.compare"(%5#0, %8) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
    %10 = "stablehlo.constant"() <{value = dense<2> : tensor<i32>}> : () -> tensor<i32>
    %11 = "stablehlo.broadcast_in_dim"(%10) <{broadcast_dimensions = array<i64>}> : (tensor<i32>) -> tensor<2x3xi32>
    %12 = "stablehlo.compare"(%5#0, %11) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
    %13 = "stablehlo.select"(%12, %5#2, %5#3) : (tensor<2x3xi1>, tensor<2x3xui16>, tensor<2x3xui16>) -> tensor<2x3xui16>
    %14 = "stablehlo.select"(%9, %5#1, %13) : (tensor<2x3xi1>, tensor<2x3xui16>, tensor<2x3xui16>) -> tensor<2x3xui16>
    "stablehlo.custom_call"(%14, %6) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<2x3xui16>, tensor<2x3xui16>) -> ()
    "func.return"(%14) : (tensor<2x3xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x3xi32>, tensor<2x3xui16>, tensor<2x3xui16>, tensor<2x3xui16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[0, 2, 1], [1, 1, 2]]> : tensor<2x3xi32>}> : () -> tensor<2x3xi32>
    %2 = "stablehlo.constant"() <{value = dense<[[1, 0, 0], [3, 1, 0]]> : tensor<2x3xui16>}> : () -> tensor<2x3xui16>
    %3 = "stablehlo.constant"() <{value = dense<[[0, 6, 5], [1, 1, 2]]> : tensor<2x3xui16>}> : () -> tensor<2x3xui16>
    %4 = "stablehlo.constant"() <{value = dense<[[0, 3, 3], [1, 1, 1]]> : tensor<2x3xui16>}> : () -> tensor<2x3xui16>
    "func.return"(%1, %2, %3, %4) : (tensor<2x3xi32>, tensor<2x3xui16>, tensor<2x3xui16>, tensor<2x3xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xui16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1, 3, 5], [1, 1, 1]]> : tensor<2x3xui16>}> : () -> tensor<2x3xui16>
    "func.return"(%0) : (tensor<2x3xui16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

