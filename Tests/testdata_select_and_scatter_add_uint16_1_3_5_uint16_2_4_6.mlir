"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x4x6xui16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x3x5xui16>, tensor<2x4x6xui16>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<2x4x6xui16>
    %5 = "stablehlo.constant"() <{value = dense<0> : tensor<ui16>}> : () -> tensor<ui16>
    %6 = "stablehlo.pad"(%3#1, %5) <{edge_padding_high = array<i64: 0, 0, 0>, edge_padding_low = array<i64: 0, 0, 0>, interior_padding = array<i64: 0, 0, 0>}> : (tensor<2x4x6xui16>, tensor<ui16>) -> tensor<2x4x6xui16>
    %7 = "stablehlo.constant"() <{value = dense<0> : tensor<ui16>}> : () -> tensor<ui16>
    %8 = "stablehlo.select_and_scatter"(%6, %3#0, %7) <{window_dimensions = array<i64: 2, 2, 2>}> ({
    ^bb0(%arg2: tensor<ui16>, %arg3: tensor<ui16>):
      %11 = "stablehlo.compare"(%arg2, %arg3) <{compare_type = #stablehlo<comparison_type UNSIGNED>, comparison_direction = #stablehlo<comparison_direction GE>}> : (tensor<ui16>, tensor<ui16>) -> tensor<i1>
      "stablehlo.return"(%11) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      %10 = "stablehlo.add"(%arg0, %arg1) : (tensor<ui16>, tensor<ui16>) -> tensor<ui16>
      "stablehlo.return"(%10) : (tensor<ui16>) -> ()
    }) : (tensor<2x4x6xui16>, tensor<1x3x5xui16>, tensor<ui16>) -> tensor<2x4x6xui16>
    %9 = "stablehlo.slice"(%8) <{limit_indices = array<i64: 2, 4, 6>, start_indices = array<i64: 0, 0, 0>, strides = array<i64: 1, 1, 1>}> : (tensor<2x4x6xui16>) -> tensor<2x4x6xui16>
    "stablehlo.custom_call"(%9, %4) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<2x4x6xui16>, tensor<2x4x6xui16>) -> ()
    "func.return"(%9) : (tensor<2x4x6xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x3x5xui16>, tensor<2x4x6xui16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[2, 0, 5, 3, 0], [2, 0, 0, 3, 1], [4, 0, 1, 1, 0]]]> : tensor<1x3x5xui16>}> : () -> tensor<1x3x5xui16>
    %2 = "stablehlo.constant"() <{value = dense<[[[2, 3, 0, 1, 1, 2], [3, 0, 7, 3, 0, 7], [2, 3, 2, 1, 0, 3], [1, 0, 4, 1, 2, 5]], [[0, 0, 0, 2, 7, 3], [4, 3, 1, 0, 0, 0], [2, 4, 1, 3, 3, 0], [0, 3, 1, 2, 2, 3]]]> : tensor<2x4x6xui16>}> : () -> tensor<2x4x6xui16>
    "func.return"(%1, %2) : (tensor<1x3x5xui16>, tensor<2x4x6xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x4x6xui16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[0, 0, 0, 0, 0, 0], [0, 0, 5, 3, 0, 1], [0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]], [[0, 0, 0, 0, 3, 0], [4, 0, 0, 0, 0, 0], [0, 4, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0]]]> : tensor<2x4x6xui16>}> : () -> tensor<2x4x6xui16>
    "func.return"(%0) : (tensor<2x4x6xui16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

