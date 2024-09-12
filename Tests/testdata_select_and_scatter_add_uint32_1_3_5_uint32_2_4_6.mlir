"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x4x6xui32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x3x5xui32>, tensor<2x4x6xui32>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<2x4x6xui32>
    %5 = "stablehlo.constant"() <{value = dense<0> : tensor<ui32>}> : () -> tensor<ui32>
    %6 = "stablehlo.pad"(%3#1, %5) <{edge_padding_high = array<i64: 0, 0, 0>, edge_padding_low = array<i64: 0, 0, 0>, interior_padding = array<i64: 0, 0, 0>}> : (tensor<2x4x6xui32>, tensor<ui32>) -> tensor<2x4x6xui32>
    %7 = "stablehlo.constant"() <{value = dense<0> : tensor<ui32>}> : () -> tensor<ui32>
    %8 = "stablehlo.select_and_scatter"(%6, %3#0, %7) <{window_dimensions = array<i64: 2, 2, 2>}> ({
    ^bb0(%arg2: tensor<ui32>, %arg3: tensor<ui32>):
      %11 = "stablehlo.compare"(%arg2, %arg3) <{compare_type = #stablehlo<comparison_type UNSIGNED>, comparison_direction = #stablehlo<comparison_direction GE>}> : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
      "stablehlo.return"(%11) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg0: tensor<ui32>, %arg1: tensor<ui32>):
      %10 = "stablehlo.add"(%arg0, %arg1) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
      "stablehlo.return"(%10) : (tensor<ui32>) -> ()
    }) : (tensor<2x4x6xui32>, tensor<1x3x5xui32>, tensor<ui32>) -> tensor<2x4x6xui32>
    %9 = "stablehlo.slice"(%8) <{limit_indices = array<i64: 2, 4, 6>, start_indices = array<i64: 0, 0, 0>, strides = array<i64: 1, 1, 1>}> : (tensor<2x4x6xui32>) -> tensor<2x4x6xui32>
    "stablehlo.custom_call"(%9, %4) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<2x4x6xui32>, tensor<2x4x6xui32>) -> ()
    "func.return"(%9) : (tensor<2x4x6xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x3x5xui32>, tensor<2x4x6xui32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[0, 2, 0, 1, 1], [0, 2, 3, 1, 5], [3, 1, 0, 2, 6]]]> : tensor<1x3x5xui32>}> : () -> tensor<1x3x5xui32>
    %2 = "stablehlo.constant"() <{value = dense<[[[3, 1, 3, 2, 7, 1], [0, 2, 1, 0, 1, 1], [1, 5, 4, 3, 2, 5], [1, 0, 1, 0, 5, 2]], [[6, 2, 3, 0, 2, 0], [2, 0, 0, 0, 1, 0], [2, 3, 0, 3, 4, 1], [0, 2, 0, 2, 6, 1]]]> : tensor<2x4x6xui32>}> : () -> tensor<2x4x6xui32>
    "func.return"(%1, %2) : (tensor<1x3x5xui32>, tensor<2x4x6xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x4x6xui32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[0, 0, 2, 0, 2, 0], [0, 0, 0, 0, 0, 0], [0, 6, 3, 0, 0, 5], [0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 8, 0]]]> : tensor<2x4x6xui32>}> : () -> tensor<2x4x6xui32>
    "func.return"(%0) : (tensor<2x4x6xui32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

