"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x4x6xi64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x1x6xi64>, tensor<2x4x6xi64>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<2x4x6xi64>
    %5 = "stablehlo.constant"() <{value = dense<-9223372036854775808> : tensor<i64>}> : () -> tensor<i64>
    %6 = "stablehlo.pad"(%3#1, %5) <{edge_padding_high = array<i64: 0, 0, 0>, edge_padding_low = array<i64: 0, 0, 0>, interior_padding = array<i64: 0, 0, 0>}> : (tensor<2x4x6xi64>, tensor<i64>) -> tensor<2x4x6xi64>
    %7 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %8 = "stablehlo.select_and_scatter"(%6, %3#0, %7) <{window_dimensions = array<i64: 1, 3, 1>, window_strides = array<i64: 1, 2, 1>}> ({
    ^bb0(%arg2: tensor<i64>, %arg3: tensor<i64>):
      %11 = "stablehlo.compare"(%arg2, %arg3) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction GE>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
      "stablehlo.return"(%11) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %10 = "stablehlo.add"(%arg0, %arg1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%10) : (tensor<i64>) -> ()
    }) : (tensor<2x4x6xi64>, tensor<2x1x6xi64>, tensor<i64>) -> tensor<2x4x6xi64>
    %9 = "stablehlo.slice"(%8) <{limit_indices = array<i64: 2, 4, 6>, start_indices = array<i64: 0, 0, 0>, strides = array<i64: 1, 1, 1>}> : (tensor<2x4x6xi64>) -> tensor<2x4x6xi64>
    "stablehlo.custom_call"(%9, %4) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<2x4x6xi64>, tensor<2x4x6xi64>) -> ()
    "func.return"(%9) : (tensor<2x4x6xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x1x6xi64>, tensor<2x4x6xi64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[-4, 0, -1, -2, 2, -3]], [[-3, 3, 7, -2, 0, -3]]]> : tensor<2x1x6xi64>}> : () -> tensor<2x1x6xi64>
    %2 = "stablehlo.constant"() <{value = dense<[[[0, 0, 1, -3, 1, -2], [-1, -3, -3, 0, -1, 2], [4, -1, 0, 1, -1, 1], [0, 6, 0, -1, 1, -1]], [[3, 4, 6, 2, -1, 2], [-2, 2, 0, 0, 1, -2], [4, -1, -4, -1, 2, -3], [5, -5, 0, 0, -2, 0]]]> : tensor<2x4x6xi64>}> : () -> tensor<2x4x6xi64>
    "func.return"(%1, %2) : (tensor<2x1x6xi64>, tensor<2x4x6xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x4x6xi64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[0, 0, -1, 0, 2, 0], [0, 0, 0, 0, 0, -3], [-4, 0, 0, -2, 0, 0], [0, 0, 0, 0, 0, 0]], [[0, 3, 7, -2, 0, -3], [0, 0, 0, 0, 0, 0], [-3, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]]> : tensor<2x4x6xi64>}> : () -> tensor<2x4x6xi64>
    "func.return"(%0) : (tensor<2x4x6xi64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

