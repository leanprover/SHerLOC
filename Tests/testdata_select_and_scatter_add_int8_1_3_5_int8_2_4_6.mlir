"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x4x6xi8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x3x5xi8>, tensor<2x4x6xi8>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<2x4x6xi8>
    %5 = "stablehlo.constant"() <{value = dense<-128> : tensor<i8>}> : () -> tensor<i8>
    %6 = "stablehlo.pad"(%3#1, %5) <{edge_padding_high = array<i64: 0, 0, 0>, edge_padding_low = array<i64: 0, 0, 0>, interior_padding = array<i64: 0, 0, 0>}> : (tensor<2x4x6xi8>, tensor<i8>) -> tensor<2x4x6xi8>
    %7 = "stablehlo.constant"() <{value = dense<0> : tensor<i8>}> : () -> tensor<i8>
    %8 = "stablehlo.select_and_scatter"(%6, %3#0, %7) <{window_dimensions = array<i64: 2, 2, 2>}> ({
    ^bb0(%arg2: tensor<i8>, %arg3: tensor<i8>):
      %11 = "stablehlo.compare"(%arg2, %arg3) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction GE>}> : (tensor<i8>, tensor<i8>) -> tensor<i1>
      "stablehlo.return"(%11) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %10 = "stablehlo.add"(%arg0, %arg1) : (tensor<i8>, tensor<i8>) -> tensor<i8>
      "stablehlo.return"(%10) : (tensor<i8>) -> ()
    }) : (tensor<2x4x6xi8>, tensor<1x3x5xi8>, tensor<i8>) -> tensor<2x4x6xi8>
    %9 = "stablehlo.slice"(%8) <{limit_indices = array<i64: 2, 4, 6>, start_indices = array<i64: 0, 0, 0>, strides = array<i64: 1, 1, 1>}> : (tensor<2x4x6xi8>) -> tensor<2x4x6xi8>
    "stablehlo.custom_call"(%9, %4) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<2x4x6xi8>, tensor<2x4x6xi8>) -> ()
    "func.return"(%9) : (tensor<2x4x6xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x3x5xi8>, tensor<2x4x6xi8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[-5, 1, 2, -1, 0], [-1, 4, 4, -3, 0], [2, 0, -3, 0, -7]]]> : tensor<1x3x5xi8>}> : () -> tensor<1x3x5xi8>
    %2 = "stablehlo.constant"() <{value = dense<[[[-3, -6, 3, -4, 4, 0], [2, 2, -1, 2, 5, 2], [-1, 3, 5, 5, -1, 3], [0, 0, 1, 3, -1, -2]], [[1, 1, -3, -5, 0, 2], [-3, -4, -6, 2, 6, -1], [-3, 2, -1, 0, 2, 1], [4, 2, 2, -2, 1, 2]]]> : tensor<2x4x6xi8>}> : () -> tensor<2x4x6xi8>
    "func.return"(%1, %2) : (tensor<1x3x5xi8>, tensor<2x4x6xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x4x6xi8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[0, 0, 3, 0, 0, 0], [-5, 0, 0, 0, 0, 0], [0, -1, 5, 0, 0, -7], [0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, -4, 0], [0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0]]]> : tensor<2x4x6xi8>}> : () -> tensor<2x4x6xi8>
    "func.return"(%0) : (tensor<2x4x6xi8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

