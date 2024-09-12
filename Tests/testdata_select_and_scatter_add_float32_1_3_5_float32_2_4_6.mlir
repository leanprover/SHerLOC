"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x4x6xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x3x5xf32>, tensor<2x4x6xf32>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<2x4x6xf32>
    %5 = "stablehlo.constant"() <{value = dense<0x7F800000> : tensor<f32>}> : () -> tensor<f32>
    %6 = "stablehlo.pad"(%3#1, %5) <{edge_padding_high = array<i64: 0, 0, 0>, edge_padding_low = array<i64: 0, 0, 0>, interior_padding = array<i64: 0, 0, 0>}> : (tensor<2x4x6xf32>, tensor<f32>) -> tensor<2x4x6xf32>
    %7 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %8 = "stablehlo.select_and_scatter"(%6, %3#0, %7) <{window_dimensions = array<i64: 2, 2, 2>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %11 = "stablehlo.compare"(%arg2, %arg3) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "stablehlo.return"(%11) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %10 = "stablehlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%10) : (tensor<f32>) -> ()
    }) : (tensor<2x4x6xf32>, tensor<1x3x5xf32>, tensor<f32>) -> tensor<2x4x6xf32>
    %9 = "stablehlo.slice"(%8) <{limit_indices = array<i64: 2, 4, 6>, start_indices = array<i64: 0, 0, 0>, strides = array<i64: 1, 1, 1>}> : (tensor<2x4x6xf32>) -> tensor<2x4x6xf32>
    "stablehlo.custom_call"(%9, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x4x6xf32>, tensor<2x4x6xf32>) -> ()
    "func.return"(%9) : (tensor<2x4x6xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x3x5xf32>, tensor<2x4x6xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[3.68262458, 3.82101059, -2.2947979, 2.8329246, -1.15248716], [3.55029321, 1.37898874, 3.77211261, 2.91995788, 4.76481152], [-0.391007394, 1.91863632, -0.352135569, -2.31635451, 1.65467286]]]> : tensor<1x3x5xf32>}> : () -> tensor<1x3x5xf32>
    %2 = "stablehlo.constant"() <{value = dense<[[[-3.01613498, -1.59692681, 1.84716642, 3.11134791, 1.66431081, 0.179143101], [-7.45665121, -1.83812463, 4.40967798, -1.26709688, 0.945837199, -4.26534128], [-6.25084066, 0.386573404, 3.05768943, 5.35562325, 1.75902784, 6.14879847], [-2.54304743, -0.416835576, -5.58269119, 6.80337095, 5.6900878, -5.19403219]], [[-3.99047351, -0.0503688157, -2.61801505, -1.74413991, 3.63066554, -0.434025705], [2.22170758, -3.62837696, 0.0174580626, 3.51287198, 5.15925407, 0.14256601], [0.16070427, -1.49531329, -2.80724978, -3.98462272, -1.87461674, 2.1211102], [0.858028948, -0.988228261, -3.62104201, -0.313145071, -1.7683264, 2.46107268]]]> : tensor<2x4x6xf32>}> : () -> tensor<2x4x6xf32>
    "func.return"(%1, %2) : (tensor<1x3x5xf32>, tensor<2x4x6xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x4x6xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [7.23291778, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 3.61232424], [-0.391007394, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 1.56650078, 0.000000e+00, 0.000000e+00, 1.65467286]], [[0.000000e+00, 0.000000e+00, -2.2947979, 2.8329246, 0.000000e+00, 0.000000e+00], [0.000000e+00, 5.19999933, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 4.37571621, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]]> : tensor<2x4x6xf32>}> : () -> tensor<2x4x6xf32>
    "func.return"(%0) : (tensor<2x4x6xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

