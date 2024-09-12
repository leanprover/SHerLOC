"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x4x6xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x3x4xf32>, tensor<2x4x6xf32>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<2x4x6xf32>
    %5 = "stablehlo.constant"() <{value = dense<0xFF800000> : tensor<f32>}> : () -> tensor<f32>
    %6 = "stablehlo.pad"(%3#1, %5) <{edge_padding_high = array<i64: 0, 0, 0>, edge_padding_low = array<i64: 0, 0, 0>, interior_padding = array<i64: 0, 0, 0>}> : (tensor<2x4x6xf32>, tensor<f32>) -> tensor<2x4x6xf32>
    %7 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %8 = "stablehlo.select_and_scatter"(%6, %3#0, %7) <{window_dimensions = array<i64: 1, 2, 3>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %11 = "stablehlo.compare"(%arg2, %arg3) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GE>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "stablehlo.return"(%11) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %10 = "stablehlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%10) : (tensor<f32>) -> ()
    }) : (tensor<2x4x6xf32>, tensor<2x3x4xf32>, tensor<f32>) -> tensor<2x4x6xf32>
    %9 = "stablehlo.slice"(%8) <{limit_indices = array<i64: 2, 4, 6>, start_indices = array<i64: 0, 0, 0>, strides = array<i64: 1, 1, 1>}> : (tensor<2x4x6xf32>) -> tensor<2x4x6xf32>
    "stablehlo.custom_call"(%9, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x4x6xf32>, tensor<2x4x6xf32>) -> ()
    "func.return"(%9) : (tensor<2x4x6xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x3x4xf32>, tensor<2x4x6xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[-0.141975358, 0.899714291, 2.56063509, 4.19124699], [-3.65281343, -1.86750543, -4.47070408, -2.39011383], [-6.062790e+00, 2.59964061, -0.644573092, -2.90732551]], [[-0.199792087, 6.34653139, 6.94955348, 4.45443439], [2.708630e+00, 1.3340112, 4.9623251, -1.93976092], [0.264342725, 3.3276751, 2.26464963, 1.62383819]]]> : tensor<2x3x4xf32>}> : () -> tensor<2x3x4xf32>
    %2 = "stablehlo.constant"() <{value = dense<[[[-4.83879852, 0.13970837, 1.20698881, 0.763655662, -1.41355038, -0.808346152], [4.13756514, -2.28614664, -2.28975868, 1.78466904, 0.397792876, 0.802912414], [1.32220972, -3.08647823, 1.04847288, -0.915023326, -0.125590891, 1.63029754], [3.96157026, 5.32061863, -1.98978925, -5.22456074, 0.259640813, -3.288920e+00]], [[-0.2864438, -0.0922905281, -1.75957131, -2.51840854, 2.27084374, -1.36041808], [0.440647334, -1.8337611, -2.76243925, 1.22662556, -5.52954626, 1.53385413], [-0.209350288, -1.79192138, -3.08719182, 1.85777771, 1.66460097, 3.35604095], [0.884390711, -3.70861816, 0.817521929, -0.586911082, 0.451280475, -2.13313031]]]> : tensor<2x4x6xf32>}> : () -> tensor<2x4x6xf32>
    "func.return"(%1, %2) : (tensor<2x3x4xf32>, tensor<2x4x6xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x4x6xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [-3.79478884, 0.000000e+00, 0.000000e+00, -1.07672739, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, -0.644573092, 0.000000e+00, 0.000000e+00, -2.90732551], [0.000000e+00, -3.46314931, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 11.4039879, 0.000000e+00], [2.50883794, 0.000000e+00, 0.000000e+00, 6.34653139, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 11.8886604, 0.000000e+00, -0.315922737], [0.264342725, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]]> : tensor<2x4x6xf32>}> : () -> tensor<2x4x6xf32>
    "func.return"(%0) : (tensor<2x4x6xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

