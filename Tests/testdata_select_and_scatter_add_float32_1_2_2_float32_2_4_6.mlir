"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x4x6xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x2x2xf32>, tensor<2x4x6xf32>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<2x4x6xf32>
    %5 = "stablehlo.constant"() <{value = dense<0xFF800000> : tensor<f32>}> : () -> tensor<f32>
    %6 = "stablehlo.pad"(%3#1, %5) <{edge_padding_high = array<i64: 0, 0, 0>, edge_padding_low = array<i64: 0, 0, 0>, interior_padding = array<i64: 0, 0, 0>}> : (tensor<2x4x6xf32>, tensor<f32>) -> tensor<2x4x6xf32>
    %7 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %8 = "stablehlo.select_and_scatter"(%6, %3#0, %7) <{window_dimensions = array<i64: 2, 2, 2>, window_strides = array<i64: 1, 2, 3>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %11 = "stablehlo.compare"(%arg2, %arg3) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GE>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "stablehlo.return"(%11) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %10 = "stablehlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%10) : (tensor<f32>) -> ()
    }) : (tensor<2x4x6xf32>, tensor<1x2x2xf32>, tensor<f32>) -> tensor<2x4x6xf32>
    %9 = "stablehlo.slice"(%8) <{limit_indices = array<i64: 2, 4, 6>, start_indices = array<i64: 0, 0, 0>, strides = array<i64: 1, 1, 1>}> : (tensor<2x4x6xf32>) -> tensor<2x4x6xf32>
    "stablehlo.custom_call"(%9, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x4x6xf32>, tensor<2x4x6xf32>) -> ()
    "func.return"(%9) : (tensor<2x4x6xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x2x2xf32>, tensor<2x4x6xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[-4.74842882, -3.46046281], [-5.42686605, 0.153474241]]]> : tensor<1x2x2xf32>}> : () -> tensor<1x2x2xf32>
    %2 = "stablehlo.constant"() <{value = dense<[[[-2.15483785, -1.2998265, -2.06859159, 2.06262922, -5.04161739, -2.80120754], [-0.731763124, -0.849812805, 1.3785845, -3.86032128, 4.344930e+00, -0.56643635], [-4.96906567, -4.41274834, 0.0296598822, -0.221065283, 2.1941483, -3.0988338], [-6.20206594, -2.26923609, -3.92311668, -5.02652597, 0.513001502, -3.97359276]], [[-6.76744461, 0.523507416, 1.75738323, 1.93691206, -1.67728722, 0.715135514], [-4.50445032, 0.349508524, -2.15930676, 2.86179519, -1.65786111, -0.0169575941], [2.90006447, 1.82748401, -0.0447356291, 5.61337233, -1.01589382, 2.75839496], [-3.58221126, -3.38783431, 2.8053546, 5.55537176, 0.202345371, 2.61038017]]]> : tensor<2x4x6xf32>}> : () -> tensor<2x4x6xf32>
    "func.return"(%1, %2) : (tensor<1x2x2xf32>, tensor<2x4x6xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x4x6xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, -3.46046281, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]], [[0.000000e+00, -4.74842882, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [-5.42686605, 0.000000e+00, 0.000000e+00, 0.153474241, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]]> : tensor<2x4x6xf32>}> : () -> tensor<2x4x6xf32>
    "func.return"(%0) : (tensor<2x4x6xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

