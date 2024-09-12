"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x5xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<4x6xf16>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x5xf16>
    %4 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f16>}> : () -> tensor<f16>
    %5 = "stablehlo.broadcast_in_dim"(%4) <{broadcast_dimensions = array<i64>}> : (tensor<f16>) -> tensor<f16>
    %6 = "stablehlo.reduce_window"(%2, %5) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<f16>, tensor<f16>) -> tensor<f16>
      "stablehlo.return"(%7) : (tensor<f16>) -> ()
    }) : (tensor<4x6xf16>, tensor<f16>) -> tensor<3x5xf16>
    "stablehlo.custom_call"(%6, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x5xf16>, tensor<3x5xf16>) -> ()
    "func.return"(%6) : (tensor<3x5xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-2.099610e+00, -5.273440e+00, -1.480470e+00, 1.727540e+00, -2.679690e+00, 1.786130e+00], [1.854490e+00, 3.128910e+00, -8.891600e-01, 1.997070e+00, -6.285150e+00, -3.148440e+00], [7.609380e+00, 1.309570e+00, -1.653320e+00, -2.687500e+00, 1.111330e+00, 4.589840e+00], [-3.228520e+00, -2.228520e+00, 5.283200e-01, -3.468750e+00, -4.370120e-01, 1.621090e+00]]> : tensor<4x6xf16>}> : () -> tensor<4x6xf16>
    "func.return"(%1) : (tensor<4x6xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x5xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-2.390630e+00, -4.515630e+00, 1.355470e+00, -5.242190e+00, -1.032810e+01], [1.390630e+01, 1.897460e+00, -3.234380e+00, -5.867190e+00, -3.738280e+00], [3.466800e+00, -2.042970e+00, -7.281250e+00, -5.484380e+00, 6.886710e+00]]> : tensor<3x5xf16>}> : () -> tensor<3x5xf16>
    "func.return"(%0) : (tensor<3x5xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

