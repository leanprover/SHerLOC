"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x5xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<4x6xf16>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x5xf16>
    %4 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f16>}> : () -> tensor<f16>
    %5 = "stablehlo.reduce_window"(%2, %4) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %6 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<f16>, tensor<f16>) -> tensor<f16>
      "stablehlo.return"(%6) : (tensor<f16>) -> ()
    }) : (tensor<4x6xf16>, tensor<f16>) -> tensor<3x5xf16>
    "stablehlo.custom_call"(%5, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x5xf16>, tensor<3x5xf16>) -> ()
    "func.return"(%5) : (tensor<3x5xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-2.824220e+00, -2.403560e-01, 3.193360e-01, 2.466800e+00, -1.509770e+00, 4.246090e+00], [-7.609380e+00, -5.324220e+00, -3.806150e-01, -2.015380e-01, 2.214840e+00, 5.929690e+00], [3.134770e+00, 2.857420e+00, 4.218750e+00, -1.056640e+00, 3.454590e-01, -2.564450e+00], [-1.566410e+00, 2.164060e+00, -3.417970e+00, 2.779300e+00, 3.009770e+00, -2.099610e+00]]> : tensor<4x6xf16>}> : () -> tensor<4x6xf16>
    "func.return"(%1) : (tensor<4x6xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x5xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1.000000e+00, 1.000000e+00, 2.466800e+00, 2.466800e+00, 5.929690e+00], [3.134770e+00, 4.218750e+00, 4.218750e+00, 2.214840e+00, 5.929690e+00], [3.134770e+00, 4.218750e+00, 4.218750e+00, 3.009770e+00, 3.009770e+00]]> : tensor<3x5xf16>}> : () -> tensor<3x5xf16>
    "func.return"(%0) : (tensor<3x5xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

