"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xi32>, tensor<3x6xbf16>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xbf16>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xi32>) -> tensor<4x3xbf16>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xbf16>) -> tensor<3x6xbf16>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xbf16>, tensor<3x6xbf16>) -> tensor<4x6xbf16>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x6xbf16>, tensor<4x6xbf16>) -> ()
    "func.return"(%7) : (tensor<4x6xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xi32>, tensor<3x6xbf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[0, -2, 1], [-3, 7, -5], [2, 5, 1], [0, -4, 0]]> : tensor<4x3xi32>}> : () -> tensor<4x3xi32>
    %2 = "stablehlo.constant"() <{value = dense<[[1.664060e+00, -1.382810e+00, -3.812500e+00, 1.523440e+00, -8.867180e-01, 3.457030e-01], [7.656250e-01, -7.375000e+00, 5.718750e+00, 2.203130e+00, 7.343750e-01, -6.738280e-02], [6.875000e-01, -2.832030e-01, -5.531250e+00, 1.187500e+00, -8.359380e-01, -2.078130e+00]]> : tensor<3x6xbf16>}> : () -> tensor<3x6xbf16>
    "func.return"(%1, %2) : (tensor<4x3xi32>, tensor<3x6xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-8.437500e-01, 1.443750e+01, -1.700000e+01, -3.218750e+00, -2.312500e+00, -1.945310e+00], [-3.062500e+00, -4.600000e+01, 7.900000e+01, 4.906250e+00, 1.200000e+01, 8.875000e+00], [7.843750e+00, -4.000000e+01, 1.543750e+01, 1.525000e+01, 1.062500e+00, -1.726560e+00], [-3.062500e+00, 2.950000e+01, -2.287500e+01, -8.812500e+00, -2.937500e+00, 2.695310e-01]]> : tensor<4x6xbf16>}> : () -> tensor<4x6xbf16>
    "func.return"(%0) : (tensor<4x6xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

