"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xui16>, tensor<3x6xbf16>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xbf16>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xui16>) -> tensor<4x3xbf16>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xbf16>) -> tensor<3x6xbf16>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xbf16>, tensor<3x6xbf16>) -> tensor<4x6xbf16>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_almost_eq", has_side_effect = true}> : (tensor<4x6xbf16>, tensor<4x6xbf16>) -> ()
    "func.return"(%7) : (tensor<4x6xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xui16>, tensor<3x6xbf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[2, 2, 3], [1, 2, 1], [4, 3, 4], [1, 4, 1]]> : tensor<4x3xui16>}> : () -> tensor<4x3xui16>
    %2 = "stablehlo.constant"() <{value = dense<[[-1.664060e+00, -6.937500e+00, 3.203130e-01, -1.765630e+00, 1.476560e+00, 1.562500e+00], [-2.765630e+00, 1.468750e+00, 3.476560e-01, 3.398440e-01, 2.218750e+00, 9.843750e-01], [1.531250e+00, -2.296880e+00, -5.781250e-01, -2.171880e+00, 5.562500e+00, -2.796880e+00]]> : tensor<3x6xbf16>}> : () -> tensor<3x6xbf16>
    "func.return"(%1, %2) : (tensor<4x3xui16>, tensor<3x6xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-4.250000e+00, -1.787500e+01, -3.984380e-01, -9.375000e+00, 2.412500e+01, -3.296880e+00], [-5.656250e+00, -6.312500e+00, 4.375000e-01, -3.250000e+00, 1.150000e+01, 7.343750e-01], [-8.812500e+00, -3.250000e+01, 1.171880e-02, -1.475000e+01, 3.475000e+01, -1.984380e+00], [-1.118750e+01, -3.359380e+00, 1.132810e+00, -2.578130e+00, 1.593750e+01, 2.703130e+00]]> : tensor<4x6xbf16>}> : () -> tensor<4x6xbf16>
    "func.return"(%0) : (tensor<4x6xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

