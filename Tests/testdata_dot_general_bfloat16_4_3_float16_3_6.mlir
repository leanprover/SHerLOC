"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xbf16>, tensor<3x6xf16>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xbf16>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xbf16>) -> tensor<4x3xbf16>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xf16>) -> tensor<3x6xbf16>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xbf16>, tensor<3x6xbf16>) -> tensor<4x6xbf16>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x6xbf16>, tensor<4x6xbf16>) -> ()
    "func.return"(%7) : (tensor<4x6xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xbf16>, tensor<3x6xf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-4.711910e-02, 6.367190e-01, -3.027340e-01], [3.203130e+00, 4.199220e-01, 3.062500e+00], [2.203130e+00, 1.031250e+00, 1.632810e+00], [4.003910e-01, 4.375000e+00, -2.171880e+00]]> : tensor<4x3xbf16>}> : () -> tensor<4x3xbf16>
    %2 = "stablehlo.constant"() <{value = dense<[[-6.925780e+00, -4.007810e+00, 2.437740e-01, -5.703130e+00, 5.039060e+00, -1.204100e+00], [-1.289060e+00, 4.794920e-01, -1.965820e+00, -6.714840e+00, -2.487180e-03, -5.101560e+00], [-2.066410e+00, 1.361330e+00, 3.738280e+00, 6.655270e-01, 4.648440e+00, -4.504390e-01]]> : tensor<3x6xf16>}> : () -> tensor<3x6xf16>
    "func.return"(%1, %2) : (tensor<4x3xbf16>, tensor<3x6xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1.308590e-01, 8.300780e-02, -2.390630e+00, -4.218750e+00, -1.648440e+00, -3.046880e+00], [-2.912500e+01, -8.437500e+00, 1.137500e+01, -1.900000e+01, 3.037500e+01, -7.375000e+00], [-2.000000e+01, -6.093750e+00, 4.593750e+00, -1.837500e+01, 1.862500e+01, -8.625000e+00], [-3.937500e+00, -2.453130e+00, -1.662500e+01, -3.300000e+01, -8.125000e+00, -2.175000e+01]]> : tensor<4x6xbf16>}> : () -> tensor<4x6xbf16>
    "func.return"(%0) : (tensor<4x6xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

