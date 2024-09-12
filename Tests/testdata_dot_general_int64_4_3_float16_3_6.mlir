"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xi64>, tensor<3x6xf16>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xf16>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xi64>) -> tensor<4x3xf16>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xf16>) -> tensor<3x6xf16>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xf16>, tensor<3x6xf16>) -> tensor<4x6xf16>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x6xf16>, tensor<4x6xf16>) -> ()
    "func.return"(%7) : (tensor<4x6xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xi64>, tensor<3x6xf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[0, -2, -7], [-1, -2, 5], [1, 4, 1], [-6, 2, -1]]> : tensor<4x3xi64>}> : () -> tensor<4x3xi64>
    %2 = "stablehlo.constant"() <{value = dense<[[1.152340e+00, -4.539060e+00, -6.997070e-01, 2.373050e+00, -2.224610e+00, 1.533200e+00], [-3.447270e+00, 8.027340e-01, -2.769530e+00, -8.725580e-01, -4.027340e+00, -3.121090e+00], [1.754880e+00, -2.937500e+00, 2.408200e+00, 3.320310e+00, 2.689450e+00, 4.411620e-01]]> : tensor<3x6xf16>}> : () -> tensor<3x6xf16>
    "func.return"(%1, %2) : (tensor<4x3xi64>, tensor<3x6xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-5.390630e+00, 1.895310e+01, -1.132030e+01, -2.150000e+01, -1.077340e+01, 3.154300e+00], [1.451560e+01, -1.175000e+01, 1.828130e+01, 1.597660e+01, 2.371880e+01, 6.914060e+00], [-1.088280e+01, -4.265630e+00, -9.367180e+00, 2.203130e+00, -1.564060e+01, -1.050780e+01], [-1.556250e+01, 3.178130e+01, -3.750000e+00, -1.929690e+01, 2.603520e+00, -1.588280e+01]]> : tensor<4x6xf16>}> : () -> tensor<4x6xf16>
    "func.return"(%0) : (tensor<4x6xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

