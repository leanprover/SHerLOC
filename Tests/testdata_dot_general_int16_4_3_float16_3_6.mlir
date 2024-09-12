"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xi16>, tensor<3x6xf16>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xf16>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xi16>) -> tensor<4x3xf16>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xf16>) -> tensor<3x6xf16>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xf16>, tensor<3x6xf16>) -> tensor<4x6xf16>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x6xf16>, tensor<4x6xf16>) -> ()
    "func.return"(%7) : (tensor<4x6xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xi16>, tensor<3x6xf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[0, 0, 2], [7, -2, -1], [-5, 0, 0], [3, -2, 4]]> : tensor<4x3xi16>}> : () -> tensor<4x3xi16>
    %2 = "stablehlo.constant"() <{value = dense<[[-4.972660e+00, 9.633780e-01, 9.179680e-01, 2.687500e+00, 1.023440e+00, -1.614260e+00], [7.612300e-01, 3.560550e+00, -1.004880e+00, 1.511720e+00, 3.806640e+00, -6.101560e+00], [-5.164060e+00, 5.737300e-01, -3.597660e+00, 4.582030e+00, -2.884770e+00, 1.960940e+00]]> : tensor<3x6xf16>}> : () -> tensor<3x6xf16>
    "func.return"(%1, %2) : (tensor<4x3xi16>, tensor<3x6xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-1.032810e+01, 1.147460e+00, -7.195310e+00, 9.164060e+00, -5.769530e+00, 3.921880e+00], [-3.117190e+01, -9.511710e-01, 1.203130e+01, 1.120310e+01, 2.435550e+00, -1.057620e+00], [2.485940e+01, -4.816410e+00, -4.589840e+00, -1.343750e+01, -5.117190e+00, 8.070310e+00], [-3.709380e+01, -1.935550e+00, -9.625000e+00, 2.337500e+01, -1.607810e+01, 1.520310e+01]]> : tensor<4x6xf16>}> : () -> tensor<4x6xf16>
    "func.return"(%0) : (tensor<4x6xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

