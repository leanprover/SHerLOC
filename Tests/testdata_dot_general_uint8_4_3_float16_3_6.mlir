"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xui8>, tensor<3x6xf16>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xf16>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xui8>) -> tensor<4x3xf16>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xf16>) -> tensor<3x6xf16>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xf16>, tensor<3x6xf16>) -> tensor<4x6xf16>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x6xf16>, tensor<4x6xf16>) -> ()
    "func.return"(%7) : (tensor<4x6xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xui8>, tensor<3x6xf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[2, 0, 3], [1, 1, 0], [0, 1, 0], [0, 4, 0]]> : tensor<4x3xui8>}> : () -> tensor<4x3xui8>
    %2 = "stablehlo.constant"() <{value = dense<[[4.367190e+00, -6.535150e+00, 4.644530e+00, 8.007810e-01, 5.117190e+00, -3.431640e+00], [-1.705080e+00, -4.338380e-01, 3.429690e+00, -2.292480e-01, -1.335940e+00, 1.079100e+00], [5.113280e+00, -4.445310e+00, 2.050780e-01, -2.207030e+00, -1.338870e+00, 1.314450e+00]]> : tensor<3x6xf16>}> : () -> tensor<3x6xf16>
    "func.return"(%1, %2) : (tensor<4x3xui8>, tensor<3x6xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[2.407810e+01, -2.640630e+01, 9.906250e+00, -5.019530e+00, 6.218750e+00, -2.919920e+00], [2.662110e+00, -6.968750e+00, 8.078130e+00, 5.712890e-01, 3.781250e+00, -2.351560e+00], [-1.705080e+00, -4.338380e-01, 3.429690e+00, -2.292480e-01, -1.335940e+00, 1.079100e+00], [-6.820310e+00, -1.735350e+00, 1.371880e+01, -9.169920e-01, -5.343750e+00, 4.316410e+00]]> : tensor<4x6xf16>}> : () -> tensor<4x6xf16>
    "func.return"(%0) : (tensor<4x6xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

