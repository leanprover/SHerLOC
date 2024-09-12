"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "reduce_precision_op_test_f64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0x7FF0000000000000, 0x7FFFFFFFFFFFFFFF, 4.940660e-324, 0.000000e+00, 6.550500e+04, 6.552000e+04]> : tensor<6xf64>}> : () -> tensor<6xf64>
    %1 = "stablehlo.reduce_precision"(%0) <{exponent_bits = 5 : i32, mantissa_bits = 10 : i32}> : (tensor<6xf64>) -> tensor<6xf64>
    "check.expect_almost_eq_const"(%1) <{value = dense<[0x7FF0000000000000, 0x7FFFFFFFFFFFFFFF, 0.000000e+00, 0.000000e+00, 6.550400e+04, 0x7FF0000000000000]> : tensor<6xf64>}> : (tensor<6xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "reduce_precision_op_test_f64_zero_mantissa_bits"}> ({
    %0 = "stablehlo.constant"() <{value = dense<0x7FFFFFFFFFFFFFFF> : tensor<f64>}> : () -> tensor<f64>
    %1 = "stablehlo.reduce_precision"(%0) <{exponent_bits = 5 : i32, mantissa_bits = 0 : i32}> : (tensor<f64>) -> tensor<f64>
    "check.expect_almost_eq_const"(%1) <{value = dense<0x7FF0000000000000> : tensor<f64>}> : (tensor<f64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

