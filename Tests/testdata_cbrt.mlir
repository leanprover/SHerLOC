"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "cbrt_op_test_f64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 8.000000e+00, 2.700000e+01]> : tensor<4xf64>}> : () -> tensor<4xf64>
    %1 = "stablehlo.cbrt"(%0) : (tensor<4xf64>) -> tensor<4xf64>
    "check.expect_almost_eq_const"(%1) <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<4xf64>}> : (tensor<4xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "cbrt_op_test_c128"}> ({
    %0 = "stablehlo.constant"() <{value = dense<(3.000000e+00,4.000000e+00)> : tensor<complex<f64>>}> : () -> tensor<complex<f64>>
    %1 = "stablehlo.cbrt"(%0) : (tensor<complex<f64>>) -> tensor<complex<f64>>
    "check.expect_almost_eq_const"(%1) <{value = dense<(1.628937145922176,0.52017450230454576)> : tensor<complex<f64>>}> : (tensor<complex<f64>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

