"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "exponential_op_test_f64"}> ({
    %2 = "stablehlo.constant"() <{value = dense<[[0.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00]]> : tensor<2x2xf64>}> : () -> tensor<2x2xf64>
    %3 = "stablehlo.exponential"(%2) : (tensor<2x2xf64>) -> tensor<2x2xf64>
    "check.expect_almost_eq_const"(%3) <{value = dense<[[1.000000e+00, 2.7182818284590451], [7.3890560989306504, 20.085536923187668]]> : tensor<2x2xf64>}> : (tensor<2x2xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "exponential_op_test_c128"}> ({
    %0 = "stablehlo.constant"() <{value = dense<(1.000000e+00,2.000000e+00)> : tensor<complex<f64>>}> : () -> tensor<complex<f64>>
    %1 = "stablehlo.exponential"(%0) : (tensor<complex<f64>>) -> tensor<complex<f64>>
    "check.expect_almost_eq_const"(%1) <{value = dense<(-1.1312043837568135,2.4717266720048188)> : tensor<complex<f64>>}> : (tensor<complex<f64>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

