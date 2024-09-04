"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "log_op_test_f64"}> ({
    %2 = "stablehlo.constant"() <{value = dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>}> : () -> tensor<2x2xf64>
    %3 = "stablehlo.log"(%2) : (tensor<2x2xf64>) -> tensor<2x2xf64>
    "check.expect_almost_eq_const"(%3) <{value = dense<[[0.000000e+00, 0.69314718055994529], [1.0986122886681098, 1.3862943611198906]]> : tensor<2x2xf64>}> : (tensor<2x2xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "log_op_test_c128"}> ({
    %0 = "stablehlo.constant"() <{value = dense<(1.000000e+00,2.000000e+00)> : tensor<complex<f64>>}> : () -> tensor<complex<f64>>
    %1 = "stablehlo.log"(%0) : (tensor<complex<f64>>) -> tensor<complex<f64>>
    "check.expect_almost_eq_const"(%1) <{value = dense<(0.80471895621705025,1.1071487177940904)> : tensor<complex<f64>>}> : (tensor<complex<f64>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

