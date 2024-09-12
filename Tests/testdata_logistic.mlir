"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "logistic_op_test_f64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>}> : () -> tensor<2x2xf64>
    %1 = "stablehlo.logistic"(%0) : (tensor<2x2xf64>) -> tensor<2x2xf64>
    "check.expect_almost_eq_const"(%1) <{value = dense<[[0.7310585786300049, 0.88079707797788243], [0.95257412682243325, 0.98201379003790845]]> : tensor<2x2xf64>}> : (tensor<2x2xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "logistic_op_test_c128"}> ({
    %0 = "stablehlo.constant"() <{value = dense<(1.000000e+00,2.000000e+00)> : tensor<complex<f64>>}> : () -> tensor<complex<f64>>
    %1 = "stablehlo.logistic"(%0) : (tensor<complex<f64>>) -> tensor<complex<f64>>
    "check.expect_almost_eq_const"(%1) <{value = dense<(1.0214153641721806,0.40343870608154248)> : tensor<complex<f64>>}> : (tensor<complex<f64>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

