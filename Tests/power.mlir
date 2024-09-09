"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "power_op_test_si64"}> ({
    %9 = "stablehlo.constant"() <{value = dense<[-1, -1, -3, 1, -3, 0]> : tensor<6xi64>}> : () -> tensor<6xi64>
    %10 = "stablehlo.constant"() <{value = dense<[1, 0, -3, -3, 3, 2]> : tensor<6xi64>}> : () -> tensor<6xi64>
    %11 = "stablehlo.power"(%9, %10) : (tensor<6xi64>, tensor<6xi64>) -> tensor<6xi64>
    "check.expect_eq_const"(%11) <{value = dense<[-1, 1, 0, 1, -27, 0]> : tensor<6xi64>}> : (tensor<6xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "power_op_test_ui64"}> ({
    %6 = "stablehlo.constant"() <{value = dense<[0, 0, 1, 1, 5]> : tensor<5xui64>}> : () -> tensor<5xui64>
    %7 = "stablehlo.constant"() <{value = dense<[0, 1, 0, 2, 5]> : tensor<5xui64>}> : () -> tensor<5xui64>
    %8 = "stablehlo.power"(%6, %7) : (tensor<5xui64>, tensor<5xui64>) -> tensor<5xui64>
    "check.expect_eq_const"(%8) <{value = dense<[1, 0, 1, 1, 3125]> : tensor<5xui64>}> : (tensor<5xui64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "power_op_test_f64"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[-2.000000e+00, -0.000000e+00, -3.600000e+01, 5.000000e+00, 3.000000e+00, 1.000000e+04]> : tensor<6xf64>}> : () -> tensor<6xf64>
    %4 = "stablehlo.constant"() <{value = dense<[2.000000e+00, 2.000000e+00, 1.100000e+00, 2.000000e+00, -1.000000e+00, 1.000000e+01]> : tensor<6xf64>}> : () -> tensor<6xf64>
    %5 = "stablehlo.power"(%3, %4) : (tensor<6xf64>, tensor<6xf64>) -> tensor<6xf64>
    "check.expect_almost_eq_const"(%5) <{value = dense<[4.000000e+00, 0.000000e+00, 0xFFF8000000000000, 2.500000e+01, 0.33333333333333331, 1.000000e+40]> : tensor<6xf64>}> : (tensor<6xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "power_op_test_c128"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(1.500000e+00,2.500000e+00), (7.500000e+00,5.500000e+00)]> : tensor<2xcomplex<f64>>}> : () -> tensor<2xcomplex<f64>>
    %1 = "stablehlo.constant"() <{value = dense<[(2.500000e+00,1.500000e+00), (5.500000e+00,7.500000e+00)]> : tensor<2xcomplex<f64>>}> : () -> tensor<2xcomplex<f64>>
    %2 = "stablehlo.power"(%0, %1) : (tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>) -> tensor<2xcomplex<f64>>
    "check.expect_almost_eq_const"(%2) <{value = dense<[(-1.5679313814305016,-2.6674775446623613), (392.89270835580857,1801.8249193362644)]> : tensor<2xcomplex<f64>>}> : (tensor<2xcomplex<f64>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

