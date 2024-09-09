"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "sign_op_test_si64"}> ({
    %4 = "stablehlo.constant"() <{value = dense<[-1, 0, 1]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %5 = "stablehlo.sign"(%4) : (tensor<3xi64>) -> tensor<3xi64>
    "check.expect_eq_const"(%5) <{value = dense<[-1, 0, 1]> : tensor<3xi64>}> : (tensor<3xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "sign_op_test_f64"}> ({
    %2 = "stablehlo.constant"() <{value = dense<[0x7FFFFFFFFFFFFFFF, -1.000000e+00, -0.000000e+00, 0.000000e+00, 1.000000e+00]> : tensor<5xf64>}> : () -> tensor<5xf64>
    %3 = "stablehlo.sign"(%2) : (tensor<5xf64>) -> tensor<5xf64>
    "check.expect_almost_eq_const"(%3) <{value = dense<[0x7FFFFFFFFFFFFFFF, -1.000000e+00, -0.000000e+00, 0.000000e+00, 1.000000e+00]> : tensor<5xf64>}> : (tensor<5xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "sign_op_test_c128"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(0x7FF0000000000001,0.000000e+00), (0.000000e+00,0x7FF0000000000001), (0.000000e+00,0.000000e+00), (0.000000e+00,1.000000e+00)]> : tensor<4xcomplex<f64>>}> : () -> tensor<4xcomplex<f64>>
    %1 = "stablehlo.sign"(%0) : (tensor<4xcomplex<f64>>) -> tensor<4xcomplex<f64>>
    "check.expect_almost_eq_const"(%1) <{value = dense<[(0x7FF0000000000001,0x7FF0000000000001), (0x7FF0000000000001,0x7FF0000000000001), (0.000000e+00,0.000000e+00), (0.000000e+00,1.000000e+00)]> : tensor<4xcomplex<f64>>}> : (tensor<4xcomplex<f64>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

