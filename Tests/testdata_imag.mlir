"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "imag_op_test_f64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>}> : () -> tensor<2xf64>
    %1 = "stablehlo.imag"(%0) : (tensor<2xf64>) -> tensor<2xf64>
    "check.expect_almost_eq_const"(%1) <{value = dense<0.000000e+00> : tensor<2xf64>}> : (tensor<2xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "imag_op_test_c128"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(1.000000e+00,2.000000e+00), (3.000000e+00,4.000000e+00)]> : tensor<2xcomplex<f64>>}> : () -> tensor<2xcomplex<f64>>
    %1 = "stablehlo.imag"(%0) : (tensor<2xcomplex<f64>>) -> tensor<2xf64>
    "check.expect_almost_eq_const"(%1) <{value = dense<[2.000000e+00, 4.000000e+00]> : tensor<2xf64>}> : (tensor<2xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

