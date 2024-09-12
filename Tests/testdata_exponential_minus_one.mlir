"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "exponential_minus_one_op_test_f64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.000000e+00, 1.000000e+00]> : tensor<2xf64>}> : () -> tensor<2xf64>
    %1 = "stablehlo.exponential_minus_one"(%0) : (tensor<2xf64>) -> tensor<2xf64>
    "check.expect_almost_eq_const"(%1) <{value = dense<[0.000000e+00, 1.7182818284590451]> : tensor<2xf64>}> : (tensor<2xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "exponential_minus_one_op_test_c128"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(1.000000e+00,2.000000e+00), (2.000000e+00,1.000000e+00), (0.000000e+00,0.000000e+00)]> : tensor<3xcomplex<f64>>}> : () -> tensor<3xcomplex<f64>>
    %1 = "stablehlo.exponential_minus_one"(%0) : (tensor<3xcomplex<f64>>) -> tensor<3xcomplex<f64>>
    "check.expect_almost_eq_const"(%1) <{value = dense<[(-2.1312043837568138,2.4717266720048188), (2.9923240484412714,6.2176763123679679), (0.000000e+00,0.000000e+00)]> : tensor<3xcomplex<f64>>}> : (tensor<3xcomplex<f64>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

