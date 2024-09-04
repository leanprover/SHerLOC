"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "complex_op_test_f64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[1.000000e+00, 3.000000e+00]> : tensor<2xf64>}> : () -> tensor<2xf64>
    %1 = "stablehlo.constant"() <{value = dense<[2.000000e+00, 4.000000e+00]> : tensor<2xf64>}> : () -> tensor<2xf64>
    %2 = "stablehlo.complex"(%0, %1) : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xcomplex<f64>>
    "check.expect_almost_eq_const"(%2) <{value = dense<[(1.000000e+00,2.000000e+00), (3.000000e+00,4.000000e+00)]> : tensor<2xcomplex<f64>>}> : (tensor<2xcomplex<f64>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

