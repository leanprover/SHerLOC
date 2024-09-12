"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "atan2_op_test_f64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, -1.000000e+00]> : tensor<3xf64>}> : () -> tensor<3xf64>
    %1 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<3xf64>}> : () -> tensor<3xf64>
    %2 = "stablehlo.atan2"(%0, %1) : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
    "check.expect_almost_eq_const"(%2) <{value = dense<[0.000000e+00, 1.5707963267948966, -1.5707963267948966]> : tensor<3xf64>}> : (tensor<3xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "atan2_op_test_c128"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(1.000000e+00,0.000000e+00), (-1.000000e+00,0.000000e+00)]> : tensor<2xcomplex<f64>>}> : () -> tensor<2xcomplex<f64>>
    %1 = "stablehlo.constant"() <{value = dense<(0.000000e+00,0.000000e+00)> : tensor<2xcomplex<f64>>}> : () -> tensor<2xcomplex<f64>>
    %2 = "stablehlo.atan2"(%0, %1) : (tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>) -> tensor<2xcomplex<f64>>
    "check.expect_almost_eq_const"(%2) <{value = dense<[(1.5707963267948966,-0.000000e+00), (-1.5707963267948966,0.000000e+00)]> : tensor<2xcomplex<f64>>}> : (tensor<2xcomplex<f64>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

