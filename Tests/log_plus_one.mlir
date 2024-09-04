"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "log_plus_one_op_test_f64"}> ({
    %2 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.99899999999999999, 7.000000e+00, 6.3890562099999997, 1.500000e+01]> : tensor<5xf64>}> : () -> tensor<5xf64>
    %3 = "stablehlo.log_plus_one"(%2) : (tensor<5xf64>) -> tensor<5xf64>
    "check.expect_almost_eq_const"(%3) <{value = dense<[0.000000e+00, -6.9077682500000002, 2.0794415499999999, 2.000000e+00, 2.7725887299999998]> : tensor<5xf64>}> : (tensor<5xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "log_plus_one_op_test_c128"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(1.000000e+00,2.000000e+00), (2.000000e+00,1.000000e+00), (0.000000e+00,0.000000e+00)]> : tensor<3xcomplex<f64>>}> : () -> tensor<3xcomplex<f64>>
    %1 = "stablehlo.log_plus_one"(%0) : (tensor<3xcomplex<f64>>) -> tensor<3xcomplex<f64>>
    "check.expect_almost_eq_const"(%1) <{value = dense<[(1.0397207708399101,0.78539816339743995), (1.1512925464970201,0.32175055439664002), (0.000000e+00,0.000000e+00)]> : tensor<3xcomplex<f64>>}> : (tensor<3xcomplex<f64>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

