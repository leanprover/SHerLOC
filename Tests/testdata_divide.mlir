"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "divide_op_test_si64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[17, -17, 17, -17]> : tensor<4xi64>}> : () -> tensor<4xi64>
    %1 = "stablehlo.constant"() <{value = dense<[3, 3, -3, -3]> : tensor<4xi64>}> : () -> tensor<4xi64>
    %2 = "stablehlo.divide"(%0, %1) : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi64>
    "check.expect_eq_const"(%2) <{value = dense<[5, -5, -5, 5]> : tensor<4xi64>}> : (tensor<4xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "divide_op_test_ui64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[17, 18, 19, 20]> : tensor<4xui64>}> : () -> tensor<4xui64>
    %1 = "stablehlo.constant"() <{value = dense<[3, 4, 5, 7]> : tensor<4xui64>}> : () -> tensor<4xui64>
    %2 = "stablehlo.divide"(%0, %1) : (tensor<4xui64>, tensor<4xui64>) -> tensor<4xui64>
    "check.expect_eq_const"(%2) <{value = dense<[5, 4, 3, 2]> : tensor<4xui64>}> : (tensor<4xui64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "divide_op_test_f64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[1.710000e+01, -1.710000e+01, 1.710000e+01, -1.710000e+01]> : tensor<4xf64>}> : () -> tensor<4xf64>
    %1 = "stablehlo.constant"() <{value = dense<[3.000000e+00, 3.000000e+00, -3.000000e+00, -3.000000e+00]> : tensor<4xf64>}> : () -> tensor<4xf64>
    %2 = "stablehlo.divide"(%0, %1) : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
    "check.expect_almost_eq_const"(%2) <{value = dense<[5.700000e+00, -5.700000e+00, -5.700000e+00, 5.700000e+00]> : tensor<4xf64>}> : (tensor<4xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "divide_op_test_c128"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(1.500000e+00,2.500000e+00), (7.500000e+00,5.500000e+00)]> : tensor<2xcomplex<f64>>}> : () -> tensor<2xcomplex<f64>>
    %1 = "stablehlo.constant"() <{value = dense<[(2.500000e+00,1.500000e+00), (5.500000e+00,7.500000e+00)]> : tensor<2xcomplex<f64>>}> : () -> tensor<2xcomplex<f64>>
    %2 = "stablehlo.divide"(%0, %1) : (tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>) -> tensor<2xcomplex<f64>>
    "check.expect_almost_eq_const"(%2) <{value = dense<[(0.88235294117647056,0.4705882352941177), (0.95375722543352603,-0.30057803468208094)]> : tensor<2xcomplex<f64>>}> : (tensor<2xcomplex<f64>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

