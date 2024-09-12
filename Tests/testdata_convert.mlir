"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_i1_to_i1"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[true, false]> : tensor<2xi1>}> : () -> tensor<2xi1>
    %1 = "stablehlo.convert"(%0) : (tensor<2xi1>) -> tensor<2xi1>
    "check.expect_eq_const"(%1) <{value = dense<[true, false]> : tensor<2xi1>}> : (tensor<2xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_i1_to_si64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[true, false]> : tensor<2xi1>}> : () -> tensor<2xi1>
    %1 = "stablehlo.convert"(%0) : (tensor<2xi1>) -> tensor<2xi64>
    "check.expect_eq_const"(%1) <{value = dense<[1, 0]> : tensor<2xi64>}> : (tensor<2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_i1_to_ui64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[true, false]> : tensor<2xi1>}> : () -> tensor<2xi1>
    %1 = "stablehlo.convert"(%0) : (tensor<2xi1>) -> tensor<2xui64>
    "check.expect_eq_const"(%1) <{value = dense<[1, 0]> : tensor<2xui64>}> : (tensor<2xui64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_i1_to_f64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[true, false]> : tensor<2xi1>}> : () -> tensor<2xi1>
    %1 = "stablehlo.convert"(%0) : (tensor<2xi1>) -> tensor<2xf64>
    "check.expect_almost_eq_const"(%1) <{value = dense<[1.000000e+00, 0.000000e+00]> : tensor<2xf64>}> : (tensor<2xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_i1_to_c128"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[true, false]> : tensor<2xi1>}> : () -> tensor<2xi1>
    %1 = "stablehlo.convert"(%0) : (tensor<2xi1>) -> tensor<2xcomplex<f64>>
    "check.expect_almost_eq_const"(%1) <{value = dense<[(1.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)]> : tensor<2xcomplex<f64>>}> : (tensor<2xcomplex<f64>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_si64_to_i1"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-1, 0, 1]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %1 = "stablehlo.convert"(%0) : (tensor<3xi64>) -> tensor<3xi1>
    "check.expect_eq_const"(%1) <{value = dense<[true, false, true]> : tensor<3xi1>}> : (tensor<3xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_si64_to_si64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-1, 0, 1]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %1 = "stablehlo.convert"(%0) : (tensor<3xi64>) -> tensor<3xi64>
    "check.expect_eq_const"(%1) <{value = dense<[-1, 0, 1]> : tensor<3xi64>}> : (tensor<3xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_si64_to_ui64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 1]> : tensor<2xi4>}> : () -> tensor<2xi4>
    %1 = "stablehlo.convert"(%0) : (tensor<2xi4>) -> tensor<2xui64>
    "check.expect_eq_const"(%1) <{value = dense<[0, 1]> : tensor<2xui64>}> : (tensor<2xui64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_si64_to_f64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-1, 0, 1]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %1 = "stablehlo.convert"(%0) : (tensor<3xi64>) -> tensor<3xf64>
    "check.expect_almost_eq_const"(%1) <{value = dense<[-1.000000e+00, 0.000000e+00, 1.000000e+00]> : tensor<3xf64>}> : (tensor<3xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_si64_to_c128"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-1, 0, 1]> : tensor<3xi4>}> : () -> tensor<3xi4>
    %1 = "stablehlo.convert"(%0) : (tensor<3xi4>) -> tensor<3xcomplex<f64>>
    "check.expect_almost_eq_const"(%1) <{value = dense<[(-1.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (1.000000e+00,0.000000e+00)]> : tensor<3xcomplex<f64>>}> : (tensor<3xcomplex<f64>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_ui64_to_i1"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 1]> : tensor<2xui64>}> : () -> tensor<2xui64>
    %1 = "stablehlo.convert"(%0) : (tensor<2xui64>) -> tensor<2xi1>
    "check.expect_eq_const"(%1) <{value = dense<[false, true]> : tensor<2xi1>}> : (tensor<2xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_ui64_to_si64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 1]> : tensor<2xui64>}> : () -> tensor<2xui64>
    %1 = "stablehlo.convert"(%0) : (tensor<2xui64>) -> tensor<2xi64>
    "check.expect_eq_const"(%1) <{value = dense<[0, 1]> : tensor<2xi64>}> : (tensor<2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_ui64_to_ui64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 1]> : tensor<2xui4>}> : () -> tensor<2xui4>
    %1 = "stablehlo.convert"(%0) : (tensor<2xui4>) -> tensor<2xui64>
    "check.expect_eq_const"(%1) <{value = dense<[0, 1]> : tensor<2xui64>}> : (tensor<2xui64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_ui64_to_f64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 1]> : tensor<2xui64>}> : () -> tensor<2xui64>
    %1 = "stablehlo.convert"(%0) : (tensor<2xui64>) -> tensor<2xf64>
    "check.expect_almost_eq_const"(%1) <{value = dense<[0.000000e+00, 1.000000e+00]> : tensor<2xf64>}> : (tensor<2xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_ui64_to_c128"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 1]> : tensor<2xui4>}> : () -> tensor<2xui4>
    %1 = "stablehlo.convert"(%0) : (tensor<2xui4>) -> tensor<2xcomplex<f64>>
    "check.expect_almost_eq_const"(%1) <{value = dense<[(0.000000e+00,0.000000e+00), (1.000000e+00,0.000000e+00)]> : tensor<2xcomplex<f64>>}> : (tensor<2xcomplex<f64>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_f64_to_i1"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-1.000000e+00, 0.000000e+00, 1.000000e+00]> : tensor<3xf64>}> : () -> tensor<3xf64>
    %1 = "stablehlo.convert"(%0) : (tensor<3xf64>) -> tensor<3xi1>
    "check.expect_eq_const"(%1) <{value = dense<[true, false, true]> : tensor<3xi1>}> : (tensor<3xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_f64_to_si64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-1.000000e+00, 0.000000e+00, 1.000000e+00]> : tensor<3xf64>}> : () -> tensor<3xf64>
    %1 = "stablehlo.convert"(%0) : (tensor<3xf64>) -> tensor<3xi64>
    "check.expect_eq_const"(%1) <{value = dense<[-1, 0, 1]> : tensor<3xi64>}> : (tensor<3xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_f64_to_ui64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.000000e+00, 1.000000e+00]> : tensor<2xf64>}> : () -> tensor<2xf64>
    %1 = "stablehlo.convert"(%0) : (tensor<2xf64>) -> tensor<2xui64>
    "check.expect_eq_const"(%1) <{value = dense<[0, 1]> : tensor<2xui64>}> : (tensor<2xui64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_f64_to_f64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-1.000000e+00, 0.000000e+00, 1.000000e+00]> : tensor<3xf64>}> : () -> tensor<3xf64>
    %1 = "stablehlo.convert"(%0) : (tensor<3xf64>) -> tensor<3xf64>
    "check.expect_almost_eq_const"(%1) <{value = dense<[-1.000000e+00, 0.000000e+00, 1.000000e+00]> : tensor<3xf64>}> : (tensor<3xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_f64_to_c128"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-1.000000e+00, 0.000000e+00, 1.000000e+00]> : tensor<3xf64>}> : () -> tensor<3xf64>
    %1 = "stablehlo.convert"(%0) : (tensor<3xf64>) -> tensor<3xcomplex<f64>>
    "check.expect_almost_eq_const"(%1) <{value = dense<[(-1.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (1.000000e+00,0.000000e+00)]> : tensor<3xcomplex<f64>>}> : (tensor<3xcomplex<f64>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_c128_to_i1"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(-1.000000e+00,0.000000e+00), (0.000000e+00,1.000000e+00), (1.000000e+00,0.000000e+00)]> : tensor<3xcomplex<f64>>}> : () -> tensor<3xcomplex<f64>>
    %1 = "stablehlo.convert"(%0) : (tensor<3xcomplex<f64>>) -> tensor<3xi1>
    "check.expect_eq_const"(%1) <{value = dense<[true, false, true]> : tensor<3xi1>}> : (tensor<3xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_c128_to_si64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(-1.000000e+00,0.000000e+00), (0.000000e+00,1.000000e+00), (1.000000e+00,0.000000e+00)]> : tensor<3xcomplex<f64>>}> : () -> tensor<3xcomplex<f64>>
    %1 = "stablehlo.convert"(%0) : (tensor<3xcomplex<f64>>) -> tensor<3xi64>
    "check.expect_eq_const"(%1) <{value = dense<[-1, 0, 1]> : tensor<3xi64>}> : (tensor<3xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_c128_to_ui64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(0.000000e+00,1.000000e+00), (1.000000e+00,0.000000e+00)]> : tensor<2xcomplex<f64>>}> : () -> tensor<2xcomplex<f64>>
    %1 = "stablehlo.convert"(%0) : (tensor<2xcomplex<f64>>) -> tensor<2xui64>
    "check.expect_eq_const"(%1) <{value = dense<[0, 1]> : tensor<2xui64>}> : (tensor<2xui64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_c128_to_f64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(-1.000000e+00,0.000000e+00), (0.000000e+00,1.000000e+00), (1.000000e+00,0.000000e+00)]> : tensor<3xcomplex<f64>>}> : () -> tensor<3xcomplex<f64>>
    %1 = "stablehlo.convert"(%0) : (tensor<3xcomplex<f64>>) -> tensor<3xf64>
    "check.expect_almost_eq_const"(%1) <{value = dense<[-1.000000e+00, 0.000000e+00, 1.000000e+00]> : tensor<3xf64>}> : (tensor<3xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_c128_to_c128"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(-1.000000e+00,0.000000e+00), (0.000000e+00,1.000000e+00), (1.000000e+00,0.000000e+00)]> : tensor<3xcomplex<f64>>}> : () -> tensor<3xcomplex<f64>>
    %1 = "stablehlo.convert"(%0) : (tensor<3xcomplex<f64>>) -> tensor<3xcomplex<f64>>
    "check.expect_almost_eq_const"(%1) <{value = dense<[(-1.000000e+00,0.000000e+00), (0.000000e+00,1.000000e+00), (1.000000e+00,0.000000e+00)]> : tensor<3xcomplex<f64>>}> : (tensor<3xcomplex<f64>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

