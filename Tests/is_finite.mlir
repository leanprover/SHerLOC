"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "is_finite_op_test_f64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0xFFF0000000000000, 0x7FF0000000000000, 0x7FF8000000000000, -1.000000e+01, -0.000000e+00, 0.000000e+00, 1.000000e+01]> : tensor<7xf64>}> : () -> tensor<7xf64>
    %1 = "stablehlo.is_finite"(%0) : (tensor<7xf64>) -> tensor<7xi1>
    "check.expect_eq_const"(%1) <{value = dense<[false, false, false, true, true, true, true]> : tensor<7xi1>}> : (tensor<7xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

