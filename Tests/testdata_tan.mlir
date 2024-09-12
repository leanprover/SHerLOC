"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "tan_op_test_f64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.000000e+00, 1.5707963199999999, 3.1415926500000002, 4.7123889800000001]> : tensor<4xf64>}> : () -> tensor<4xf64>
    %1 = "stablehlo.tan"(%0) : (tensor<4xf64>) -> tensor<4xf64>
    "check.expect_almost_eq_const"(%1) <{value = dense<[0.000000e+00, 147169271.76124874, -3.5897930298416118E-9, 2599497068.2695704]> : tensor<4xf64>}> : (tensor<4xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

