"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_c128"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(0x7FF0000000000001,0.000000e+00), (0.000000e+00,0x7FF0000000000001), (-0.000000e+00,0.000000e+00), (2.000000e+00,2.000000e+00)]> : tensor<4xcomplex<f64>>}> : () -> tensor<4xcomplex<f64>>
    %1 = "stablehlo.constant"() <{value = dense<[(0x7FF0000000000001,-0.000000e+00), (-0.000000e+00,0x7FF0000000000001), (0.000000e+00,0.000000e+00), (2.000000e+00,1.000000e+00)]> : tensor<4xcomplex<f64>>}> : () -> tensor<4xcomplex<f64>>
    %2 = "stablehlo.compare"(%0, %1) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>) -> tensor<4xi1>
    "check.expect_eq_const"(%2) <{value = dense<[false, false, true, false]> : tensor<4xi1>}> : (tensor<4xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

