"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "tuple"}> ({
    %7 = "stablehlo.constant"() <{value = dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>}> : () -> tensor<2xf64>
    %8 = "stablehlo.constant"() <{value = dense<3> : tensor<i64>}> : () -> tensor<i64>
    %9 = "stablehlo.tuple"(%8) : (tensor<i64>) -> tuple<tensor<i64>>
    %10 = "stablehlo.tuple"(%7, %9) : (tensor<2xf64>, tuple<tensor<i64>>) -> tuple<tensor<2xf64>, tuple<tensor<i64>>>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "get_tuple_element"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>}> : () -> tensor<2xf64>
    %1 = "stablehlo.constant"() <{value = dense<3> : tensor<i64>}> : () -> tensor<i64>
    %2 = "stablehlo.tuple"(%1) : (tensor<i64>) -> tuple<tensor<i64>>
    %3 = "stablehlo.tuple"(%0, %2) : (tensor<2xf64>, tuple<tensor<i64>>) -> tuple<tensor<2xf64>, tuple<tensor<i64>>>
    %4 = "stablehlo.get_tuple_element"(%3) <{index = 0 : i32}> : (tuple<tensor<2xf64>, tuple<tensor<i64>>>) -> tensor<2xf64>
    %5 = "stablehlo.get_tuple_element"(%3) <{index = 1 : i32}> : (tuple<tensor<2xf64>, tuple<tensor<i64>>>) -> tuple<tensor<i64>>
    %6 = "stablehlo.get_tuple_element"(%5) <{index = 0 : i32}> : (tuple<tensor<i64>>) -> tensor<i64>
    "check.expect_almost_eq_const"(%4) <{value = dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>}> : (tensor<2xf64>) -> ()
    "check.expect_eq_const"(%6) <{value = dense<3> : tensor<i64>}> : (tensor<i64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

