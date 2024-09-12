"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_si64_default"}> ({
    %0 = "stablehlo.constant"() <{value = dense<-2> : tensor<i64>}> : () -> tensor<i64>
    %1 = "stablehlo.constant"() <{value = dense<-2> : tensor<i64>}> : () -> tensor<i64>
    %2 = "stablehlo.compare"(%0, %1) <{comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
    "check.expect_eq_const"(%2) <{value = dense<true> : tensor<i1>}> : (tensor<i1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_si64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-2, -1, 0, 2, 2]> : tensor<5xi64>}> : () -> tensor<5xi64>
    %1 = "stablehlo.constant"() <{value = dense<[-2, -2, 0, 1, 2]> : tensor<5xi64>}> : () -> tensor<5xi64>
    %2 = "stablehlo.compare"(%0, %1) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<5xi64>, tensor<5xi64>) -> tensor<5xi1>
    "check.expect_eq_const"(%2) <{value = dense<[true, false, true, false, true]> : tensor<5xi1>}> : (tensor<5xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_si64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-2, -1, 0, 2, 2]> : tensor<5xi64>}> : () -> tensor<5xi64>
    %1 = "stablehlo.constant"() <{value = dense<[-2, -2, 0, 1, 2]> : tensor<5xi64>}> : () -> tensor<5xi64>
    %2 = "stablehlo.compare"(%0, %1) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<5xi64>, tensor<5xi64>) -> tensor<5xi1>
    "check.expect_eq_const"(%2) <{value = dense<[false, true, false, true, false]> : tensor<5xi1>}> : (tensor<5xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_si64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-2, -1, 0, 2, 2]> : tensor<5xi64>}> : () -> tensor<5xi64>
    %1 = "stablehlo.constant"() <{value = dense<[-2, -2, 0, 1, 2]> : tensor<5xi64>}> : () -> tensor<5xi64>
    %2 = "stablehlo.compare"(%0, %1) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction GE>}> : (tensor<5xi64>, tensor<5xi64>) -> tensor<5xi1>
    "check.expect_eq_const"(%2) <{value = dense<true> : tensor<5xi1>}> : (tensor<5xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_si64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-2, -1, 0, 2, 2]> : tensor<5xi64>}> : () -> tensor<5xi64>
    %1 = "stablehlo.constant"() <{value = dense<[-2, -2, 0, 1, 2]> : tensor<5xi64>}> : () -> tensor<5xi64>
    %2 = "stablehlo.compare"(%0, %1) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<5xi64>, tensor<5xi64>) -> tensor<5xi1>
    "check.expect_eq_const"(%2) <{value = dense<[false, true, false, true, false]> : tensor<5xi1>}> : (tensor<5xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_si64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-2, -1, 0, 2, 2]> : tensor<5xi64>}> : () -> tensor<5xi64>
    %1 = "stablehlo.constant"() <{value = dense<[-2, -2, 0, 1, 2]> : tensor<5xi64>}> : () -> tensor<5xi64>
    %2 = "stablehlo.compare"(%0, %1) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<5xi64>, tensor<5xi64>) -> tensor<5xi1>
    "check.expect_eq_const"(%2) <{value = dense<[true, false, true, false, true]> : tensor<5xi1>}> : (tensor<5xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_si64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-2, -1, 0, 2, 2]> : tensor<5xi64>}> : () -> tensor<5xi64>
    %1 = "stablehlo.constant"() <{value = dense<[-2, -2, 0, 1, 2]> : tensor<5xi64>}> : () -> tensor<5xi64>
    %2 = "stablehlo.compare"(%0, %1) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<5xi64>, tensor<5xi64>) -> tensor<5xi1>
    "check.expect_eq_const"(%2) <{value = dense<false> : tensor<5xi1>}> : (tensor<5xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_ui64_default"}> ({
    %0 = "stablehlo.constant"() <{value = dense<0> : tensor<ui64>}> : () -> tensor<ui64>
    %1 = "stablehlo.constant"() <{value = dense<0> : tensor<ui64>}> : () -> tensor<ui64>
    %2 = "stablehlo.compare"(%0, %1) <{comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<ui64>, tensor<ui64>) -> tensor<i1>
    "check.expect_eq_const"(%2) <{value = dense<true> : tensor<i1>}> : (tensor<i1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_ui64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 1]> : tensor<2xui64>}> : () -> tensor<2xui64>
    %1 = "stablehlo.constant"() <{value = dense<0> : tensor<2xui64>}> : () -> tensor<2xui64>
    %2 = "stablehlo.compare"(%0, %1) <{compare_type = #stablehlo<comparison_type UNSIGNED>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<2xui64>, tensor<2xui64>) -> tensor<2xi1>
    "check.expect_eq_const"(%2) <{value = dense<[true, false]> : tensor<2xi1>}> : (tensor<2xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_ui64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 1]> : tensor<2xui64>}> : () -> tensor<2xui64>
    %1 = "stablehlo.constant"() <{value = dense<0> : tensor<2xui64>}> : () -> tensor<2xui64>
    %2 = "stablehlo.compare"(%0, %1) <{compare_type = #stablehlo<comparison_type UNSIGNED>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<2xui64>, tensor<2xui64>) -> tensor<2xi1>
    "check.expect_eq_const"(%2) <{value = dense<[false, true]> : tensor<2xi1>}> : (tensor<2xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_ui64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 1]> : tensor<2xui64>}> : () -> tensor<2xui64>
    %1 = "stablehlo.constant"() <{value = dense<0> : tensor<2xui64>}> : () -> tensor<2xui64>
    %2 = "stablehlo.compare"(%0, %1) <{compare_type = #stablehlo<comparison_type UNSIGNED>, comparison_direction = #stablehlo<comparison_direction GE>}> : (tensor<2xui64>, tensor<2xui64>) -> tensor<2xi1>
    "check.expect_eq_const"(%2) <{value = dense<true> : tensor<2xi1>}> : (tensor<2xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_ui64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 1]> : tensor<2xui64>}> : () -> tensor<2xui64>
    %1 = "stablehlo.constant"() <{value = dense<0> : tensor<2xui64>}> : () -> tensor<2xui64>
    %2 = "stablehlo.compare"(%0, %1) <{compare_type = #stablehlo<comparison_type UNSIGNED>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<2xui64>, tensor<2xui64>) -> tensor<2xi1>
    "check.expect_eq_const"(%2) <{value = dense<[false, true]> : tensor<2xi1>}> : (tensor<2xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_ui64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 1]> : tensor<2xui64>}> : () -> tensor<2xui64>
    %1 = "stablehlo.constant"() <{value = dense<0> : tensor<2xui64>}> : () -> tensor<2xui64>
    %2 = "stablehlo.compare"(%0, %1) <{compare_type = #stablehlo<comparison_type UNSIGNED>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<2xui64>, tensor<2xui64>) -> tensor<2xi1>
    "check.expect_eq_const"(%2) <{value = dense<[true, false]> : tensor<2xi1>}> : (tensor<2xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_ui64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 1]> : tensor<2xui64>}> : () -> tensor<2xui64>
    %1 = "stablehlo.constant"() <{value = dense<0> : tensor<2xui64>}> : () -> tensor<2xui64>
    %2 = "stablehlo.compare"(%0, %1) <{compare_type = #stablehlo<comparison_type UNSIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<2xui64>, tensor<2xui64>) -> tensor<2xi1>
    "check.expect_eq_const"(%2) <{value = dense<false> : tensor<2xi1>}> : (tensor<2xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_i1_default"}> ({
    %0 = "stablehlo.constant"() <{value = dense<true> : tensor<i1>}> : () -> tensor<i1>
    %1 = "stablehlo.constant"() <{value = dense<true> : tensor<i1>}> : () -> tensor<i1>
    %2 = "stablehlo.compare"(%0, %1) <{comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<i1>, tensor<i1>) -> tensor<i1>
    "check.expect_eq_const"(%2) <{value = dense<true> : tensor<i1>}> : (tensor<i1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_i1"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[true, true, false, false]> : tensor<4xi1>}> : () -> tensor<4xi1>
    %1 = "stablehlo.constant"() <{value = dense<[true, false, true, false]> : tensor<4xi1>}> : () -> tensor<4xi1>
    %2 = "stablehlo.compare"(%0, %1) <{compare_type = #stablehlo<comparison_type UNSIGNED>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    "check.expect_eq_const"(%2) <{value = dense<[true, false, false, true]> : tensor<4xi1>}> : (tensor<4xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_i1"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[true, true, false, false]> : tensor<4xi1>}> : () -> tensor<4xi1>
    %1 = "stablehlo.constant"() <{value = dense<[true, false, true, false]> : tensor<4xi1>}> : () -> tensor<4xi1>
    %2 = "stablehlo.compare"(%0, %1) <{compare_type = #stablehlo<comparison_type UNSIGNED>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    "check.expect_eq_const"(%2) <{value = dense<[false, true, true, false]> : tensor<4xi1>}> : (tensor<4xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_i1"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[true, true, false, false]> : tensor<4xi1>}> : () -> tensor<4xi1>
    %1 = "stablehlo.constant"() <{value = dense<[true, false, true, false]> : tensor<4xi1>}> : () -> tensor<4xi1>
    %2 = "stablehlo.compare"(%0, %1) <{compare_type = #stablehlo<comparison_type UNSIGNED>, comparison_direction = #stablehlo<comparison_direction GE>}> : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    "check.expect_eq_const"(%2) <{value = dense<[true, true, false, true]> : tensor<4xi1>}> : (tensor<4xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_i1"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[true, true, false, false]> : tensor<4xi1>}> : () -> tensor<4xi1>
    %1 = "stablehlo.constant"() <{value = dense<[true, false, true, false]> : tensor<4xi1>}> : () -> tensor<4xi1>
    %2 = "stablehlo.compare"(%0, %1) <{compare_type = #stablehlo<comparison_type UNSIGNED>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    "check.expect_eq_const"(%2) <{value = dense<[false, true, false, false]> : tensor<4xi1>}> : (tensor<4xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_i1"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[true, true, false, false]> : tensor<4xi1>}> : () -> tensor<4xi1>
    %1 = "stablehlo.constant"() <{value = dense<[true, false, true, false]> : tensor<4xi1>}> : () -> tensor<4xi1>
    %2 = "stablehlo.compare"(%0, %1) <{compare_type = #stablehlo<comparison_type UNSIGNED>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    "check.expect_eq_const"(%2) <{value = dense<[true, false, true, true]> : tensor<4xi1>}> : (tensor<4xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_i1"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[true, true, false, false]> : tensor<4xi1>}> : () -> tensor<4xi1>
    %1 = "stablehlo.constant"() <{value = dense<[true, false, true, false]> : tensor<4xi1>}> : () -> tensor<4xi1>
    %2 = "stablehlo.compare"(%0, %1) <{compare_type = #stablehlo<comparison_type UNSIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    "check.expect_eq_const"(%2) <{value = dense<[false, false, true, false]> : tensor<4xi1>}> : (tensor<4xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_f64_default"}> ({
    %0 = "stablehlo.constant"() <{value = dense<0xFFF0000000000001> : tensor<f64>}> : () -> tensor<f64>
    %1 = "stablehlo.constant"() <{value = dense<0xFFF0000000000001> : tensor<f64>}> : () -> tensor<f64>
    %2 = "stablehlo.compare"(%0, %1) <{comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<f64>, tensor<f64>) -> tensor<i1>
    "check.expect_eq_const"(%2) <{value = dense<false> : tensor<i1>}> : (tensor<i1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_f64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0xFFF0000000000001, 0xFFF0000000000001, 0xFFF0000000000000, 0xFFF0000000000000, -2.000000e+00, -2.000000e+00, -0.000000e+00, -0.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FF0000000000001]> : tensor<14xf64>}> : () -> tensor<14xf64>
    %1 = "stablehlo.constant"() <{value = dense<[0xFFF0000000000001, 0x7FF0000000000001, 0xFFF0000000000000, 0x7FF0000000000000, -2.000000e+00, -1.000000e+00, -0.000000e+00, 0.000000e+00, 0.000000e+00, 2.000000e+00, 2.000000e+00, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FFFFFFFFFFFFFFF]> : tensor<14xf64>}> : () -> tensor<14xf64>
    %2 = "stablehlo.compare"(%0, %1) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<14xf64>, tensor<14xf64>) -> tensor<14xi1>
    "check.expect_eq_const"(%2) <{value = dense<[false, false, true, false, true, false, true, true, true, false, true, true, false, false]> : tensor<14xi1>}> : (tensor<14xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_f64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0xFFF0000000000001, 0xFFF0000000000001, 0xFFF0000000000000, 0xFFF0000000000000, -2.000000e+00, -2.000000e+00, -0.000000e+00, -0.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FF0000000000001]> : tensor<14xf64>}> : () -> tensor<14xf64>
    %1 = "stablehlo.constant"() <{value = dense<[0xFFF0000000000001, 0x7FF0000000000001, 0xFFF0000000000000, 0x7FF0000000000000, -2.000000e+00, -1.000000e+00, -0.000000e+00, 0.000000e+00, 0.000000e+00, 2.000000e+00, 2.000000e+00, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FFFFFFFFFFFFFFF]> : tensor<14xf64>}> : () -> tensor<14xf64>
    %2 = "stablehlo.compare"(%0, %1) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<14xf64>, tensor<14xf64>) -> tensor<14xi1>
    "check.expect_eq_const"(%2) <{value = dense<[true, true, false, true, false, true, false, false, false, true, false, false, true, true]> : tensor<14xi1>}> : (tensor<14xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_f64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0xFFF0000000000001, 0xFFF0000000000001, 0xFFF0000000000000, 0xFFF0000000000000, -2.000000e+00, -2.000000e+00, -0.000000e+00, -0.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FF0000000000001]> : tensor<14xf64>}> : () -> tensor<14xf64>
    %1 = "stablehlo.constant"() <{value = dense<[0xFFF0000000000001, 0x7FF0000000000001, 0xFFF0000000000000, 0x7FF0000000000000, -2.000000e+00, -1.000000e+00, -0.000000e+00, 0.000000e+00, 0.000000e+00, 2.000000e+00, 2.000000e+00, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FFFFFFFFFFFFFFF]> : tensor<14xf64>}> : () -> tensor<14xf64>
    %2 = "stablehlo.compare"(%0, %1) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GE>}> : (tensor<14xf64>, tensor<14xf64>) -> tensor<14xi1>
    "check.expect_eq_const"(%2) <{value = dense<[false, false, true, false, true, false, true, true, true, false, true, true, false, false]> : tensor<14xi1>}> : (tensor<14xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_f64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0xFFF0000000000001, 0xFFF0000000000001, 0xFFF0000000000000, 0xFFF0000000000000, -2.000000e+00, -2.000000e+00, -0.000000e+00, -0.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FF0000000000001]> : tensor<14xf64>}> : () -> tensor<14xf64>
    %1 = "stablehlo.constant"() <{value = dense<[0xFFF0000000000001, 0x7FF0000000000001, 0xFFF0000000000000, 0x7FF0000000000000, -2.000000e+00, -1.000000e+00, -0.000000e+00, 0.000000e+00, 0.000000e+00, 2.000000e+00, 2.000000e+00, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FFFFFFFFFFFFFFF]> : tensor<14xf64>}> : () -> tensor<14xf64>
    %2 = "stablehlo.compare"(%0, %1) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<14xf64>, tensor<14xf64>) -> tensor<14xi1>
    "check.expect_eq_const"(%2) <{value = dense<false> : tensor<14xi1>}> : (tensor<14xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_f64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0xFFF0000000000001, 0xFFF0000000000001, 0xFFF0000000000000, 0xFFF0000000000000, -2.000000e+00, -2.000000e+00, -0.000000e+00, -0.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FF0000000000001]> : tensor<14xf64>}> : () -> tensor<14xf64>
    %1 = "stablehlo.constant"() <{value = dense<[0xFFF0000000000001, 0x7FF0000000000001, 0xFFF0000000000000, 0x7FF0000000000000, -2.000000e+00, -1.000000e+00, -0.000000e+00, 0.000000e+00, 0.000000e+00, 2.000000e+00, 2.000000e+00, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FFFFFFFFFFFFFFF]> : tensor<14xf64>}> : () -> tensor<14xf64>
    %2 = "stablehlo.compare"(%0, %1) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<14xf64>, tensor<14xf64>) -> tensor<14xi1>
    "check.expect_eq_const"(%2) <{value = dense<[false, false, true, true, true, true, true, true, true, true, true, true, false, false]> : tensor<14xi1>}> : (tensor<14xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_f64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0xFFF0000000000001, 0xFFF0000000000001, 0xFFF0000000000000, 0xFFF0000000000000, -2.000000e+00, -2.000000e+00, -0.000000e+00, -0.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FF0000000000001]> : tensor<14xf64>}> : () -> tensor<14xf64>
    %1 = "stablehlo.constant"() <{value = dense<[0xFFF0000000000001, 0x7FF0000000000001, 0xFFF0000000000000, 0x7FF0000000000000, -2.000000e+00, -1.000000e+00, -0.000000e+00, 0.000000e+00, 0.000000e+00, 2.000000e+00, 2.000000e+00, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FFFFFFFFFFFFFFF]> : tensor<14xf64>}> : () -> tensor<14xf64>
    %2 = "stablehlo.compare"(%0, %1) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<14xf64>, tensor<14xf64>) -> tensor<14xi1>
    "check.expect_eq_const"(%2) <{value = dense<[false, false, false, true, false, true, false, false, false, true, false, false, false, false]> : tensor<14xi1>}> : (tensor<14xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_c128_default"}> ({
    %0 = "stablehlo.constant"() <{value = dense<(0x7FF0000000000001,0.000000e+00)> : tensor<complex<f64>>}> : () -> tensor<complex<f64>>
    %1 = "stablehlo.constant"() <{value = dense<(0x7FF0000000000001,-0.000000e+00)> : tensor<complex<f64>>}> : () -> tensor<complex<f64>>
    %2 = "stablehlo.compare"(%0, %1) <{comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<complex<f64>>, tensor<complex<f64>>) -> tensor<i1>
    "check.expect_eq_const"(%2) <{value = dense<false> : tensor<i1>}> : (tensor<i1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_c128"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(0x7FF0000000000001,0.000000e+00), (0.000000e+00,0x7FF0000000000001), (-0.000000e+00,0.000000e+00), (2.000000e+00,2.000000e+00)]> : tensor<4xcomplex<f64>>}> : () -> tensor<4xcomplex<f64>>
    %1 = "stablehlo.constant"() <{value = dense<[(0x7FF0000000000001,-0.000000e+00), (-0.000000e+00,0x7FF0000000000001), (0.000000e+00,0.000000e+00), (2.000000e+00,1.000000e+00)]> : tensor<4xcomplex<f64>>}> : () -> tensor<4xcomplex<f64>>
    %2 = "stablehlo.compare"(%0, %1) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>) -> tensor<4xi1>
    "check.expect_eq_const"(%2) <{value = dense<[false, false, true, false]> : tensor<4xi1>}> : (tensor<4xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_c128"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(0x7FF0000000000001,0.000000e+00), (0.000000e+00,0x7FF0000000000001), (-0.000000e+00,0.000000e+00), (2.000000e+00,2.000000e+00)]> : tensor<4xcomplex<f64>>}> : () -> tensor<4xcomplex<f64>>
    %1 = "stablehlo.constant"() <{value = dense<[(0x7FF0000000000001,-0.000000e+00), (-0.000000e+00,0x7FF0000000000001), (0.000000e+00,0.000000e+00), (2.000000e+00,1.000000e+00)]> : tensor<4xcomplex<f64>>}> : () -> tensor<4xcomplex<f64>>
    %2 = "stablehlo.compare"(%0, %1) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>) -> tensor<4xi1>
    "check.expect_eq_const"(%2) <{value = dense<[true, true, false, true]> : tensor<4xi1>}> : (tensor<4xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

