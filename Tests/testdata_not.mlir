"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "not_op_test_si4"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[7, -8, 0]> : tensor<3xi4>}> : () -> tensor<3xi4>
    %1 = "stablehlo.not"(%0) : (tensor<3xi4>) -> tensor<3xi4>
    "check.expect_eq_const"(%1) <{value = dense<[-8, 7, -1]> : tensor<3xi4>}> : (tensor<3xi4>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "not_op_test_ui4"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 7, 15]> : tensor<3xui4>}> : () -> tensor<3xui4>
    %1 = "stablehlo.not"(%0) : (tensor<3xui4>) -> tensor<3xui4>
    "check.expect_eq_const"(%1) <{value = dense<[15, 8, 0]> : tensor<3xui4>}> : (tensor<3xui4>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "not_op_test_si8"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[127, -128, 0]> : tensor<3xi8>}> : () -> tensor<3xi8>
    %1 = "stablehlo.not"(%0) : (tensor<3xi8>) -> tensor<3xi8>
    "check.expect_eq_const"(%1) <{value = dense<[-128, 127, -1]> : tensor<3xi8>}> : (tensor<3xi8>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "not_op_test_ui8"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 127, 255]> : tensor<3xui8>}> : () -> tensor<3xui8>
    %1 = "stablehlo.not"(%0) : (tensor<3xui8>) -> tensor<3xui8>
    "check.expect_eq_const"(%1) <{value = dense<[255, 128, 0]> : tensor<3xui8>}> : (tensor<3xui8>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "not_op_test_si16"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[32767, -32768, 0]> : tensor<3xi16>}> : () -> tensor<3xi16>
    %1 = "stablehlo.not"(%0) : (tensor<3xi16>) -> tensor<3xi16>
    "check.expect_eq_const"(%1) <{value = dense<[-32768, 32767, -1]> : tensor<3xi16>}> : (tensor<3xi16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "not_op_test_ui16"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 32767, 65535]> : tensor<3xui16>}> : () -> tensor<3xui16>
    %1 = "stablehlo.not"(%0) : (tensor<3xui16>) -> tensor<3xui16>
    "check.expect_eq_const"(%1) <{value = dense<[65535, 32768, 0]> : tensor<3xui16>}> : (tensor<3xui16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "not_op_test_si32"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[2147483647, -2147483648, 0]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %1 = "stablehlo.not"(%0) : (tensor<3xi32>) -> tensor<3xi32>
    "check.expect_eq_const"(%1) <{value = dense<[-2147483648, 2147483647, -1]> : tensor<3xi32>}> : (tensor<3xi32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "not_op_test_ui32"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 2147483647, 4294967295]> : tensor<3xui32>}> : () -> tensor<3xui32>
    %1 = "stablehlo.not"(%0) : (tensor<3xui32>) -> tensor<3xui32>
    "check.expect_eq_const"(%1) <{value = dense<[4294967295, 2147483648, 0]> : tensor<3xui32>}> : (tensor<3xui32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "not_op_test_si64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[9223372036854775807, -9223372036854775808, 0]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %1 = "stablehlo.not"(%0) : (tensor<3xi64>) -> tensor<3xi64>
    "check.expect_eq_const"(%1) <{value = dense<[-9223372036854775808, 9223372036854775807, -1]> : tensor<3xi64>}> : (tensor<3xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "not_op_test_ui64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 9223372036854775807, 18446744073709551615]> : tensor<3xui64>}> : () -> tensor<3xui64>
    %1 = "stablehlo.not"(%0) : (tensor<3xui64>) -> tensor<3xui64>
    "check.expect_eq_const"(%1) <{value = dense<[18446744073709551615, 9223372036854775808, 0]> : tensor<3xui64>}> : (tensor<3xui64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "not_op_test_i1"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[false, true]> : tensor<2xi1>}> : () -> tensor<2xi1>
    %1 = "stablehlo.not"(%0) : (tensor<2xi1>) -> tensor<2xi1>
    "check.expect_eq_const"(%1) <{value = dense<[true, false]> : tensor<2xi1>}> : (tensor<2xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "not_op_test_i1_splat_false"}> ({
    %0 = "stablehlo.constant"() <{value = dense<false> : tensor<i1>}> : () -> tensor<i1>
    %1 = "stablehlo.not"(%0) : (tensor<i1>) -> tensor<i1>
    "check.expect_eq_const"(%1) <{value = dense<true> : tensor<i1>}> : (tensor<i1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "not_op_test_i1_splat_true"}> ({
    %0 = "stablehlo.constant"() <{value = dense<true> : tensor<i1>}> : () -> tensor<i1>
    %1 = "stablehlo.not"(%0) : (tensor<i1>) -> tensor<i1>
    "check.expect_eq_const"(%1) <{value = dense<false> : tensor<i1>}> : (tensor<i1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

