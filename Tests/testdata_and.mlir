"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "and_op_test_si4"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[7, -8, -8]> : tensor<3xi4>}> : () -> tensor<3xi4>
    %1 = "stablehlo.constant"() <{value = dense<[0, 7, -8]> : tensor<3xi4>}> : () -> tensor<3xi4>
    %2 = "stablehlo.and"(%0, %1) : (tensor<3xi4>, tensor<3xi4>) -> tensor<3xi4>
    "check.expect_eq_const"(%2) <{value = dense<[0, 0, -8]> : tensor<3xi4>}> : (tensor<3xi4>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "and_op_test_ui4"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 7, 15]> : tensor<3xui4>}> : () -> tensor<3xui4>
    %1 = "stablehlo.constant"() <{value = dense<15> : tensor<3xui4>}> : () -> tensor<3xui4>
    %2 = "stablehlo.and"(%0, %1) : (tensor<3xui4>, tensor<3xui4>) -> tensor<3xui4>
    "check.expect_eq_const"(%2) <{value = dense<[0, 7, 15]> : tensor<3xui4>}> : (tensor<3xui4>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "and_op_test_si8"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[127, -128, -128]> : tensor<3xi8>}> : () -> tensor<3xi8>
    %1 = "stablehlo.constant"() <{value = dense<[0, 127, -128]> : tensor<3xi8>}> : () -> tensor<3xi8>
    %2 = "stablehlo.and"(%0, %1) : (tensor<3xi8>, tensor<3xi8>) -> tensor<3xi8>
    "check.expect_eq_const"(%2) <{value = dense<[0, 0, -128]> : tensor<3xi8>}> : (tensor<3xi8>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "and_op_test_ui8"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 127, 255]> : tensor<3xui8>}> : () -> tensor<3xui8>
    %1 = "stablehlo.constant"() <{value = dense<255> : tensor<3xui8>}> : () -> tensor<3xui8>
    %2 = "stablehlo.and"(%0, %1) : (tensor<3xui8>, tensor<3xui8>) -> tensor<3xui8>
    "check.expect_eq_const"(%2) <{value = dense<[0, 127, 255]> : tensor<3xui8>}> : (tensor<3xui8>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "and_op_test_si16"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[32767, -32768, -32768]> : tensor<3xi16>}> : () -> tensor<3xi16>
    %1 = "stablehlo.constant"() <{value = dense<[0, 32767, -32768]> : tensor<3xi16>}> : () -> tensor<3xi16>
    %2 = "stablehlo.and"(%0, %1) : (tensor<3xi16>, tensor<3xi16>) -> tensor<3xi16>
    "check.expect_eq_const"(%2) <{value = dense<[0, 0, -32768]> : tensor<3xi16>}> : (tensor<3xi16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "and_op_test_ui16"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 32767, 65535]> : tensor<3xui16>}> : () -> tensor<3xui16>
    %1 = "stablehlo.constant"() <{value = dense<65535> : tensor<3xui16>}> : () -> tensor<3xui16>
    %2 = "stablehlo.and"(%0, %1) : (tensor<3xui16>, tensor<3xui16>) -> tensor<3xui16>
    "check.expect_eq_const"(%2) <{value = dense<[0, 32767, 65535]> : tensor<3xui16>}> : (tensor<3xui16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "and_op_test_si32"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[2147483647, -2147483648, -2147483648]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %1 = "stablehlo.constant"() <{value = dense<[0, 2147483647, -2147483648]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %2 = "stablehlo.and"(%0, %1) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
    "check.expect_eq_const"(%2) <{value = dense<[0, 0, -2147483648]> : tensor<3xi32>}> : (tensor<3xi32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "and_op_test_ui32"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 2147483647, 4294967295]> : tensor<3xui32>}> : () -> tensor<3xui32>
    %1 = "stablehlo.constant"() <{value = dense<4294967295> : tensor<3xui32>}> : () -> tensor<3xui32>
    %2 = "stablehlo.and"(%0, %1) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    "check.expect_eq_const"(%2) <{value = dense<[0, 2147483647, 4294967295]> : tensor<3xui32>}> : (tensor<3xui32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "and_op_test_si64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[9223372036854775807, -9223372036854775808, -9223372036854775808]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %1 = "stablehlo.constant"() <{value = dense<[0, 9223372036854775807, -9223372036854775808]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %2 = "stablehlo.and"(%0, %1) : (tensor<3xi64>, tensor<3xi64>) -> tensor<3xi64>
    "check.expect_eq_const"(%2) <{value = dense<[0, 0, -9223372036854775808]> : tensor<3xi64>}> : (tensor<3xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "and_op_test_ui64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 9223372036854775807, 18446744073709551615]> : tensor<3xui64>}> : () -> tensor<3xui64>
    %1 = "stablehlo.constant"() <{value = dense<18446744073709551615> : tensor<3xui64>}> : () -> tensor<3xui64>
    %2 = "stablehlo.and"(%0, %1) : (tensor<3xui64>, tensor<3xui64>) -> tensor<3xui64>
    "check.expect_eq_const"(%2) <{value = dense<[0, 9223372036854775807, 18446744073709551615]> : tensor<3xui64>}> : (tensor<3xui64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "and_op_test_i1"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[false, false, true, true]> : tensor<4xi1>}> : () -> tensor<4xi1>
    %1 = "stablehlo.constant"() <{value = dense<[false, true, false, true]> : tensor<4xi1>}> : () -> tensor<4xi1>
    %2 = "stablehlo.and"(%0, %1) : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    "check.expect_eq_const"(%2) <{value = dense<[false, false, false, true]> : tensor<4xi1>}> : (tensor<4xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "and_op_test_i1_splat_false"}> ({
    %0 = "stablehlo.constant"() <{value = dense<false> : tensor<2xi1>}> : () -> tensor<2xi1>
    %1 = "stablehlo.constant"() <{value = dense<[false, true]> : tensor<2xi1>}> : () -> tensor<2xi1>
    %2 = "stablehlo.and"(%0, %1) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
    "check.expect_eq_const"(%2) <{value = dense<false> : tensor<2xi1>}> : (tensor<2xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "and_op_test_i1_splat_true"}> ({
    %0 = "stablehlo.constant"() <{value = dense<true> : tensor<2xi1>}> : () -> tensor<2xi1>
    %1 = "stablehlo.constant"() <{value = dense<[false, true]> : tensor<2xi1>}> : () -> tensor<2xi1>
    %2 = "stablehlo.and"(%0, %1) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
    "check.expect_eq_const"(%2) <{value = dense<[false, true]> : tensor<2xi1>}> : (tensor<2xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

