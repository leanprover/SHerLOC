"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "iota_op_test_si4_dim_0"}> ({
    %0 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<3x4xi4>
    "check.expect_eq_const"(%0) <{value = dense<[[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]> : tensor<3x4xi4>}> : (tensor<3x4xi4>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "iota_op_test_si4_dim_1"}> ({
    %0 = "stablehlo.iota"() <{iota_dimension = 1 : i64}> : () -> tensor<3x4xi4>
    "check.expect_eq_const"(%0) <{value = dense<[[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]> : tensor<3x4xi4>}> : (tensor<3x4xi4>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "iota_op_test_si8_dim_0"}> ({
    %0 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<3x4xi8>
    "check.expect_eq_const"(%0) <{value = dense<[[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]> : tensor<3x4xi8>}> : (tensor<3x4xi8>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "iota_op_test_si8_dim_1"}> ({
    %0 = "stablehlo.iota"() <{iota_dimension = 1 : i64}> : () -> tensor<3x4xi8>
    "check.expect_eq_const"(%0) <{value = dense<[[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]> : tensor<3x4xi8>}> : (tensor<3x4xi8>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "iota_op_test_si16_dim_0"}> ({
    %0 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<3x4xi16>
    "check.expect_eq_const"(%0) <{value = dense<[[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]> : tensor<3x4xi16>}> : (tensor<3x4xi16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "iota_op_test_si16_dim_1"}> ({
    %0 = "stablehlo.iota"() <{iota_dimension = 1 : i64}> : () -> tensor<3x4xi16>
    "check.expect_eq_const"(%0) <{value = dense<[[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]> : tensor<3x4xi16>}> : (tensor<3x4xi16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "iota_op_test_si32_dim_0"}> ({
    %0 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<3x4xi32>
    "check.expect_eq_const"(%0) <{value = dense<[[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]> : tensor<3x4xi32>}> : (tensor<3x4xi32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "iota_op_test_si32_dim_1"}> ({
    %0 = "stablehlo.iota"() <{iota_dimension = 1 : i64}> : () -> tensor<3x4xi32>
    "check.expect_eq_const"(%0) <{value = dense<[[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]> : tensor<3x4xi32>}> : (tensor<3x4xi32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "iota_op_test_si64_dim_0"}> ({
    %0 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<3x4xi64>
    "check.expect_eq_const"(%0) <{value = dense<[[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]> : tensor<3x4xi64>}> : (tensor<3x4xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "iota_op_test_si64_dim_1"}> ({
    %0 = "stablehlo.iota"() <{iota_dimension = 1 : i64}> : () -> tensor<3x4xi64>
    "check.expect_eq_const"(%0) <{value = dense<[[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]> : tensor<3x4xi64>}> : (tensor<3x4xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "iota_op_test_ui64_dim_0"}> ({
    %0 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<2x3x2xui64>
    "check.expect_eq_const"(%0) <{value = dense<[[[0, 0], [0, 0], [0, 0]], [[1, 1], [1, 1], [1, 1]]]> : tensor<2x3x2xui64>}> : (tensor<2x3x2xui64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "iota_op_test_ui64_dim_1"}> ({
    %0 = "stablehlo.iota"() <{iota_dimension = 1 : i64}> : () -> tensor<2x3x2xui64>
    "check.expect_eq_const"(%0) <{value = dense<[[[0, 0], [1, 1], [2, 2]], [[0, 0], [1, 1], [2, 2]]]> : tensor<2x3x2xui64>}> : (tensor<2x3x2xui64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "iota_op_test_ui64_dim_2"}> ({
    %0 = "stablehlo.iota"() <{iota_dimension = 2 : i64}> : () -> tensor<2x3x2xui64>
    "check.expect_eq_const"(%0) <{value = dense<[[[0, 1], [0, 1], [0, 1]], [[0, 1], [0, 1], [0, 1]]]> : tensor<2x3x2xui64>}> : (tensor<2x3x2xui64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "iota_op_test_bf16_dim_0"}> ({
    %0 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<3x4xbf16>
    "check.expect_almost_eq_const"(%0) <{value = dense<[[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00], [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]]> : tensor<3x4xbf16>}> : (tensor<3x4xbf16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "iota_op_test_bf16_dim_1"}> ({
    %0 = "stablehlo.iota"() <{iota_dimension = 1 : i64}> : () -> tensor<3x4xbf16>
    "check.expect_almost_eq_const"(%0) <{value = dense<[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00], [0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00], [0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]]> : tensor<3x4xbf16>}> : (tensor<3x4xbf16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "iota_op_test_f16_dim_0"}> ({
    %0 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<3x4xf16>
    "check.expect_almost_eq_const"(%0) <{value = dense<[[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00], [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]]> : tensor<3x4xf16>}> : (tensor<3x4xf16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "iota_op_test_f16_dim_1"}> ({
    %0 = "stablehlo.iota"() <{iota_dimension = 1 : i64}> : () -> tensor<3x4xf16>
    "check.expect_almost_eq_const"(%0) <{value = dense<[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00], [0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00], [0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]]> : tensor<3x4xf16>}> : (tensor<3x4xf16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "iota_op_test_f32_dim_0"}> ({
    %0 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<3x4xf32>
    "check.expect_almost_eq_const"(%0) <{value = dense<[[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00], [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]]> : tensor<3x4xf32>}> : (tensor<3x4xf32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "iota_op_test_f32_dim_1"}> ({
    %0 = "stablehlo.iota"() <{iota_dimension = 1 : i64}> : () -> tensor<3x4xf32>
    "check.expect_almost_eq_const"(%0) <{value = dense<[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00], [0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00], [0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]]> : tensor<3x4xf32>}> : (tensor<3x4xf32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "iota_op_test_f64_dim_0"}> ({
    %0 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<3x4xf64>
    "check.expect_almost_eq_const"(%0) <{value = dense<[[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00], [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]]> : tensor<3x4xf64>}> : (tensor<3x4xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "iota_op_test_f64_dim_1"}> ({
    %0 = "stablehlo.iota"() <{iota_dimension = 1 : i64}> : () -> tensor<3x4xf64>
    "check.expect_almost_eq_const"(%0) <{value = dense<[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00], [0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00], [0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]]> : tensor<3x4xf64>}> : (tensor<3x4xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "iota_op_test_c64_dim_0"}> ({
    %0 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<3x4xcomplex<f32>>
    "check.expect_almost_eq_const"(%0) <{value = dense<[[(0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)], [(1.000000e+00,0.000000e+00), (1.000000e+00,0.000000e+00), (1.000000e+00,0.000000e+00), (1.000000e+00,0.000000e+00)], [(2.000000e+00,0.000000e+00), (2.000000e+00,0.000000e+00), (2.000000e+00,0.000000e+00), (2.000000e+00,0.000000e+00)]]> : tensor<3x4xcomplex<f32>>}> : (tensor<3x4xcomplex<f32>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "iota_op_test_c64_dim_1"}> ({
    %0 = "stablehlo.iota"() <{iota_dimension = 1 : i64}> : () -> tensor<3x4xcomplex<f32>>
    "check.expect_almost_eq_const"(%0) <{value = dense<[[(0.000000e+00,0.000000e+00), (1.000000e+00,0.000000e+00), (2.000000e+00,0.000000e+00), (3.000000e+00,0.000000e+00)], [(0.000000e+00,0.000000e+00), (1.000000e+00,0.000000e+00), (2.000000e+00,0.000000e+00), (3.000000e+00,0.000000e+00)], [(0.000000e+00,0.000000e+00), (1.000000e+00,0.000000e+00), (2.000000e+00,0.000000e+00), (3.000000e+00,0.000000e+00)]]> : tensor<3x4xcomplex<f32>>}> : (tensor<3x4xcomplex<f32>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "iota_op_test_c128_dim_0"}> ({
    %0 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<3x4xcomplex<f64>>
    "check.expect_almost_eq_const"(%0) <{value = dense<[[(0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)], [(1.000000e+00,0.000000e+00), (1.000000e+00,0.000000e+00), (1.000000e+00,0.000000e+00), (1.000000e+00,0.000000e+00)], [(2.000000e+00,0.000000e+00), (2.000000e+00,0.000000e+00), (2.000000e+00,0.000000e+00), (2.000000e+00,0.000000e+00)]]> : tensor<3x4xcomplex<f64>>}> : (tensor<3x4xcomplex<f64>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "iota_op_test_c128_dim_1"}> ({
    %0 = "stablehlo.iota"() <{iota_dimension = 1 : i64}> : () -> tensor<3x4xcomplex<f64>>
    "check.expect_almost_eq_const"(%0) <{value = dense<[[(0.000000e+00,0.000000e+00), (1.000000e+00,0.000000e+00), (2.000000e+00,0.000000e+00), (3.000000e+00,0.000000e+00)], [(0.000000e+00,0.000000e+00), (1.000000e+00,0.000000e+00), (2.000000e+00,0.000000e+00), (3.000000e+00,0.000000e+00)], [(0.000000e+00,0.000000e+00), (1.000000e+00,0.000000e+00), (2.000000e+00,0.000000e+00), (3.000000e+00,0.000000e+00)]]> : tensor<3x4xcomplex<f64>>}> : (tensor<3x4xcomplex<f64>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

