"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x4x2xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<2x3x4xf32>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x4x2xf32>
    %4 = "stablehlo.transpose"(%2) <{permutation = array<i64: 1, 2, 0>}> : (tensor<2x3x4xf32>) -> tensor<3x4x2xf32>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x4x2xf32>, tensor<3x4x2xf32>) -> ()
    "func.return"(%4) : (tensor<3x4x2xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3x4xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[-3.80841613, -3.58268166, -3.428400e-01, 1.12658536], [-1.35501349, 5.86460495, -2.9754107, -2.14586616], [2.40116119, 1.97292423, 0.00829547829, -2.05573058]], [[0.500765204, -1.24809313, -0.254042923, 0.0727051944], [-3.1472609, 6.320640e-01, 1.86220384, 2.29526496], [0.630201399, 3.82720256, 0.950467467, 0.584882617]]]> : tensor<2x3x4xf32>}> : () -> tensor<2x3x4xf32>
    "func.return"(%1) : (tensor<2x3x4xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x4x2xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[-3.80841613, 0.500765204], [-3.58268166, -1.24809313], [-3.428400e-01, -0.254042923], [1.12658536, 0.0727051944]], [[-1.35501349, -3.1472609], [5.86460495, 6.320640e-01], [-2.9754107, 1.86220384], [-2.14586616, 2.29526496]], [[2.40116119, 0.630201399], [1.97292423, 3.82720256], [0.00829547829, 0.950467467], [-2.05573058, 0.584882617]]]> : tensor<3x4x2xf32>}> : () -> tensor<3x4x2xf32>
    "func.return"(%0) : (tensor<3x4x2xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

